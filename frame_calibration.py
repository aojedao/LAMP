#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Frame calibration tool for the SO100 arm — rotation matrix edition.

You physically move the arm along each real-world axis (X, Y, Z) while the
script records raw FK position deltas.  It builds a full 3x3 rotation matrix
from those three delta vectors, handling any URDF frame orientation — not just
axis permutations.

Workflow:
  1. Connect, disable torque (arm is free to move by hand).
  2. For each real-world axis (X=forward, Y=left, Z=up):
       a. Return arm to home, press ENTER.
       b. Push arm CLEARLY along that direction, press ENTER.
       c. Script records the raw FK delta vector.
  3. Compute R_display_to_raw from the three delta vectors.
  4. Print:
       - The matrix to paste into all scripts.
       - Whether it is close to a clean permutation (for sanity checking).

Example:
    python frame_calibration.py \\
        --port /dev/ttyACM0 --id my_lamp \\
        --urdf-path /path/to/so100.urdf
"""

import argparse
import threading
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate SO100 real-world <-> URDF frame mapping via manual arm movement."
    )
    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--calibration-dir", default=None)
    p.add_argument("--urdf-path", required=True)
    p.add_argument(
        "--sample-rate", type=int, default=20,
        help="FK samples per second while streaming live position (default 20).",
    )
    p.add_argument(
        "--min-motion", type=float, default=0.01,
        help="Minimum net displacement (m) required to accept a probe (default 0.01 m).",
    )
    p.add_argument("--use-degrees", action="store_true", default=True)
    p.add_argument("--max-relative-target", type=float, default=10.0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_joints(robot: SO100Follower, motor_names: list) -> np.ndarray:
    obs_norm = robot.get_observation()
    return np.array(
        [float(obs_norm[f"{m}.pos"]) for m in motor_names if f"{m}.pos" in obs_norm],
        dtype=float,
    )


def fk_pos(kinematics: RobotKinematics, joints: np.ndarray) -> np.ndarray:
    t = kinematics.forward_kinematics(joints)
    return t[:3, 3].copy()


def stream_and_capture(
    robot: SO100Follower,
    kinematics: RobotKinematics,
    motor_names: list,
    sample_rate: int,
) -> tuple:
    """
    Capture start position, stream live FK delta, return (start_pos, end_pos).
    """
    entered = threading.Event()

    def wait_enter():
        input("")          # blank — prompt already printed before this call
        entered.set()

    joints_start = read_joints(robot, motor_names)
    start_pos = fk_pos(kinematics, joints_start)

    t = threading.Thread(target=wait_enter, daemon=True)
    t.start()

    print("  >> Move the arm now. Press ENTER when done. <<")
    while not entered.is_set():
        try:
            joints_now = read_joints(robot, motor_names)
        except Exception:
            joints_now = read_joints(robot, motor_names)
        pos_now = fk_pos(kinematics, joints_now)
        d = pos_now - start_pos
        print(
            f"\r  raw delta:  axis0={d[0]:+.4f}m  axis1={d[1]:+.4f}m  axis2={d[2]:+.4f}m   ",
            end="", flush=True,
        )
        precise_sleep(1.0 / sample_rate)

    print()
    joints_end = read_joints(robot, motor_names)
    end_pos = fk_pos(kinematics, joints_end)
    return start_pos, end_pos


def describe_matrix(R: np.ndarray) -> str:
    """
    Check if a 3x3 matrix is close to a permutation+sign matrix and describe it.
    """
    lines = []
    for col in range(3):
        best_row = int(np.argmax(np.abs(R[:, col])))
        val = R[best_row, col]
        frac = abs(val) / (np.linalg.norm(R[:, col]) + 1e-9)
        sign_str = "+" if val > 0 else "-"
        axis_label = ["raw[0]", "raw[1]", "raw[2]"][best_row]
        real_label = ["display_X", "display_Y", "display_Z"][col]
        lines.append(
            f"  {real_label} <- {sign_str}{axis_label}   "
            f"(dominant component {frac*100:.0f}%)"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None
    cfg = SO100FollowerConfig(
        port=args.port, id=args.id,
        calibration_dir=calibration_dir,
        use_degrees=args.use_degrees,
        max_relative_target=args.max_relative_target,
    )
    robot = SO100Follower(cfg)

    try:
        print(f"Connecting to {args.port}...")
        robot.connect(calibrate=False)
        motor_names = list(robot.bus.motors.keys())

        kinematics = RobotKinematics(
            urdf_path=args.urdf_path,
            target_frame_name="jaw",
            joint_names=motor_names,
        )

        robot.bus.disable_torque()
        print("Torque DISABLED — arm is free to move by hand.\n")

        home_joints = read_joints(robot, motor_names)
        home_raw = fk_pos(kinematics, home_joints)
        print(f"Home EE position (raw URDF): {home_raw}\n")

        print("=" * 60)
        print("MANUAL FRAME CALIBRATION  (rotation-matrix method)")
        print("=" * 60)
        print(
            "You will push the arm along THREE real-world axes one at a time.\n"
            "Stand BEHIND the robot:\n"
            "\n"
            "  Real +X = FORWARD  (away from base along table)\n"
            "  Real +Y = LEFT     (robot's left side)\n"
            "  Real +Z = UP\n"
            "\n"
            "The script records the full raw FK delta vector for each push.\n"
            "It does NOT care which raw axis dominates — it uses ALL three\n"
            "components to build a proper rotation matrix.\n"
            "\n"
            "Push at least 5-10 cm.  Return to HOME between probes.\n"
        )

        real_axis_names  = ["+X (FORWARD)", "+Y (LEFT)", "+Z (UP)"]
        real_instructions = [
            "Push gripper FORWARD (away from base)",
            "Push gripper to the LEFT (robot's left side)",
            "Push gripper UPWARD",
        ]

        delta_vectors = []   # one raw-3-vector per real axis

        for real_i in range(3):
            print(f"\n{'─' * 55}")
            print(f"PROBE {real_i + 1}/3  —  Real-world {real_axis_names[real_i]}")
            print(f"  {real_instructions[real_i]}, then press ENTER.")

            while True:
                input("\n  Return arm to HOME. Press ENTER when ready...")
                start_pos, end_pos = stream_and_capture(
                    robot, kinematics, motor_names, args.sample_rate
                )

                delta = end_pos - start_pos
                net = float(np.linalg.norm(delta))

                print(f"\n  Captured delta: {delta}")
                print(f"  Net displacement: {net:.4f} m")

                if net < args.min_motion:
                    print(f"  Too small (< {args.min_motion} m). Move more and retry.")
                    if input("  Retry? (y/n): ").strip().lower() != "n":
                        continue

                delta_vectors.append(delta.copy())
                print(f"  Accepted for real-world {real_axis_names[real_i]}.")
                break

        # ── Build rotation matrix ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("COMPUTING TRANSFORM")
        print("=" * 60)

        # Each delta_vectors[i] is a raw-space vector pointing in real direction i.
        # We want: raw = R_d2r @ display
        # So columns of R_d2r are the raw-space unit vectors for each display axis.
        R_d2r = np.column_stack([v / np.linalg.norm(v) for v in delta_vectors])

        # raw_to_display: display = R_d2r.T @ raw  (inverse = transpose for rotation)
        R_r2d = R_d2r.T

        print("\nR_display_to_raw  (columns = raw-space unit vectors for X, Y, Z):")
        print(np.array2string(R_d2r, precision=4, suppress_small=True))

        print("\nR_raw_to_display  (= R_display_to_raw.T):")
        print(np.array2string(R_r2d, precision=4, suppress_small=True))

        cond = np.linalg.cond(R_d2r)
        print(f"\nMatrix condition number: {cond:.2f}  "
              f"({'good' if cond < 3 else 'high — probes may not be orthogonal; try again'})")

        print("\nDominant-component interpretation:")
        print(describe_matrix(R_d2r))

        # ── Print paste-ready code ────────────────────────────────────────────
        def fmt_matrix(M: np.ndarray) -> str:
            rows = []
            for row in M:
                vals = ", ".join(f"{v:+.6f}" for v in row)
                rows.append(f"    [{vals}]")
            return "np.array([\n" + ",\n".join(rows) + "\n])"

        print("\n" + "─" * 60)
        print("PASTE THESE into move_to_3d_location.py, axis_testing.py,")
        print("and current_location_mapping.py  (replace the old transform lines):\n")

        print("# Near top of file, after imports:")
        print(f"FRAME_R_DISPLAY_TO_RAW = {fmt_matrix(R_d2r)}")
        print(f"FRAME_R_RAW_TO_DISPLAY = {fmt_matrix(R_r2d)}")
        print()
        print("# raw URDF -> display frame  (after forward_kinematics)")
        print("ee_pos = FRAME_R_RAW_TO_DISPLAY @ raw")
        print()
        print("# display frame -> raw URDF  (before inverse_kinematics target)")
        print("target_ik = FRAME_R_DISPLAY_TO_RAW @ display")
        print("─" * 60)

        # Also print the compact raw array literals for easy grepping
        print("\nNumerical arrays (copy-paste values):")
        print(f"FRAME_R_DISPLAY_TO_RAW = np.array({R_d2r.tolist()})")
        print(f"FRAME_R_RAW_TO_DISPLAY = np.array({R_r2d.tolist()})")

    finally:
        if robot.is_connected:
            try:
                robot.bus.enable_torque()
            except Exception:
                pass
            robot.disconnect()
            print("\nRobot disconnected.")


if __name__ == "__main__":
    main()
