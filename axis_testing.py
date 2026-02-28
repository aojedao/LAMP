#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Axis testing for the SO100 arm.

Moves the end-effector along a single axis (X, Y, or Z) between two offsets
relative to the initial (home) position. The robot:
  1. Reads current joint positions and saves them as the home position.
  2. Moves to the start of the axis range.
  3. Sweeps to the end of the axis range.
  4. Returns to the home position.

Ctrl+C at any time triggers an immediate return to home.

Example:
    python axis_testing.py --port /dev/ttyACM0 --id my_lamp --urdf-path /home/aojedao/Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf --axis x --range-start -0.1 --range-end 0.1 --duration 4.0
"""

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

# Calibrated frame transform matrices — Gram-Schmidt orthogonalized (Z fixed, X/Y decoupled)
FRAME_R_DISPLAY_TO_RAW = np.array([
    [-0.2702471023729073, -0.8897091717756613,  0.367945774968696   ],
    [-0.5045256680845093,  0.4563583697590681,  0.7329330723843245  ],
    [ 0.8200124108224991, -0.012434951110182071, 0.5722106413620428 ],
])
FRAME_R_RAW_TO_DISPLAY = np.array([
    [-0.2702471023729073, -0.5045256680845093,  0.8200124108224991  ],
    [-0.8897091717756613,  0.4563583697590681, -0.012434951110182071],
    [ 0.367945774968696,   0.7329330723843245,  0.5722106413620428  ],
])

# ---------------------------------------------------------------------------
# Global home position (set once at startup)
# ---------------------------------------------------------------------------
_HOME_JOINTS: np.ndarray | None = None
_HOME_EE_POS: np.ndarray | None = None  # in display frame [X, Y, Z]


def set_home(joints: np.ndarray, kinematics: RobotKinematics) -> None:
    global _HOME_JOINTS, _HOME_EE_POS
    _HOME_JOINTS = joints.copy()
    t = kinematics.forward_kinematics(_HOME_JOINTS)
    raw = t[:3, 3]
    # Display frame: raw URDF -> display (X=forward, Y=left, Z=up)
    _HOME_EE_POS = FRAME_R_RAW_TO_DISPLAY @ raw
    print(f"Home joints saved     : {_HOME_JOINTS}")
    print(f"Home EE position (XYZ): {_HOME_EE_POS}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_urdf_path() -> str | None:
    possible_paths = [
        Path.home() / "SO-ARM100-main" / "Simulation" / "SO100" / "so100.urdf",
        Path.home() / "SO-ARM100-main" / "Simulation" / "SO101" / "so101.urdf",
    ]
    for p in possible_paths:
        if p.exists():
            print(f"Found URDF: {p}")
            return str(p)
    return None


def create_trajectory(start: np.ndarray, end: np.ndarray, num_points: int) -> np.ndarray:
    return np.linspace(start, end, max(num_points, 2))


def run_trajectory(
    robot: SO100Follower,
    kinematics: RobotKinematics,
    trajectory: np.ndarray,
    current_joints: np.ndarray,
    motor_names: list[str],
    ee_rot: np.ndarray,
    fps: int,
    position_weight: float,
    label: str = "",
) -> np.ndarray:
    """
    Execute a Cartesian trajectory via IK, return final joint positions.
    Raises KeyboardInterrupt if Ctrl+C is pressed mid-trajectory.
    """
    num_points = len(trajectory)
    start_time = time.perf_counter()

    for i, step_pos in enumerate(trajectory):
        elapsed = time.perf_counter() - start_time

        # Transform display frame → IK frame: display -> raw URDF
        step_pos_ik = FRAME_R_DISPLAY_TO_RAW @ step_pos

        step_pose = np.eye(4, dtype=float)
        step_pose[:3, :3] = ee_rot
        step_pose[:3, 3] = step_pos_ik

        try:
            ik_solution = kinematics.inverse_kinematics(
                current_joints,
                step_pose,
                position_weight=position_weight,
                orientation_weight=0.01,
            )

            # Singularity check
            joint_delta = np.abs(ik_solution - current_joints)
            max_delta = float(np.max(joint_delta))
            if max_delta > 45.0:
                print(
                    f"⚠️  SINGULARITY WARNING: joint jump {max_delta:.1f}° at step {i + 1}. "
                    "Target may be near singularity or workspace boundary."
                )

            # Command only first 4 motors; keep last 2 fixed
            arm_joints = ik_solution[: len(motor_names) - 2]
            action = {f"{motor}.pos": float(arm_joints[j]) for j, motor in enumerate(motor_names[:-2])}
            action[motor_names[-2] + ".pos"] = float(current_joints[-2])
            action[motor_names[-1] + ".pos"] = float(current_joints[-1])
            robot.send_action(action)
            current_joints = ik_solution

            print(
                f"{label}Step {i + 1:3d}/{num_points} | t={elapsed:5.2f}s | "
                f"EE: [{step_pos[0]:7.3f}, {step_pos[1]:7.3f}, {step_pos[2]:7.3f}]"
            )

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Warning: IK error at step {i}: {e}")
            if i > 0:
                print("Continuing with previous solution...")
            else:
                raise

        precise_sleep(1.0 / fps)

    return current_joints


def return_to_home_joints(
    robot: SO100Follower,
    current_joints: np.ndarray,
    motor_names: list[str],
    fps: int,
    return_duration: float,
) -> None:
    """Interpolate in joint space back to the saved starting joint configuration."""
    if _HOME_JOINTS is None:
        print("No home joints saved, cannot return home.")
        return

    num_points = max(int(return_duration * fps), 2)
    print(f"\nReturning to start configuration over {return_duration:.1f}s ({num_points} steps)...")
    traj = np.linspace(current_joints, _HOME_JOINTS, num_points)
    try:
        for i, target_joints in enumerate(traj):
            arm_joints = target_joints[: len(motor_names) - 2]
            action = {f"{motor}.pos": float(arm_joints[j]) for j, motor in enumerate(motor_names[:-2])}
            action[motor_names[-2] + ".pos"] = float(_HOME_JOINTS[-2])
            action[motor_names[-1] + ".pos"] = float(_HOME_JOINTS[-1])
            robot.send_action(action)
            print(
                f"[HOME] Step {i + 1:3d}/{num_points} | joints: {np.round(target_joints[:4], 1)}",
                end="\r",
            )
            precise_sleep(1.0 / fps)
    except KeyboardInterrupt:
        print("\nForce-stopped during home return.")
    print("\nReached start configuration.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move SO100 arm along a single axis to test workspace reachability."
    )
    parser.add_argument("--port", required=True, help="Serial port of the SO100 arm.")
    parser.add_argument("--id", required=True, help="Robot calibration id.")
    parser.add_argument(
        "--calibration-dir", default=None,
        help="Optional calibration directory.",
    )
    parser.add_argument(
        "--axis", choices=["x", "y", "z"], default="x",
        help="Axis to sweep along: x, y, or z (default: x).",
    )
    parser.add_argument(
        "--range-start", type=float, default=-0.1,
        help="Start offset (meters) relative to home EE position (default: -0.1).",
    )
    parser.add_argument(
        "--range-end", type=float, default=0.1,
        help="End offset (meters) relative to home EE position (default: 0.1).",
    )
    parser.add_argument(
        "--sweeps", type=int, default=1,
        help="Number of full range-start→range-end sweeps to perform (default: 1).",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Duration of each single sweep in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--return-duration", type=float, default=5.0,
        help="Duration of the return-to-home move in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Control loop frequency (default: 15).",
    )
    parser.add_argument(
        "--urdf-path", default=None,
        help="Path to SO100 URDF file.",
    )
    parser.add_argument(
        "--use-degrees", action="store_true", default=True,
        help="Use degree normalization mode (default: True).",
    )
    parser.add_argument(
        "--max-relative-target", type=float, default=10.0,
        help="Safety cap for per-step joint movement (default: 10.0).",
    )
    parser.add_argument(
        "--position-weight", type=float, default=10.0,
        help="IK position accuracy weight (default: 10.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    urdf_path = args.urdf_path or find_urdf_path()
    if not urdf_path:
        print("Warning: No URDF path found. Provide --urdf-path.")
        raise SystemExit(1)
    print(f"Using URDF: {urdf_path}")

    calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None

    cfg = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        calibration_dir=calibration_dir,
        use_degrees=args.use_degrees,
        max_relative_target=args.max_relative_target,
    )
    robot = SO100Follower(cfg)

    try:
        print(f"Connecting to robot on {args.port}...")
        robot.connect(calibrate=False)
        print("Robot connected.")

        # Read current joints and lock them in as home
        initial_obs = robot.get_observation()
        motor_names = list(robot.bus.motors.keys())
        initial_joints = np.array(
            [float(initial_obs[f"{motor}.pos"]) for motor in motor_names if f"{motor}.pos" in initial_obs],
            dtype=float,
        )

        # Initialise kinematics
        print("Initialising kinematics solver...")
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="jaw",
            joint_names=motor_names,
        )

        # Save home position (uses FK, so kinematics must be ready first)
        set_home(initial_joints, kinematics)
        home_ee_pos = _HOME_EE_POS.copy()

        # Initial EE orientation (maintained throughout)
        t0 = kinematics.forward_kinematics(initial_joints)
        ee_rot = t0[:3, :3].copy()

        # Map axis label to index in display frame [X, Y, Z]
        axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]

        # Build sweep waypoints (all offsets relative to home EE position)
        sweep_start_pos = home_ee_pos.copy()
        sweep_start_pos[axis_index] += args.range_start

        sweep_end_pos = home_ee_pos.copy()
        sweep_end_pos[axis_index] += args.range_end

        print(f"\n{'='*60}")
        print(f"Axis test: {args.axis.upper()}  |  range [{args.range_start:+.3f} → {args.range_end:+.3f}] m")
        print(f"Sweeps: {args.sweeps}  |  {args.duration:.1f}s per sweep  |  {args.fps} fps")
        print(f"Home EE     : {home_ee_pos}")
        print(f"Sweep start : {sweep_start_pos}")
        print(f"Sweep end   : {sweep_end_pos}")
        print(f"{'='*60}\n")

        current_joints = initial_joints.copy()

        # Compute current EE pos (display frame) from latest joints
        def current_ee(joints: np.ndarray) -> np.ndarray:
            t = kinematics.forward_kinematics(joints)
            raw = t[:3, 3]
            return FRAME_R_RAW_TO_DISPLAY @ raw

        try:
            # --- Phase 1: move from home to sweep start ---
            print("[Phase 1] Moving to sweep start...")
            traj_to_start = create_trajectory(home_ee_pos, sweep_start_pos, max(int(args.duration * args.fps / 2), 10))
            current_joints = run_trajectory(
                robot, kinematics, traj_to_start, current_joints, motor_names,
                ee_rot, args.fps, args.position_weight, "[TO START] ",
            )

            # --- Phase 2: perform sweeps ---
            for sweep_i in range(args.sweeps):
                print(f"\n[Sweep {sweep_i + 1}/{args.sweeps}] {args.axis.upper()}: {args.range_start:+.3f} → {args.range_end:+.3f} m")
                num_pts = max(int(args.duration * args.fps), 2)
                traj_sweep = create_trajectory(sweep_start_pos, sweep_end_pos, num_pts)
                current_joints = run_trajectory(
                    robot, kinematics, traj_sweep, current_joints, motor_names,
                    ee_rot, args.fps, args.position_weight, f"[SWEEP {sweep_i + 1}] ",
                )

                # If more sweeps, reverse direction for the next one
                if sweep_i + 1 < args.sweeps:
                    sweep_start_pos, sweep_end_pos = sweep_end_pos, sweep_start_pos

            print("\nAxis sweep completed!")

        except KeyboardInterrupt:
            print("\nCtrl+C detected during axis sweep.")

        # --- Phase 3: always return to starting joint configuration ---
        return_to_home_joints(
            robot, current_joints, motor_names, args.fps, args.return_duration,
        )

    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
