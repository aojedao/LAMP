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
Real-time end-effector position mapping for SO100 arm.

Continuously displays the current 3D position and orientation of the end-effector
using forward kinematics. Useful for understanding the arm's state and calibration.

Example:
    python current_location_mapping.py --port /dev/ttyACM0 --id my_lamp --fps 10 --urdf-path /home/aojedao/Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time end-effector position mapping using forward kinematics."
    )
    parser.add_argument("--port", required=True, help="Serial port of the SO100 arm.")
    parser.add_argument("--id", required=True, help="Robot calibration id.")
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Optional calibration directory. Defaults to ~/.cache/huggingface/lerobot/calibration/robots/so_follower.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Update frequency in Hz.",
    )
    parser.add_argument(
        "--urdf-path",
        default=None,
        help=(
            "Path to SO100 URDF file. If not provided, tries to find it in common locations "
            "or uses placo's built-in model."
        ),
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        default=True,
        help="Use degree normalization mode (default: True).",
    )
    parser.add_argument(
        "--format",
        choices=["compact", "detailed"],
        default="compact",
        help="Output format: 'compact' for single line, 'detailed' for full information.",
    )
    return parser.parse_args()


def find_urdf_path() -> str | None:
    """Try to find the SO100 URDF file in common locations."""
    possible_paths = [
        Path.home() / "SO-ARM100-main" / "Simulation" / "SO100" / "so100.urdf",
        Path.home() / "SO-ARM100-main" / "Simulation" / "SO100" / "so100_calib.urdf",
        Path.home() / "SO-ARM100-main" / "Simulation" / "SO101" / "so101.urdf",
        Path.home() / "SO-ARM100-main" / "Simulation" / "SO101" / "so101_new_calib.urdf",
        Path("/opt/placo/models/so100/so100.urdf"),
        Path("/opt/placo/models/so101/so101.urdf"),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Found URDF: {path}")
            return str(path)

    print("Warning: Could not find URDF file. Placo may have a built-in model.")
    return None


def print_pose_compact(step: int, pos: np.ndarray, rot: np.ndarray, joint_pos: np.ndarray | None = None) -> None:
    """Print pose in compact single-line format."""
    # Convert rotation matrix to Euler angles (simplified representation)
    output = f"[Step {step:4d}] EE Pos: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}] m"
    if joint_pos is not None:
        output += f" | Joints(°): [{joint_pos[0]:6.1f}, {joint_pos[1]:6.1f}, {joint_pos[2]:6.1f}, {joint_pos[3]:6.1f}, {joint_pos[4]:6.1f}]"
    print(output)


def print_pose_detailed(step: int, pos: np.ndarray, rot: np.ndarray, joint_pos: np.ndarray) -> None:
    """Print pose in detailed format with full information."""
    print("\n" + "="*80)
    print(f"Step {step}")
    print("="*80)
    print(f"End-Effector Position (meters):")
    print(f"  X: {pos[0]:8.4f}")
    print(f"  Y: {pos[1]:8.4f}")
    print(f"  Z: {pos[2]:8.4f}")
    print(f"\nEnd-Effector Orientation (rotation matrix):")
    for i, row in enumerate(rot):
        print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
    print(f"\nJoint Positions (degrees):")
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for name, angle in zip(joint_names, joint_pos):
        print(f"  {name:15s}: {angle:8.2f}°")
    print("="*80)


def main() -> None:
    args = parse_args()

    # Find or use provided URDF path
    urdf_path = args.urdf_path or find_urdf_path()
    if not urdf_path:
        print(
            "Warning: No URDF path found. Attempting to use placo's built-in SO100 model. "
            "If this fails, provide --urdf-path."
        )
        urdf_path = "so100"

    print(f"Using URDF: {urdf_path}")

    # Setup calibration directory
    calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None

    # Create robot config and instance
    cfg = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        calibration_dir=calibration_dir,
        use_degrees=args.use_degrees,
    )
    robot = SO100Follower(cfg)

    try:
        # Connect to robot
        print(f"Connecting to robot on {args.port}...")
        robot.connect(calibrate=False)
        print("Robot connected.")
        
        # Disable torque on all motors so they can be moved manually
        print("Disabling motor torque to allow manual movement...")
        robot.bus.disable_torque()
        print("Motors disabled - you can now move them manually.")
        
        # Initialize kinematics solver
        print("Initializing kinematics solver...")
        try:
            all_joint_names = list(robot.bus.motors.keys())
            print(f"Using all joints for kinematics: {all_joint_names}")
            kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="jaw",
                joint_names=all_joint_names,
            )
        except Exception as e:
            print(f"Error initializing kinematics: {e}")
            print("Make sure placo is installed: pip install placo")
            raise

        print("\n" + "="*80)
        print("Real-Time End-Effector Position Mapping")
        print(f"Update frequency: {args.fps} Hz")
        print(f"Format: {args.format}")
        print("Press Ctrl+C to stop.")
        print("="*80 + "\n")

        motor_names = list(robot.bus.motors.keys())
        step = 0

        try:
            while True:
                # Read joint positions directly from motor bus (allows manual movement)
                present_pos = robot.bus.sync_read("Present_Position")
                joint_pos = np.array(
                    [float(present_pos[motor]) for motor in motor_names],
                    dtype=float,
                )

                # Get end-effector pose via forward kinematics
                t_ee = kinematics.forward_kinematics(joint_pos)
                ee_pos_raw = t_ee[:3, 3]
                ee_rot = t_ee[:3, :3]
                
                # Transform coordinate frame: raw URDF -> display (X=forward, Y=left, Z=up)
                ee_pos = FRAME_R_RAW_TO_DISPLAY @ ee_pos_raw

                # Print based on format
                if args.format == "compact":
                    print_pose_compact(step, ee_pos, ee_rot, joint_pos[:-1])  # Exclude gripper from display
                else:  # detailed
                    print_pose_detailed(step, ee_pos, ee_rot, joint_pos)

                step += 1
                precise_sleep(1.0 / args.fps)

        except KeyboardInterrupt:
            print("\n\nCtrl+C detected. Stopping...")

    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
