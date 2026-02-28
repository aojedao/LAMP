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
Move the SO100 arm to a 3D location relative to its initial position using inverse kinematics.

This module provides a function to smoothly move the arm's end-effector to a target 3D position
using linear interpolation for smooth trajectory planning and inverse kinematics for joint control.

Example:
    python move_to_3d_location.py --port /dev/ttyACM0 --id my_lamp --target-x 0.1  --target-y 0.05  --target-z -0.1 --duration 5.0  --fps 10 --urdf-path so100.urdf
"""

import argparse
import sys
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


# Global variable to store the home joint positions captured at startup
_HOME_JOINTS: np.ndarray | None = None


def set_home_position(joints: np.ndarray) -> None:
    """Save the robot's initial joint positions as the global home position."""
    global _HOME_JOINTS
    _HOME_JOINTS = joints.copy()
    print(f"Home position saved: {_HOME_JOINTS}")


def get_home_position() -> np.ndarray | None:
    """Return the saved home joint positions."""
    return _HOME_JOINTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move SO100 arm to a 3D location using inverse kinematics."
    )
    parser.add_argument("--port", required=True, help="Serial port of the SO100 arm.")
    parser.add_argument("--id", required=True, help="Robot calibration id.")
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Optional calibration directory. Defaults to ~/.cache/huggingface/lerobot/calibration/robots/so_follower.",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=0.0,
        help="Target X position relative to initial EE position (meters).",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=0.0,
        help="Target Y position relative to initial EE position (meters).",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=0.0,
        help="Target Z position relative to initial EE position (meters).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Time to reach target in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Control loop frequency.",
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
        "--max-relative-target",
        type=float,
        default=10.0,
        help="Safety cap for per-step joint movement.",
    )
    parser.add_argument(
        "--end-behavior",
        choices=["hold", "home"],
        default="hold",
        help="Behavior after reaching target: 'hold' waits for Ctrl+C, 'home' returns to initial position.",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=10.0,
        help="IK position accuracy weight (higher = more accurate to target position, lower = softer constraint).",
    )
    parser.add_argument(
        "--return-duration",
        type=float,
        default=10.0,
        help="Time to return home in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.05,
        help="Maximum acceptable position error in meters (default: 0.05m).",
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


def create_smooth_trajectory(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    Create a smooth linear trajectory from start to end position.

    Args:
        start_pos: Starting 3D position [x, y, z]
        end_pos: Target 3D position [x, y, z]
        num_points: Number of trajectory points to generate

    Returns:
        numpy array of shape (num_points, 3) with trajectory positions
    """
    trajectory = np.linspace(start_pos, end_pos, num_points)
    return trajectory


def move_to_3d_location(
    robot: SO100Follower,
    kinematics: RobotKinematics,
    target_pos: np.ndarray,
    initial_joints: np.ndarray | None = None,
    home_joints: np.ndarray | None = None,
    duration: float = 5.0,
    fps: int = 10,
    end_behavior: str = "hold",
    position_weight: float = 10.0,
    return_duration: float = 10.0,
    position_tolerance: float = 0.05,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Move the SO100 arm end-effector to a target 3D location.

    This function:
    1. Gets the current end-effector pose from forward kinematics
    2. Generates a smooth linear trajectory to the target position
    3. Uses inverse kinematics to compute joint angles for each trajectory point
    4. Sends joint commands slowly to the arm
    5. Optionally returns to home position or holds at target

    Args:
        robot: Connected SO100Follower instance
        kinematics: RobotKinematics instance with IK solver initialized
        target_pos: Target 3D position relative to initial EE pose [x, y, z]
        initial_joints: Initial joint positions (required for 'home' end behavior)
        home_joints: Home joint positions (default middle of calibration range)
        duration: Time to reach target in seconds (default: 5.0)
        fps: Control loop frequency (default: 10)
        end_behavior: 'hold' to wait for Ctrl+C, 'home' to return to initial position

    Returns:
        tuple: (final_joint_positions, final_action_dict) for holding or returning home

    Raises:
        ValueError: If robot is not connected or if IK fails consistently
    """

    if not robot.is_connected:
        raise ValueError("Robot is not connected. Call robot.connect() first.")

    # Get current observation and joint positions
    obs = robot.get_observation()
    motor_names = list(robot.bus.motors.keys())

    # Extract current joint positions in degrees
    current_joints = np.array(
        [float(obs[f"{motor}.pos"]) for motor in motor_names if f"{motor}.pos" in obs],
        dtype=float,
    )

    # Get current end-effector pose via forward kinematics
    t_current = kinematics.forward_kinematics(current_joints)
    current_ee_pos_raw = t_current[:3, 3]
    current_ee_rot = t_current[:3, :3]
    
    # Transform coordinate frame: raw URDF -> display (X=forward, Y=left, Z=up)
    current_ee_pos = FRAME_R_RAW_TO_DISPLAY @ current_ee_pos_raw
    initial_ee_pos = current_ee_pos.copy()  # Store initial position for home return

    print(f"Current EE position: {current_ee_pos}")
    print(f"Target offset: {target_pos}")

    # Compute absolute target position
    target_ee_pos = current_ee_pos + target_pos
    print(f"Absolute target EE position: {target_ee_pos}")
    
    # Transform target back to original frame for IK: X->Z, Z->X, Y unchanged
    target_ee_pos_ik = np.array([target_ee_pos[2], target_ee_pos[1], target_ee_pos[0]])

    # Create target pose (maintain current orientation) - use IK frame
    target_pose = np.eye(4, dtype=float)
    target_pose[:3, :3] = current_ee_rot
    target_pose[:3, 3] = target_ee_pos_ik

    # Generate smooth trajectory in display frame (transformed)
    num_points = max(int(duration * fps), 2)
    trajectory = create_smooth_trajectory(current_ee_pos, target_ee_pos, num_points)

    print(f"Generated trajectory with {num_points} points over {duration:.1f}s")

    # Send trajectory commands
    start_time = time.perf_counter()
    gripper_pos = float(obs["gripper.pos"]) if "gripper.pos" in obs else 0.0

    for i, target_step_pos in enumerate(trajectory):
        elapsed = time.perf_counter() - start_time

        # Transform step position back to IK frame: display -> raw URDF
        target_step_pos_ik = FRAME_R_DISPLAY_TO_RAW @ target_step_pos

        # Create target pose for this step
        step_pose = np.eye(4, dtype=float)
        step_pose[:3, :3] = current_ee_rot  # Maintain orientation
        step_pose[:3, 3] = target_step_pos_ik

        # Compute inverse kinematics
        try:
            # Use current joint positions as initial guess for IK
            ik_solution = kinematics.inverse_kinematics(
                current_joints,
                step_pose,
                position_weight=position_weight,
                orientation_weight=0.01,  # Soft orientation constraint
            )

            # Extract only first 4 arm joint angles (exclude last 2 motors: wrist_roll and gripper)
            arm_joints = ik_solution[: len(motor_names) - 2]  # Exclude last 2 motors
            action = {f"{motor}.pos": float(arm_joints[j]) for j, motor in enumerate(motor_names[:-2])}
            # Keep last 2 motors at their current positions
            action[motor_names[-2] + ".pos"] = float(current_joints[-2])
            action[motor_names[-1] + ".pos"] = float(current_joints[-1])

            # Check for near-singularity: large joint angle jump means IK jumped branches
            joint_delta = np.abs(ik_solution - current_joints)
            max_delta_deg = float(np.max(joint_delta))
            if max_delta_deg > 45.0:
                print(
                    f"⚠️  SINGULARITY WARNING: Large joint jump ({max_delta_deg:.1f}°) at step {i + 1}. "
                    "Arm may be near a singularity or target is outside smooth reach."
                )

            # Send action to robot
            robot.send_action(action)

            # Update current_joints for next iteration's IK guess
            current_joints = ik_solution

            print(
                f"Step {i + 1}/{num_points} | t={elapsed:5.2f}s | "
                f"Target: [{target_step_pos[0]:7.3f}, {target_step_pos[1]:7.3f}, {target_step_pos[2]:7.3f}]"
            )

        except Exception as e:
            print(f"Warning: IK solver error at step {i}: {e}")
            # Continue with previous solution or stop
            if i > 0:
                print("Continuing with previous joint solution...")
            else:
                print("Stopping due to initial IK failure.")
                raise

        # Sleep to maintain desired FPS
        precise_sleep(1.0 / fps)

    print("\nTrajectory completed successfully!")
    print(f"Target EE position: {target_ee_pos}")
    
    # Check position accuracy
    t_final = kinematics.forward_kinematics(current_joints)
    final_ee_pos_raw = t_final[:3, 3]
    final_ee_pos = FRAME_R_RAW_TO_DISPLAY @ final_ee_pos_raw
    position_error = np.linalg.norm(final_ee_pos - target_ee_pos)
    print(f"Final EE position:  {final_ee_pos}")
    print(f"Position error: {position_error:.4f}m (tolerance: {position_tolerance:.4f}m)")
    
    if position_error > position_tolerance:
        print(f"⚠️  WARNING: Position error ({position_error:.4f}m) exceeds tolerance ({position_tolerance:.4f}m)")
    else:
        print(f"✓ Position within tolerance")
    
    # Handle end behavior
    final_action = {f"{motor}.pos": float(current_joints[j]) for j, motor in enumerate(motor_names[:-2])}
    # Keep last 2 motors at their current positions
    final_action[motor_names[-2] + ".pos"] = float(current_joints[-2])
    final_action[motor_names[-1] + ".pos"] = float(current_joints[-1])
    
    if end_behavior == "home" and home_joints is not None:
        print("\nReturning to home position...")
        # Compute actual final EE position from current joints (after the forward trajectory)
        t_final_fk = kinematics.forward_kinematics(current_joints)
        final_ee_pos_for_return_raw = t_final_fk[:3, 3]
        final_ee_pos_for_return = FRAME_R_RAW_TO_DISPLAY @ final_ee_pos_for_return_raw
        
        # Calculate end-effector position at home joint angles
        t_home = kinematics.forward_kinematics(home_joints)
        home_ee_pos_raw = t_home[:3, 3]
        home_ee_pos = FRAME_R_RAW_TO_DISPLAY @ home_ee_pos_raw
        
        return_num_points = max(int(return_duration * fps), 2)
        return_trajectory = create_smooth_trajectory(final_ee_pos_for_return, home_ee_pos, return_num_points)
        
        try:
            for i, return_step_pos in enumerate(return_trajectory):
                # Transform step position back to IK frame: display -> raw URDF
                return_step_pos_ik = FRAME_R_DISPLAY_TO_RAW @ return_step_pos
                
                step_pose = np.eye(4, dtype=float)
                step_pose[:3, :3] = current_ee_rot
                step_pose[:3, 3] = return_step_pos_ik
                
                try:
                    ik_solution = kinematics.inverse_kinematics(
                        current_joints,
                        step_pose,
                        position_weight=position_weight,
                        orientation_weight=0.01,
                    )
                    arm_joints = ik_solution[: len(motor_names) - 2]  # Exclude last 2 motors
                    return_action = {f"{motor}.pos": float(arm_joints[j]) for j, motor in enumerate(motor_names[:-2])}
                    # Keep last 2 motors at their current positions
                    return_action[motor_names[-2] + ".pos"] = float(current_joints[-2])
                    return_action[motor_names[-1] + ".pos"] = float(current_joints[-1])
                    robot.send_action(return_action)
                    current_joints = ik_solution
                except Exception as e:
                    print(f"Warning: IK error during return: {e}")
                    if i > 0:
                        print("Continuing with previous solution...")
                    else:
                        raise
                
                precise_sleep(1.0 / fps)
        except KeyboardInterrupt:
            print("\nForce-stopped during return to home. Disconnecting now...")
        
        print("Returned to home position.")
        final_action = {f"{motor}.pos": float(current_joints[j]) for j, motor in enumerate(motor_names[:-2])}
        final_action[motor_names[-2] + ".pos"] = float(current_joints[-2])
        final_action[motor_names[-1] + ".pos"] = float(current_joints[-1])
    
    return current_joints, final_action, current_ee_pos, initial_ee_pos, current_ee_rot, motor_names


def main() -> None:
    args = parse_args()

    # Find or use provided URDF path
    urdf_path = args.urdf_path or find_urdf_path()
    if not urdf_path:
        print(
            "Warning: No URDF path found. Attempting to use placo's built-in SO100 model. "
            "If this fails, provide --urdf-path."
        )
        urdf_path = "so100"  # Placo might have built-in model

    print(f"Using URDF: {urdf_path}")

    # Setup calibration directory
    calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None

    # Create robot config and instance
    cfg = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        calibration_dir=calibration_dir,
        use_degrees=args.use_degrees,
        max_relative_target=args.max_relative_target,
    )
    robot = SO100Follower(cfg)

    try:
        # Connect to robot
        print(f"Connecting to robot on {args.port}...")
        robot.connect(calibrate=False)
        print("Robot connected.")
        
        # Get initial joint positions
        initial_obs = robot.get_observation()
        motor_names = list(robot.bus.motors.keys())
        initial_joints = np.array(
            [float(initial_obs[f"{motor}.pos"]) for motor in motor_names if f"{motor}.pos" in initial_obs],
            dtype=float,
        )
        print(f"Initial joint positions: {initial_joints}")
        
        # Save initial joint positions as the global home position
        set_home_position(initial_joints)
        home_joints = get_home_position()
        print(f"Home position locked in as: {home_joints}")

        # Initialize kinematics solver
        print("Initializing kinematics solver...")
        try:
            kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="jaw",
                joint_names=list(robot.bus.motors.keys()),  # Include all joints including gripper
            )
        except Exception as e:
            print(f"Error initializing kinematics: {e}")
            print("Make sure placo is installed: pip install placo")
            raise

        # Target position relative to initial EE pose
        target_pos = np.array([args.target_x, args.target_y, args.target_z], dtype=float)

        # Move to target location
        final_joints, final_action, current_ee_pos, initial_ee_pos, current_ee_rot, motor_names = move_to_3d_location(
            robot=robot,
            kinematics=kinematics,
            target_pos=target_pos,
            initial_joints=initial_joints,
            home_joints=home_joints,
            duration=args.duration,
            fps=args.fps,
            end_behavior=args.end_behavior,
            position_weight=args.position_weight,
            return_duration=args.return_duration,
            position_tolerance=args.position_tolerance,
        )
        
        # Handle end behavior
        if args.end_behavior == "home":
            print("\n" + "="*60)
            print("Trajectory completed. Robot returned to home position.")
            print("="*60)
        else:  # end_behavior == "hold"
            print("\n" + "="*60)
            print("Robot has reached target position.")
            print("Holding position... Press Ctrl+C to return home.")
            print("="*60)
            
            try:
                while True:
                    # Continuously send final position to hold pose
                    robot.send_action(final_action)
                    precise_sleep(0.1)  # Send hold command every 100ms
            except KeyboardInterrupt:
                print("\n\nCtrl+C detected. Returning to home position...")
                
                # Return to home position
                if home_joints is not None:
                    # Recompute actual current EE position from final_joints (arm has moved from initial)
                    t_actual = kinematics.forward_kinematics(final_joints)
                    actual_ee_pos_raw = t_actual[:3, 3]
                    actual_ee_pos = FRAME_R_RAW_TO_DISPLAY @ actual_ee_pos_raw
                    
                    # Calculate end-effector position at home joint angles
                    t_home = kinematics.forward_kinematics(home_joints)
                    home_ee_pos_raw = t_home[:3, 3]
                    home_ee_pos = FRAME_R_RAW_TO_DISPLAY @ home_ee_pos_raw
                    
                    return_num_points = max(int(args.return_duration * args.fps), 2)
                    return_trajectory = create_smooth_trajectory(actual_ee_pos, home_ee_pos, return_num_points)
                    
                    try:
                        for i, return_step_pos in enumerate(return_trajectory):
                            # Transform step position back to IK frame: display -> raw URDF
                            return_step_pos_ik = FRAME_R_DISPLAY_TO_RAW @ return_step_pos
                            
                            step_pose = np.eye(4, dtype=float)
                            step_pose[:3, :3] = current_ee_rot
                            step_pose[:3, 3] = return_step_pos_ik
                            
                            try:
                                ik_solution = kinematics.inverse_kinematics(
                                    final_joints,
                                    step_pose,
                                    position_weight=args.position_weight,
                                    orientation_weight=0.01,
                                )
                                arm_joints = ik_solution[: len(motor_names) - 2]
                                return_action = {f"{motor}.pos": float(arm_joints[j]) for j, motor in enumerate(motor_names[:-2])}
                                return_action[motor_names[-2] + ".pos"] = float(final_joints[-2])
                                return_action[motor_names[-1] + ".pos"] = float(final_joints[-1])
                                robot.send_action(return_action)
                                final_joints = ik_solution
                            except Exception as e:
                                print(f"Warning: IK error during return: {e}")
                                if i > 0:
                                    print("Continuing with previous solution...")
                                else:
                                    raise
                            
                            precise_sleep(1.0 / args.fps)
                    except KeyboardInterrupt:
                        print("\nForce-stopped during return to home. Disconnecting now...")
                    
                    print("Returned to home position.")

    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
