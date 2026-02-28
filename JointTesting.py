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
Smoke-test sending actions to a single SO follower arm.


Example:
     python JointTesting.py  --port /dev/ttyACM0  --joint wrist_flex  --mode hold --target-offset 6 --duration 8


Sine example:
    python examples/tutorial/so_follower/smoke_send_action.py \
        --port /dev/tty.usbmodem5A460814411 \
        --joint wrist_flex \
        --mode sine \
        --amplitude 8 \
        --frequency 0.2 \
        --duration 8


Tip:
    Start with a small amplitude and keep one hand near an emergency stop.
"""


import argparse
import math
import time
from pathlib import Path


from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
from lerobot.utils.robot_utils import precise_sleep




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a single-joint target to an SO follower arm.")
    parser.add_argument("--port", required=True, help="Serial port of the SO follower arm.")
    parser.add_argument("--id", default=None, help="Optional robot id (used for calibration file lookup).")
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help=(
            "Optional calibration directory. If not set, defaults to "
            "HF_LEROBOT_CALIBRATION/robots/so_follower."
        ),
    )
    parser.add_argument("--joint", default="wrist_flex", help="Joint name to command (e.g. shoulder_pan).")
    parser.add_argument(
        "--mode",
        choices=["hold", "sine"],
        default="hold",
        help="Command mode: hold one fixed target or send sinusoidal targets.",
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        default=5.0,
        help="Used in hold mode: target = home + target_offset.",
    )
    parser.add_argument(
        "--target-absolute",
        type=float,
        default=None,
        help="Used in hold mode: if set, overrides target-offset with an absolute target value.",
    )
    parser.add_argument("--amplitude", type=float, default=5.0, help="Action amplitude in normalized units.")
    parser.add_argument("--frequency", type=float, default=0.2, help="Oscillation frequency in Hz.")
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency.")
    parser.add_argument("--duration", type=float, default=6.0, help="Test duration in seconds.")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help="Safety cap for per-step movement in send_action.",
    )
    parser.add_argument(
        "--calibrate-on-connect",
        action="store_true",
        help="If set, allow calibration routine when connecting.",
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Use degree normalization mode instead of [-100, 100] for body joints.",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()

    calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None

    cfg = SO100FollowerConfig(
        port=args.port,
        id=args.id,
        calibration_dir=calibration_dir,
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
    )
    robot = SO100Follower(cfg)

    if args.id is None:
        raise ValueError("Missing --id. Provide a calibration id (e.g. --id lamp).")

    expected_calib_dir = robot.calibration_dir if calibration_dir else HF_LEROBOT_CALIBRATION / "robots" / robot.name
    expected_calib_fpath = expected_calib_dir / f"{args.id}.json"
    print(f"Calibration id: {args.id}")
    print(f"Calibration file: {expected_calib_fpath}")
    print(f"Calibration file exists: {expected_calib_fpath.is_file()}")

    if not expected_calib_fpath.is_file() and not args.calibrate_on_connect:
        raise RuntimeError(
            "No calibration file found for this id and --calibrate-on-connect is not set. "
            "Either pass an existing --id, set --calibration-dir to the correct folder, "
            "or add --calibrate-on-connect to create calibration."
        )


    try:
        robot.connect(calibrate=args.calibrate_on_connect)
        obs = robot.get_observation()


        key = f"{args.joint}.pos"
        if key not in obs:
            available = ", ".join(sorted(k.removesuffix(".pos") for k in obs if k.endswith(".pos")))
            raise ValueError(f"Unknown joint '{args.joint}'. Available joints: {available}")


        home = float(obs[key])
        if args.mode == "hold":
            hold_target = args.target_absolute if args.target_absolute is not None else home + args.target_offset
            print(
                f"Connected. Holding joint '{args.joint}' at target={hold_target:.2f} "
                f"(home={home:.2f}) for {args.duration:.1f}s"
            )
        else:
            print(f"Connected. Testing joint '{args.joint}' from home={home:.2f} for {args.duration:.1f}s")


        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= args.duration:
                break


            if args.mode == "hold":
                target = hold_target
            else:
                target = home + args.amplitude * math.sin(2.0 * math.pi * args.frequency * elapsed)
           
            sent = robot.send_action({key: float(target)})
            print(f"t={elapsed:5.2f}s  target={target:8.3f}  sent={sent[key]:8.3f}")
            precise_sleep(1.0 / args.fps)


        print("Returning to home position...")
        robot.send_action({key: home})
        precise_sleep(0.5)
        print("Smoke test completed.")
    finally:
        if robot.is_connected:
            robot.disconnect()




if __name__ == "__main__":
    main()