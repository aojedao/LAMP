#!/usr/bin/env python
"""
track_bottle_v3.py -- Bottle Tracking with bias-corrected absolute IK controller
==================================================================================
Improvements over v1/v2:
  - Fixed Z = 0.15 m (no --approach-z confusion, Z is never recalculated from vision)
  - --x-offset / --y-offset  calibration bias correction applied to every world target
  - Controller sends ABSOLUTE clamped target to IK (not ee + delta), matching the
    axis_testing.py run_trajectory pattern — avoids accumulated drift / side-bias
  - Singularity guard: skip step if any joint would jump > 45 deg
  - Last-2 joints (wrist/gripper) locked at home values
  - --position-weight tunable for tighter/looser IK tracking

Usage
-----
python track_bottle_v3.py \\
    --port /dev/ttyACM0 --id my_lamp \\
    --urdf-path ~/Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf \\
    --camera /dev/video2

# Bias correction: if arm always goes e.g. 3 cm to the right, add --x-offset -0.03
python track_bottle_v3.py ... --x-offset -0.03 --y-offset 0.0

Optional flags
--------------
  --x-offset FLOAT    World-space X bias correction in metres  (default 0.0)
  --y-offset FLOAT    World-space Y bias correction in metres  (default 0.0)
  --conf FLOAT        YOLO confidence threshold  (default 0.35)
  --fps INT           Control loop frequency  (default 15)
  --max-step FLOAT    Max EE displacement per IK step (metres, default 0.03)
  --position-weight FLOAT  IK position accuracy weight  (default 10.0)
  --bbox-point {center,top,bottom}  Which point on bbox to track (default bottom)
  --max-target-jump FLOAT  Reject detections jumping > N metres (default None)
  --no-arm            Vision-only mode
  --no-window         Disable display
"""

import argparse
import glob
import json
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.utils.robot_utils import precise_sleep
    _ARM_AVAILABLE = True
except ImportError:
    _ARM_AVAILABLE = False
    def precise_sleep(t):
        time.sleep(t)

# ── Frame-transform matrices (from move_to_3d_location.py / axis_testing.py) ──
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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vision", "yolo11n.pt")
FIXED_Z_M   = 0.15          # metres above workspace — never changes
CAM_WIDTH,  CAM_HEIGHT  = 640, 480
ZOOM_DEFAULT, ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 1.0, 5.0, 0.25

_VISION_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision")
CAMERA_CALIB_FILE    = os.path.join(_VISION_DIR, "camera_calibration.json")
WORKSPACE_CALIB_FILE = os.path.join(_VISION_DIR, "workspace_calibration.json")


# ==============================================================================
#  CALIBRATION
# ==============================================================================

def load_camera_calibration():
    if not os.path.exists(CAMERA_CALIB_FILE):
        print(f"[CALIB] {CAMERA_CALIB_FILE} not found -- no undistortion.")
        return None, None, 600.0
    with open(CAMERA_CALIB_FILE) as f:
        data = json.load(f)
    K    = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"],   dtype=np.float64)
    print(f"[CALIB] Camera intrinsics loaded -- fx={K[0,0]:.1f} px")
    return K, dist, float(K[0, 0])


def load_workspace_calibration():
    if not os.path.exists(WORKSPACE_CALIB_FILE):
        print(f"[CALIB] {WORKSPACE_CALIB_FILE} not found -- workspace DISABLED.")
        return None, None, None
    with open(WORKSPACE_CALIB_FILE) as f:
        data = json.load(f)
    poly = np.array(data["pixel_corners"], dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
    H    = np.array(data["homography"],    dtype=np.float64)
    ws   = data["workspace_m"]
    print(f"[CALIB] Workspace loaded -- {ws[0]*100:.0f} cm x {ws[1]*100:.0f} cm")
    return poly, H, ws


# ==============================================================================
#  CAMERA
# ==============================================================================

def open_camera(path):
    digits = ''.join(filter(str.isdigit, str(path).split('/')[-1]))
    idx    = int(digits) if digits else 0
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    for _ in range(20):
        cap.grab()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Opened {path}  {w}x{h}")
    return cap


def find_robot_camera():
    for path in sorted(glob.glob("/dev/video*")):
        if path == "/dev/video0":
            continue
        digits = ''.join(filter(str.isdigit, path.split('/')[-1]))
        if not digits:
            continue
        cap = cv2.VideoCapture(int(digits), cv2.CAP_V4L2)
        if cap.isOpened():
            cap.release()
            print(f"[CAM] Robot camera auto-detected: {path}")
            return path
    return "/dev/video0"


# ==============================================================================
#  VISION HELPERS
# ==============================================================================

def crop_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame
    h, w  = frame.shape[:2]
    cw, ch = int(w / zoom), int(h / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def pixel_to_world_xy(px, py, H):
    pt = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


# ==============================================================================
#  DETECTION THREAD
# ==============================================================================

class DetectionThread(threading.Thread):
    def __init__(self, cap, model, cam_matrix, dist_coeffs,
                 ws_polygon, homography,
                 conf_thresh=0.35,
                 bbox_point="bottom",
                 max_target_jump=None,
                 x_offset=0.0,
                 y_offset=0.0):
        super().__init__(daemon=True)
        self.cap             = cap
        self.model           = model
        self.cam_matrix      = cam_matrix
        self.dist_coeffs     = dist_coeffs
        self.ws_polygon      = ws_polygon
        self.homography      = homography
        self.conf_thresh     = conf_thresh
        self.bbox_point      = bbox_point
        self.max_target_jump = max_target_jump
        self.x_offset        = x_offset   # bias correction in metres
        self.y_offset        = y_offset

        self._lock   = threading.Lock()
        self._target = None    # (x_m, y_m) in world coords, bias-corrected
        self._frame  = None
        self.running = True

    @property
    def target(self):
        with self._lock:
            return self._target

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False

    def run(self):
        bottle_ids  = [cid for cid, n in self.model.names.items() if n == "bottle"]
        classes_arg = bottle_ids if bottle_ids else None

        while self.running:
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.cam_matrix is not None:
                frame = cv2.undistort(frame, self.cam_matrix, self.dist_coeffs)

            results = self.model.predict(
                source=frame,
                imgsz=max(CAM_WIDTH, CAM_HEIGHT),
                conf=self.conf_thresh,
                classes=classes_arg,
                verbose=False,
            )

            best_box, best_conf = None, 0.0
            for result in results:
                for box in result.boxes:
                    if self.model.names[int(box.cls)] == "bottle":
                        c = float(box.conf)
                        if c > best_conf:
                            best_conf = c
                            best_box  = box.xyxy[0].cpu().numpy()

            display    = frame.copy()
            new_target = None

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cx_px = (x1 + x2) / 2.0

                if self.bbox_point == "top":
                    cy_px = y1
                elif self.bbox_point == "bottom":
                    cy_px = y2
                else:
                    cy_px = (y1 + y2) / 2.0

                inside = True
                if self.ws_polygon is not None:
                    inside = cv2.pointPolygonTest(
                        self.ws_polygon, (float(cx_px), float(cy_px)), False
                    ) >= 0

                if self.homography is not None:
                    raw_x, raw_y = pixel_to_world_xy(cx_px, cy_px, self.homography)
                    # Apply bias correction offsets
                    x_m = raw_x + self.x_offset
                    y_m = raw_y + self.y_offset
                    new_target = (x_m, y_m)
                elif not np.isnan(cx_px):
                    # No homography: normalised pixel as placeholder
                    x_m = cx_px / CAM_WIDTH + self.x_offset
                    y_m = cy_px / CAM_HEIGHT + self.y_offset
                    new_target = (x_m, y_m)

                # Jump guard
                if new_target is not None and self.max_target_jump is not None:
                    with self._lock:
                        prev = self._target
                    if prev is not None:
                        jump = np.hypot(new_target[0] - prev[0],
                                        new_target[1] - prev[1])
                        if jump > self.max_target_jump:
                            print(f"[DET] Jump guard: {jump:.3f}m > "
                                  f"{self.max_target_jump:.3f}m -- ignored.")
                            new_target = prev

                # ── Annotate ──────────────────────────────────────────────────
                color = (0, 220, 0) if inside else (0, 0, 220)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                if new_target is not None:
                    lbl = (f"bottle {best_conf:.2f}  "
                           f"X={new_target[0]:.3f}m Y={new_target[1]:.3f}m "
                           f"[{self.bbox_point}]")
                else:
                    lbl = f"bottle {best_conf:.2f}"
                cv2.putText(display, lbl, (int(x1), max(int(y1)-8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(display, lbl, (int(x1), max(int(y1)-8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.drawMarker(display, (int(cx_px), int(cy_px)),
                               (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

                if not inside and self.ws_polygon is not None:
                    cv2.putText(display, "OUTSIDE",
                                (int(x1), int(y2)+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)

            if self.ws_polygon is not None:
                cv2.polylines(display, [self.ws_polygon], True, (255, 200, 0), 2)

            status = "TRACKING" if new_target is not None else "SEARCHING..."
            cv2.putText(display, status, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 0) if new_target is not None else (0, 180, 255), 1)

            with self._lock:
                self._target = new_target
                self._frame  = display


# ==============================================================================
#  ARM HELPERS  — absolute-target IK (axis_testing.py pattern)
# ==============================================================================

def get_ee_pos(kinematics, joints):
    """Current EE position in display frame."""
    T = kinematics.forward_kinematics(joints)
    return FRAME_R_RAW_TO_DISPLAY @ T[:3, 3]


def ik_to_absolute(kinematics, joints, abs_target_display, ee_rot,
                   position_weight=10.0):
    """
    Solve IK for an ABSOLUTE target position in display frame.
    Returns new joint array, or None on failure.
    """
    target_raw = FRAME_R_DISPLAY_TO_RAW @ abs_target_display
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = ee_rot
    pose[:3, 3]  = target_raw
    try:
        return kinematics.inverse_kinematics(
            joints, pose,
            position_weight=position_weight,
            orientation_weight=0.01,
        )
    except Exception as e:
        print(f"[IK] Warning: {e}")
        return None


def build_action(joints, motor_names, home_last2):
    """First N-2 joints from IK solution; last 2 locked to home."""
    n = len(motor_names) - 2
    action = {f"{motor_names[i]}.pos": float(joints[i]) for i in range(n)}
    action[motor_names[-2] + ".pos"] = float(home_last2[0])
    action[motor_names[-1] + ".pos"] = float(home_last2[1])
    return action


def find_urdf():
    candidates = [
        Path.home() / "Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf",
        Path.home() / "SO-ARM100/Simulation/SO100/so100.urdf",
        Path.home() / "SO-ARM100-main/Simulation/SO100/so100.urdf",
        Path("/opt/placo/models/so100/so100.urdf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "so100"


# ==============================================================================
#  ARGS
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Bottle tracking v3: absolute IK controller + bias correction."
    )
    p.add_argument("--port",            default=None)
    p.add_argument("--id",              default=None)
    p.add_argument("--calibration-dir", default=None)
    p.add_argument("--urdf-path",       default=None)
    p.add_argument("--camera",          default=None)
    p.add_argument("--webcam",          action="store_true")
    p.add_argument("--conf",            type=float, default=0.35)
    p.add_argument("--fps",             type=int,   default=15)
    p.add_argument("--max-step",        type=float, default=0.03,
                   help="Max EE displacement per IK step (metres, default 0.03)")
    p.add_argument("--position-weight", type=float, default=10.0,
                   help="IK position accuracy weight (default 10.0, higher = tighter)")
    p.add_argument("--x-offset",        type=float, default=0.0,
                   help="World X bias correction in metres. "
                        "If arm always overshoots right, use a negative value.")
    p.add_argument("--y-offset",        type=float, default=0.0,
                   help="World Y bias correction in metres. "
                        "If arm always overshoots forward, use a negative value.")
    p.add_argument("--bbox-point",      default="bottom",
                   choices=["center", "top", "bottom"],
                   help="Bbox point to track: bottom=table contact (default), "
                        "center=body, top=head/hands")
    p.add_argument("--max-target-jump", type=float, default=None,
                   help="Reject detections jumping > N metres from last position")
    p.add_argument("--no-arm",          action="store_true")
    p.add_argument("--no-window",       action="store_true")
    return p.parse_args()


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    args = parse_args()

    use_arm = not args.no_arm and args.port and args.id
    if not use_arm:
        print("[INFO] Vision-only mode (no arm).")
    if use_arm and not _ARM_AVAILABLE:
        print("[ERROR] lerobot not importable -- use --no-arm.")
        sys.exit(1)

    print(f"[INFO] Fixed Z = {FIXED_Z_M:.3f} m")
    print(f"[INFO] Bias correction: X offset={args.x_offset:+.3f} m  "
          f"Y offset={args.y_offset:+.3f} m")

    # -- Model -----------------------------------------------------------------
    print(f"[INFO] Loading YOLO from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    bottle_ids = [cid for cid, n in model.names.items() if n == "bottle"]
    print(f"[INFO] Model ready -- bottle YOLO IDs: {bottle_ids}")

    # -- Calibration -----------------------------------------------------------
    cam_matrix, dist_coeffs, _ = load_camera_calibration()
    ws_polygon, homography, ws_m = load_workspace_calibration()

    # -- Camera ----------------------------------------------------------------
    cam_path = ("/dev/video0" if args.webcam
                else args.camera if args.camera
                else find_robot_camera())
    cap = open_camera(cam_path)

    # -- Detection thread ------------------------------------------------------
    det = DetectionThread(
        cap, model, cam_matrix, dist_coeffs,
        ws_polygon, homography,
        conf_thresh=args.conf,
        bbox_point=args.bbox_point,
        max_target_jump=args.max_target_jump,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
    )

    # -- Arm setup -------------------------------------------------------------
    robot          = None
    kinematics     = None
    joints         = None
    motor_names    = None
    initial_ee_rot = None
    home_last2     = None
    home_joints    = None

    if use_arm:
        urdf_path = (str(Path(args.urdf_path).expanduser())
                     if args.urdf_path else find_urdf())
        cal_dir   = (Path(args.calibration_dir).expanduser()
                     if args.calibration_dir else None)

        cfg = SO100FollowerConfig(
            port=args.port,
            id=args.id,
            calibration_dir=cal_dir,
            use_degrees=True,
            max_relative_target=10.0,
        )
        robot = SO100Follower(cfg)
        print(f"[ARM] Connecting on {args.port} ...")
        robot.connect(calibrate=False)
        print("[ARM] Connected.")

        obs         = robot.get_observation()
        motor_names = list(robot.bus.motors.keys())
        joints      = np.array(
            [float(obs[f"{m}.pos"]) for m in motor_names if f"{m}.pos" in obs],
            dtype=float)
        print(f"[ARM] Initial joints: {np.round(joints, 2)}")

        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="jaw",
            joint_names=motor_names,
        )
        T0             = kinematics.forward_kinematics(joints)
        initial_ee_rot = T0[:3, :3].copy()
        home_last2     = joints[-2:].copy()
        home_joints    = joints.copy()

        home_ee = FRAME_R_RAW_TO_DISPLAY @ T0[:3, 3]
        print(f"[ARM] Home EE (display frame): {np.round(home_ee, 4)}")
        print(f"[ARM] Wrist/gripper locked at: {np.round(home_last2, 2)}")
        print(f"[ARM] Kinematics ready -- position_weight={args.position_weight}")

    # -- Main loop -------------------------------------------------------------
    det.start()
    dt                    = 1.0 / args.fps
    zoom                  = ZOOM_DEFAULT
    joint_refresh_counter = 0

    print(f"\n[INFO] Running -- 'q' quit  '+'/'-' zoom  Ctrl+C stop\n")

    try:
        while det.running:
            tick = time.perf_counter()

            tgt = det.target   # (x_m, y_m) bias-corrected, or None

            # ── Arm control ────────────────────────────────────────────────────
            if use_arm:
                # Periodically read back actual joints from robot (drift prevention)
                joint_refresh_counter += 1
                if joint_refresh_counter >= 30:
                    joint_refresh_counter = 0
                    try:
                        obs = robot.get_observation()
                        joints = np.array(
                            [float(obs[f"{m}.pos"]) for m in motor_names
                             if f"{m}.pos" in obs], dtype=float)
                    except Exception:
                        pass

                if tgt is not None and not np.any(np.isnan(tgt)):
                    x_m, y_m = tgt

                    # Absolute target: fixed Z, bias-corrected X/Y
                    abs_target = np.array([x_m, y_m, FIXED_Z_M])

                    # Clamp: if target is too far from current EE,
                    # move only max_step along the straight-line path
                    # (axis_testing.py create_trajectory + run_trajectory pattern)
                    ee_pos = get_ee_pos(kinematics, joints)
                    delta  = abs_target - ee_pos
                    dist   = np.linalg.norm(delta)

                    if dist > args.max_step:
                        # Step max_step toward the absolute target
                        step_target = ee_pos + (delta / dist) * args.max_step
                    else:
                        step_target = abs_target

                    sol = ik_to_absolute(kinematics, joints, step_target,
                                         initial_ee_rot, args.position_weight)

                    if sol is not None:
                        # Singularity guard: skip if any joint jumps > 45 deg
                        max_delta = float(np.max(np.abs(sol - joints)))
                        if max_delta > 45.0:
                            print(f"[IK] Singularity guard: {max_delta:.1f}deg -- skipped.")
                            robot.send_action(build_action(joints, motor_names, home_last2))
                        else:
                            robot.send_action(build_action(sol, motor_names, home_last2))
                            joints = sol
                            print(f"[TRACK] target=({x_m:.3f},{y_m:.3f},{FIXED_Z_M:.3f})  "
                                  f"step_tgt=({step_target[0]:.3f},{step_target[1]:.3f},{step_target[2]:.3f})  "
                                  f"EE=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})  "
                                  f"dist={dist:.3f}m")
                    else:
                        robot.send_action(build_action(joints, motor_names, home_last2))
                else:
                    # No detection -- hold current position
                    robot.send_action(build_action(joints, motor_names, home_last2))

            # ── Display ────────────────────────────────────────────────────────
            if not args.no_window:
                frame = det.latest_frame
                if frame is not None:
                    if zoom > 1.0:
                        frame = crop_zoom(frame, zoom)
                    cv2.putText(frame,
                                f"V3  Z={FIXED_Z_M:.2f}m  "
                                f"dx={args.x_offset:+.3f} dy={args.y_offset:+.3f}  "
                                f"zoom={zoom:.1f}x",
                                (6, CAM_HEIGHT - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
                    cv2.imshow("LAMP Bottle V3", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("+"), ord("=")):
                    zoom = min(zoom + ZOOM_STEP, ZOOM_MAX)
                elif key == ord("-"):
                    zoom = max(zoom - ZOOM_STEP, ZOOM_MIN)

            precise_sleep(max(0.0, dt - (time.perf_counter() - tick)))

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C -- stopping.")
    finally:
        det.stop()
        cap.release()
        cv2.destroyAllWindows()

        if robot is not None and robot.is_connected:
            # Return to home in joint space
            if home_joints is not None and joints is not None:
                print("[ARM] Returning to home ...")
                n_steps = 60
                for i in range(1, n_steps + 1):
                    t      = i / n_steps
                    interp = joints + t * (home_joints - joints)
                    try:
                        robot.send_action(
                            {f"{m}.pos": float(interp[j])
                             for j, m in enumerate(motor_names)})
                    except Exception:
                        break
                    time.sleep(0.04)
                print("[ARM] Home reached.")
            print("[ARM] Disconnecting ...")
            robot.disconnect()
            print("[ARM] Disconnected.")

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
