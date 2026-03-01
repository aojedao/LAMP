#!/usr/bin/env python
"""
track_bottle.py -- Continuous Bottle Tracking with SO-100 Arm
=============================================================
Detects a bottle in the camera frame using YOLO and continuously
steers the arm end-effector toward it using IK from move_to_3d_location.py.

Camera pipeline : main_v2.py   (V4L2 / MJPG, warm-up, undistort)
Arm controller  : move_to_3d_location.py  (IK, frame transforms, trajectory)

Coordinate system
-----------------
  X  -- workspace horizontal (metres, from workspace_calibration.json)
  Y  -- workspace vertical   (metres)
  Z  -- fixed approach height above workspace surface (--approach-z, default 0.10 m)

Usage
-----
python track_bottle.py \\
    --port /dev/ttyACM0 \\
    --id   my_lamp \\
    --urdf-path /path/to/so100.urdf \\
    --camera /dev/video2

Optional flags
--------------
  --webcam            Force /dev/video0 (no arm required -- vision only)
  --conf FLOAT        YOLO confidence threshold  (default 0.45)
  --fps INT           Control loop / display frequency  (default 15)
  --max-step FLOAT    Max EE displacement per tick in metres  (default 0.03)
  --approach-z FLOAT  Fixed Z above workspace surface in metres  (default 0.10)
  --deadzone FLOAT    Min target XY change to trigger a move  (default 0.01)
  --no-arm            Run vision only, skip arm connection
  --no-window         Disable OpenCV display

Install
-------
pip install numpy opencv-python ultralytics
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

# ── Arm / kinematics (only imported when needed) ──────────────────────────────
try:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.utils.robot_utils import precise_sleep
    _ARM_AVAILABLE = True
except ImportError:
    _ARM_AVAILABLE = False
    def precise_sleep(t):
        time.sleep(t)

# ── Frame-transform matrices from move_to_3d_location.py ──────────────────────
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

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "vision", "yolo11n.pt")
CONFIDENCE_THRESH = 0.35
BOTTLE_WIDTH_M    = 0.07   # standard 500 ml PET bottle diameter

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
#  CAMERA  (main_v2 pattern: V4L2 + MJPG + warm-up)
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
    cw    = int(w / zoom)
    ch    = int(h / zoom)
    x0    = (w - cw) // 2
    y0    = (h - ch) // 2
    return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def pixel_to_world_xy(px, py, H):
    pt = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def estimate_depth(bbox_width_px, fx):
    if bbox_width_px < 1:
        return float("inf")
    return (BOTTLE_WIDTH_M * fx) / bbox_width_px


# ==============================================================================
#  DETECTION THREAD  (keeps vision async from arm control)
# ==============================================================================

class DetectionThread(threading.Thread):
    def __init__(self, cap, model, cam_matrix, dist_coeffs,
                 ws_polygon, homography, focal_px, conf_thresh,
                 bbox_point="center", max_target_jump=None):
        super().__init__(daemon=True)
        self.cap             = cap
        self.model           = model
        self.cam_matrix      = cam_matrix
        self.dist_coeffs     = dist_coeffs
        self.ws_polygon      = ws_polygon
        self.homography      = homography
        self.focal_px        = focal_px
        self.conf_thresh     = conf_thresh
        self.bbox_point      = bbox_point      # "center", "top", or "bottom"
        self.max_target_jump = max_target_jump # metres; None = no limit

        self._lock        = threading.Lock()
        self._target      = None    # (x_m, y_m, depth_m) or None — current frame
        self._last_target = None    # last non-None target — for searching
        self._frame       = None    # latest annotated frame
        self.running      = True

    @property
    def target(self):
        with self._lock:
            return self._target

    @property
    def last_target(self):
        """Last non-None target — persists after bottle disappears."""
        with self._lock:
            return self._last_target

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False

    def run(self):
        # bottle class id
        bottle_ids = [cid for cid, n in self.model.names.items() if n == "bottle"]
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

                # Select Y point on bbox based on --bbox-point
                if self.bbox_point == "top":
                    cy_px = y1
                elif self.bbox_point == "bottom":
                    cy_px = y2
                else:  # center (default)
                    cy_px = (y1 + y2) / 2.0

                inside = True
                if self.ws_polygon is not None:
                    inside = cv2.pointPolygonTest(
                        self.ws_polygon, (float(cx_px), float(cy_px)), False
                    ) >= 0

                depth_m = estimate_depth(x2 - x1, self.focal_px)

                # Always compute world coords regardless of workspace boundary
                # (inside is display-only — arm tracks the bottle everywhere)
                if self.homography is not None:
                    x_m, y_m = pixel_to_world_xy(cx_px, cy_px, self.homography)
                    new_target = (x_m, y_m, depth_m)
                else:
                    # No workspace calib -- use normalised pixel coords as proxy
                    new_target = (cx_px / CAM_WIDTH, cy_px / CAM_HEIGHT, depth_m)

                # Jump guard: reject target if it's too far from the last accepted one
                if new_target is not None and self.max_target_jump is not None:
                    with self._lock:
                        prev = self._target
                    if prev is not None:
                        jump = np.hypot(new_target[0] - prev[0],
                                        new_target[1] - prev[1])
                        if jump > self.max_target_jump:
                            print(f"[DET] Jump guard: {jump:.3f}m > "
                                  f"{self.max_target_jump:.3f}m -- ignored.")
                            new_target = prev  # hold last known good position

                color = (0, 220, 0) if inside else (0, 0, 220)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                lbl = f"bottle {best_conf:.2f} [{self.bbox_point}]"
                if new_target is not None:
                    lbl += f"  X={new_target[0]:.2f} Y={new_target[1]:.2f} d={new_target[2]:.2f}m"
                # dark outline + coloured text
                cv2.putText(display, lbl, (int(x1), max(int(y1)-8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(display, lbl, (int(x1), max(int(y1)-8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # Draw the TARGET point (not just bbox centre)
                cv2.drawMarker(display, (int(cx_px), int(cy_px)),
                               color, cv2.MARKER_CROSS, 16, 2)

                if not inside:
                    cv2.putText(display, "OUTSIDE WORKSPACE",
                                (int(x1), int(y2)+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

            if self.ws_polygon is not None:
                cv2.polylines(display, [self.ws_polygon], True, (255, 200, 0), 2)

            status = "TRACKING" if new_target is not None else "SEARCHING..."
            cv2.putText(display, status, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 0) if new_target is not None else (0, 200, 255), 1)

            with self._lock:
                self._target = new_target
                if new_target is not None:
                    self._last_target = new_target
                self._frame  = display


# ==============================================================================
#  ARM HELPERS  (IK step from move_to_3d_location.py)
# ==============================================================================

def get_ee_pos(kinematics, joints):
    """EE position in display frame."""
    T = kinematics.forward_kinematics(joints)
    return FRAME_R_RAW_TO_DISPLAY @ T[:3, 3]


def ik_step(kinematics, joints, target_display, ee_rot,
            motor_names, position_weight=10.0):
    """One IK solution toward target_display. Returns new joints or None."""
    target_raw = FRAME_R_DISPLAY_TO_RAW @ target_display
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = ee_rot
    pose[:3, 3]  = target_raw
    try:
        sol = kinematics.inverse_kinematics(
            joints, pose,
            position_weight=position_weight,
            orientation_weight=0.01,
        )
        return sol
    except Exception as e:
        print(f"[IK] Warning: {e}")
        return None


def build_action(joints, motor_names, home_last2):
    """Build action dict: first N-2 joints from IK, last 2 locked to home."""
    n = len(motor_names) - 2
    action = {f"{motor_names[i]}.pos": float(joints[i]) for i in range(n)}
    action[motor_names[-2] + ".pos"] = float(home_last2[0])
    action[motor_names[-1] + ".pos"] = float(home_last2[1])
    return action


def find_urdf():
    candidates = [
        Path.home() / "SO-ARM100-main/Simulation/SO100/so100.urdf",
        Path.home() / "SO-ARM100-main/Simulation/SO100/so100_calib.urdf",
        Path.home() / "SO-ARM100-main/Simulation/SO101/so101.urdf",
        Path("/opt/placo/models/so100/so100.urdf"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "so100"


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Track a bottle with the SO-100 arm using YOLO + IK."
    )
    p.add_argument("--port",        default=None, help="Serial port, e.g. /dev/ttyACM0")
    p.add_argument("--id",          default=None, help="Robot calibration ID")
    p.add_argument("--calibration-dir", default=None)
    p.add_argument("--urdf-path",   default=None)
    p.add_argument("--camera",      default=None,
                   help="Camera device path, e.g. /dev/video2")
    p.add_argument("--webcam",      action="store_true",
                   help="Force /dev/video0 (vision only, no arm needed)")
    p.add_argument("--conf",        type=float, default=0.45)
    p.add_argument("--fps",         type=int,   default=15)
    p.add_argument("--max-step",    type=float, default=0.03,
                   help="Max EE displacement per tick (metres)")
    p.add_argument("--approach-z",  type=float, default=0.10,
                   help="Fixed Z above workspace surface (metres)")
    p.add_argument("--bbox-point",  default="center",
                   choices=["center", "top", "bottom"],
                   help="Which point on the bounding box to track: "
                        "center (body), top (head/hands), bottom (feet)  "
                        "(default: center)")
    p.add_argument("--max-target-jump", type=float, default=None,
                   help="Reject detections whose XY position jumps more than "
                        "this many metres from the last accepted position. "
                        "Prevents spurious detections from jerking the arm. "
                        "E.g. --max-target-jump 0.15")
    p.add_argument("--deadzone",    type=float, default=0.01,
                   help="Min XY target change to trigger a move (metres)")
    p.add_argument("--no-arm",      action="store_true",
                   help="Run vision only, skip arm connection")
    p.add_argument("--no-window",   action="store_true",
                   help="Disable OpenCV display window")
    return p.parse_args()


def main():
    args = parse_args()

    use_arm = not args.no_arm and args.port and args.id
    if not use_arm:
        print("[INFO] Running in vision-only mode (no arm).")
    if use_arm and not _ARM_AVAILABLE:
        print("[ERROR] lerobot not importable -- install it or use --no-arm.")
        sys.exit(1)

    # -- Model -----------------------------------------------------------------
    print(f"[INFO] Loading YOLO from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    bottle_ids = [cid for cid, n in model.names.items() if n == "bottle"]
    if bottle_ids:
        print(f"[INFO] Model ready -- tracking: bottle  (YOLO IDs: {bottle_ids})")
    else:
        print("[WARN] 'bottle' class not found in model -- will attempt anyway.")

    # -- Calibration -----------------------------------------------------------
    cam_matrix, dist_coeffs, focal_px = load_camera_calibration()
    ws_polygon, homography, ws_m      = load_workspace_calibration()

    # -- Camera ----------------------------------------------------------------
    if args.webcam:
        cam_path = "/dev/video0"
    elif args.camera:
        cam_path = args.camera
    else:
        cam_path = find_robot_camera()

    cap = open_camera(cam_path)

    # -- Detection thread ------------------------------------------------------
    det = DetectionThread(
        cap, model, cam_matrix, dist_coeffs,
        ws_polygon, homography, focal_px,
        conf_thresh=args.conf,
        bbox_point=args.bbox_point,
        max_target_jump=args.max_target_jump,
    )

    # -- Arm setup -------------------------------------------------------------
    robot      = None
    kinematics = None
    joints     = None
    motor_names   = None
    initial_ee_rot = None
    home_last2     = None

    if use_arm:
        urdf_path = args.urdf_path or find_urdf()
        cal_dir   = Path(args.calibration_dir).expanduser() if args.calibration_dir else None

        cfg   = SO100FollowerConfig(
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
            dtype=float,
        )
        print(f"[ARM] Initial joints: {joints}")

        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="jaw",
            joint_names=motor_names,
        )
        T0             = kinematics.forward_kinematics(joints)
        initial_ee_rot = T0[:3, :3].copy()
        home_last2     = joints[-2:].copy()  # wrist/gripper locked here
        print("[ARM] Kinematics ready.")
        print(f"[ARM] Wrist/gripper locked at: {home_last2}")

    # -- Start -----------------------------------------------------------------
    det.start()
    dt                    = 1.0 / args.fps
    zoom                  = ZOOM_DEFAULT
    joint_refresh_counter = 0

    print(f"\n[INFO] Tracking -- 'q' quit  '+'/'-' zoom  Ctrl+C stop\n")

    try:
        while det.running:
            tick = time.perf_counter()

            tgt = det.target   # (x_m, y_m, depth_m) or None

            # -- Arm control (main thread, IK step) ----------------------------
            if use_arm:
                # Periodically refresh joints from robot to prevent drift
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

                if tgt is not None:
                    x_m, y_m, _ = tgt
                    if not (np.isnan(x_m) or np.isnan(y_m)):
                        target_display = np.array([x_m, y_m, args.approach_z])
                        ee_pos    = get_ee_pos(kinematics, joints)
                        direction = target_display - ee_pos
                        dist      = np.linalg.norm(direction)
                        if dist > args.max_step:
                            direction = direction / dist * args.max_step
                        sol = ik_step(kinematics, joints,
                                      ee_pos + direction,
                                      initial_ee_rot, motor_names)
                        if sol is not None:
                            robot.send_action(build_action(sol, motor_names, home_last2))
                            joints = sol
                            print(
                                f"[TRACK] bottle X={x_m:.3f} Y={y_m:.3f}  "
                                f"EE=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})  "
                                f"dist={dist:.3f}m"
                            )
                        else:
                            # IK failed -- hold position
                            robot.send_action(build_action(joints, motor_names, home_last2))
                    else:
                        robot.send_action(build_action(joints, motor_names, home_last2))
                else:
                    # No bottle detected -- keep moving toward last known target
                    last_tgt = det.last_target
                    if last_tgt is not None:
                        x_m, y_m, _ = last_tgt
                        if not (np.isnan(x_m) or np.isnan(y_m)):
                            target_display = np.array([x_m, y_m, args.approach_z])
                            ee_pos    = get_ee_pos(kinematics, joints)
                            direction = target_display - ee_pos
                            dist      = np.linalg.norm(direction)
                            if dist > args.max_step:
                                direction = direction / dist * args.max_step
                            sol = ik_step(kinematics, joints,
                                          ee_pos + direction,
                                          initial_ee_rot, motor_names)
                            if sol is not None:
                                robot.send_action(build_action(sol, motor_names, home_last2))
                                joints = sol
                                print(f"[SEARCH] Moving toward last known position X={x_m:.3f} Y={y_m:.3f}")
                            else:
                                robot.send_action(build_action(joints, motor_names, home_last2))
                        else:
                            robot.send_action(build_action(joints, motor_names, home_last2))
                    else:
                        robot.send_action(build_action(joints, motor_names, home_last2))

            # -- Display (main thread only on Linux) ---------------------------
            if not args.no_window:
                frame = det.latest_frame
                if frame is not None:
                    # Zoom overlay
                    if zoom > 1.0:
                        frame = crop_zoom(frame, zoom)
                    label = f"BOTTLE TRACKING  zoom {zoom:.1f}x"
                    cv2.putText(frame, label,
                                (CAM_WIDTH - 210, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
                    cv2.imshow("LAMP Bottle Tracking", frame)

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
            print("[ARM] Disconnecting ...")
            robot.disconnect()
            print("[ARM] Disconnected.")

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
