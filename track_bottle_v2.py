#!/usr/bin/env python
"""
track_bottle_v2.py -- Aggressive Bottle Tracking (always move toward bottle)
=============================================================================
Like track_bottle.py but the arm ALWAYS tries to move toward the bounding-box
centre of the detected bottle.  Workspace polygon and inside/outside checks are
displayed for reference but NEVER block the arm.

If no homography is available the pixel centre is linearly mapped to a
configurable arm workspace rectangle so the arm still chases the bottle.

Usage
-----
python track_bottle_v2.py \\
    --port /dev/ttyACM0 \\
    --id   my_lamp \\
    --urdf-path /path/to/so100.urdf \\
    --camera /dev/video2

Optional flags
--------------
  --webcam                Force /dev/video0 (vision only, no arm)
  --conf   FLOAT          YOLO confidence threshold  (default 0.35)
  --fps    INT            Control loop frequency      (default 15)
  --max-step FLOAT        Max EE displacement per tick in metres (default 0.03)
  --approach-z FLOAT      Fixed Z above desk surface  (default 0.10 m)
  --pixel-x-range A B     Arm X range when no homography  (default 0.0 0.4 m)
  --pixel-y-range A B     Arm Y range when no homography  (default 0.0 0.3 m)
  --no-arm                Run vision only, skip arm connection
  --no-window             Disable OpenCV display window
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

# ── Arm / kinematics ----------------------------------------------------------
try:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.utils.robot_utils import precise_sleep
    _ARM_AVAILABLE = True
except ImportError:
    _ARM_AVAILABLE = False
    def precise_sleep(t):
        time.sleep(t)

# ── Frame-transform matrices (from move_to_3d_location.py) -------------------
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

# ── Configuration -------------------------------------------------------------
MODEL_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "vision", "yolo11n.pt")
BOTTLE_WIDTH_M = 0.07
CAM_WIDTH, CAM_HEIGHT = 640, 480
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
    poly = (np.array(data["pixel_corners"], dtype=np.float32)
            .reshape(-1, 1, 2).astype(np.int32))
    H    = np.array(data["homography"], dtype=np.float64)
    ws   = data["workspace_m"]
    print(f"[CALIB] Workspace loaded -- {ws[0]*100:.0f} cm x {ws[1]*100:.0f} cm")
    return poly, H, ws


# ==============================================================================
#  CAMERA  (V4L2 + MJPG + warm-up, identical to main_v2.py)
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
    h, w = frame.shape[:2]
    cw, ch = int(w / zoom), int(h / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def pixel_to_world_xy(px, py, H):
    pt = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def pixel_to_arm_xy(px, py, x_range, y_range):
    """Linear map from pixel (0..W, 0..H) → arm metres using given ranges."""
    x_m = x_range[0] + (px / CAM_WIDTH)  * (x_range[1] - x_range[0])
    y_m = y_range[0] + (py / CAM_HEIGHT) * (y_range[1] - y_range[0])
    return float(x_m), float(y_m)


def estimate_depth(bbox_width_px, fx):
    if bbox_width_px < 1:
        return float("inf")
    return (BOTTLE_WIDTH_M * fx) / bbox_width_px


# ==============================================================================
#  DETECTION THREAD
#  KEY DIFFERENCE vs track_bottle.py:
#    - target is set from bbox CENTER regardless of workspace polygon
#    - workspace polygon is drawn for info only, never gates the target
#    - fallback pixel→arm mapping used when no homography
# ==============================================================================

class DetectionThread(threading.Thread):
    def __init__(self, cap, model, cam_matrix, dist_coeffs,
                 ws_polygon, homography, focal_px, conf_thresh,
                 pixel_x_range, pixel_y_range):
        super().__init__(daemon=True)
        self.cap           = cap
        self.model         = model
        self.cam_matrix    = cam_matrix
        self.dist_coeffs   = dist_coeffs
        self.ws_polygon    = ws_polygon    # reference only, not a gate
        self.homography    = homography
        self.focal_px      = focal_px
        self.conf_thresh   = conf_thresh
        self.pixel_x_range = pixel_x_range
        self.pixel_y_range = pixel_y_range

        self._lock   = threading.Lock()
        self._target = None   # (x_m, y_m, depth_m) — always set when detected
        self._frame  = None   # latest annotated BGR
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

                # Always use bounding-box centre
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0

                depth_m = estimate_depth(x2 - x1, self.focal_px)

                # World XY: homography if available, else linear pixel map
                if self.homography is not None:
                    x_m, y_m = pixel_to_world_xy(cx_px, cy_px, self.homography)
                else:
                    x_m, y_m = pixel_to_arm_xy(cx_px, cy_px,
                                                self.pixel_x_range,
                                                self.pixel_y_range)

                # ALWAYS set target — no workspace gate
                new_target = (x_m, y_m, depth_m)

                # Is point inside workspace polygon? (display info only)
                inside = True
                if self.ws_polygon is not None:
                    inside = cv2.pointPolygonTest(
                        self.ws_polygon, (float(cx_px), float(cy_px)), False
                    ) >= 0

                # Draw bounding box (green always, even outside workspace)
                cv2.rectangle(display,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 220, 0), 2)

                lbl = (f"bottle {best_conf:.2f}  "
                       f"X={x_m:.2f} Y={y_m:.2f} d={depth_m:.2f}m")
                cv2.putText(display, lbl,
                            (int(x1), max(int(y1) - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(display, lbl,
                            (int(x1), max(int(y1) - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

                # Cross-hair at bbox centre (tracked point)
                cv2.drawMarker(display, (int(cx_px), int(cy_px)),
                               (0, 220, 0), cv2.MARKER_CROSS, 20, 2)

                # Workspace info label (does NOT block movement)
                if not inside and self.ws_polygon is not None:
                    cv2.putText(display, "outside ws (still tracking)",
                                (int(x1), int(y2) + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)

            # Draw workspace polygon for reference
            if self.ws_polygon is not None:
                cv2.polylines(display, [self.ws_polygon], True, (255, 200, 0), 1)

            status = "TRACKING" if new_target is not None else "SEARCHING..."
            cv2.putText(display, status, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 0) if new_target else (0, 80, 255), 1)

            with self._lock:
                self._target = new_target
                self._frame  = display


# ==============================================================================
#  ARM HELPERS
# ==============================================================================

def get_ee_pos(kinematics, joints):
    T = kinematics.forward_kinematics(joints)
    return FRAME_R_RAW_TO_DISPLAY @ T[:3, 3]


def ik_step(kinematics, joints, target_display, ee_rot):
    target_raw = FRAME_R_DISPLAY_TO_RAW @ target_display
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = ee_rot
    pose[:3, 3]  = target_raw
    try:
        return kinematics.inverse_kinematics(
            joints, pose,
            position_weight=10.0,
            orientation_weight=0.01,
        )
    except Exception as e:
        print(f"[IK] {e}")
        return None


def build_action(joints, motor_names, home_last2):
    n = len(motor_names) - 2
    action = {f"{motor_names[i]}.pos": float(joints[i]) for i in range(n)}
    action[motor_names[-2] + ".pos"] = float(home_last2[0])
    action[motor_names[-1] + ".pos"] = float(home_last2[1])
    return action


def go_home(robot, current_joints, home_joints, motor_names, steps=60, step_sleep=0.04):
    """Linearly interpolate from current_joints back to home_joints, then hold."""
    print("[ARM] Returning to home position ...")
    for i in range(1, steps + 1):
        t = i / steps
        interp = current_joints + t * (home_joints - current_joints)
        action = {f"{m}.pos": float(interp[j]) for j, m in enumerate(motor_names)}
        try:
            robot.send_action(action)
        except Exception as e:
            print(f"[ARM] go_home step {i} failed: {e}")
            break
        time.sleep(step_sleep)
    print("[ARM] Home position reached.")


def find_urdf():
    candidates = [
        Path.home() / "SO-ARM100-main/Simulation/SO100/so100.urdf",
        Path.home() / "SO-ARM100/Simulation/SO100/so100.urdf",
        Path.home() / "Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf",
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
        description="Always chase the bottle bounding-box centre with the arm."
    )
    p.add_argument("--port",       default=None)
    p.add_argument("--id",         default=None)
    p.add_argument("--calibration-dir", default=None)
    p.add_argument("--urdf-path",  default=None)
    p.add_argument("--camera",     default=None)
    p.add_argument("--webcam",     action="store_true")
    p.add_argument("--conf",       type=float, default=0.35)
    p.add_argument("--fps",        type=int,   default=15)
    p.add_argument("--max-step",   type=float, default=0.03,
                   help="Max EE displacement per tick (m)")
    p.add_argument("--approach-z", type=float, default=0.10,
                   help="Fixed Z height to aim for (m)")
    p.add_argument("--pixel-x-range", type=float, nargs=2, default=[0.0, 0.4],
                   metavar=("X_MIN", "X_MAX"),
                   help="Arm X range (m) when no homography  (default 0.0 0.4)")
    p.add_argument("--pixel-y-range", type=float, nargs=2, default=[0.0, 0.3],
                   metavar=("Y_MIN", "Y_MAX"),
                   help="Arm Y range (m) when no homography  (default 0.0 0.3)")
    p.add_argument("--no-arm",     action="store_true")
    p.add_argument("--no-window",  action="store_true")
    return p.parse_args()


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    args = parse_args()

    use_arm = not args.no_arm and bool(args.port) and bool(args.id)
    if not use_arm:
        print("[INFO] Vision-only mode.")
    if use_arm and not _ARM_AVAILABLE:
        print("[ERROR] lerobot not importable -- use --no-arm.")
        sys.exit(1)

    # -- Model -----------------------------------------------------------------
    print(f"[INFO] Loading YOLO from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    bottle_ids = [cid for cid, n in model.names.items() if n == "bottle"]
    print(f"[INFO] Model ready -- tracking bottle  (IDs: {bottle_ids})")

    # -- Calibration -----------------------------------------------------------
    cam_matrix, dist_coeffs, focal_px = load_camera_calibration()
    ws_polygon, homography, ws_m      = load_workspace_calibration()

    if homography is None:
        print(f"[INFO] No homography -- pixel→arm mapping: "
              f"X{args.pixel_x_range}  Y{args.pixel_y_range} m")

    # -- Camera ----------------------------------------------------------------
    cam_path = (
        "/dev/video0" if args.webcam
        else args.camera if args.camera
        else find_robot_camera()
    )
    cap = open_camera(cam_path)

    # -- Detection thread ------------------------------------------------------
    det = DetectionThread(
        cap, model, cam_matrix, dist_coeffs,
        ws_polygon, homography, focal_px,
        conf_thresh=args.conf,
        pixel_x_range=args.pixel_x_range,
        pixel_y_range=args.pixel_y_range,
    )

    # -- Arm -------------------------------------------------------------------
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
            dtype=float,
        )
        print(f"[ARM] Joints: {joints}")

        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="jaw",
            joint_names=motor_names,
        )
        T0             = kinematics.forward_kinematics(joints)
        initial_ee_rot = T0[:3, :3].copy()
        home_last2     = joints[-2:].copy()
        home_joints    = joints.copy()    # saved for return-to-home
        print(f"[ARM] Ready. Wrist/gripper locked at {home_last2}")

    # --------------------------------------------------------------------------
    det.start()
    dt                    = 1.0 / args.fps
    zoom                  = ZOOM_DEFAULT
    joint_refresh_counter = 0

    print("\n[INFO] Running -- 'q' quit  '+'/'-' zoom  Ctrl+C stop\n")

    try:
        while det.running:
            tick = time.perf_counter()
            tgt  = det.target   # (x_m, y_m, depth_m) or None

            # -- Arm: always move toward bottle centre -------------------------
            if use_arm:
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
                    target_disp = np.array([x_m, y_m, args.approach_z])
                    ee_pos      = get_ee_pos(kinematics, joints)
                    direction   = target_disp - ee_pos
                    dist        = np.linalg.norm(direction)
                    if dist > args.max_step:
                        direction = direction / dist * args.max_step
                    sol = ik_step(kinematics, joints,
                                  ee_pos + direction, initial_ee_rot)
                    if sol is not None:
                        robot.send_action(build_action(sol, motor_names, home_last2))
                        joints = sol
                        print(f"[TRACK] X={x_m:.3f} Y={y_m:.3f}  "
                              f"EE=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f})  "
                              f"step={min(dist, args.max_step):.3f}m")
                    else:
                        robot.send_action(build_action(joints, motor_names, home_last2))
                else:
                    # Hold current position while searching
                    robot.send_action(build_action(joints, motor_names, home_last2))

            # -- Display -------------------------------------------------------
            if not args.no_window:
                frame = det.latest_frame
                if frame is not None:
                    if zoom > 1.0:
                        frame = crop_zoom(frame, zoom)
                    arm_lbl = "ARM ON" if use_arm else "NO ARM"
                    cv2.putText(frame,
                                f"BOTTLE TRACK V2  zoom {zoom:.1f}x  {arm_lbl}",
                                (6, CAM_HEIGHT - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
                    cv2.imshow("LAMP Bottle Tracking V2", frame)

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
            if home_joints is not None and joints is not None:
                go_home(robot, joints, home_joints, motor_names)
            print("[ARM] Disconnecting ...")
            robot.disconnect()
            print("[ARM] Disconnected.")
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
