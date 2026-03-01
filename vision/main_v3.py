"""
Bottle Follower — YOLO11n  (V3: single-target continuous tracking)
===================================================================
Camera  : SO-100 robot camera (UVC), falls back to laptop webcam
Model   : yolo11n.pt
Target  : bottle  (single class — arm follows the bottle continuously)
Output  : Real-world (X, Y, Z) in metres, streamed to move_to_3d_location.py

Coordinate system
-----------------
  X  — workspace horizontal (metres from left ArUco marker edge)
  Y  — workspace vertical   (metres from top  ArUco marker edge)
  Z  — height above workspace surface (metres, estimated from depth)

  X and Y are computed via the homography in workspace_calibration.json.
  Z is estimated with the pinhole formula:
      Z = (known_object_width_m × fx) / bbox_width_px

Tracking behaviour
------------------
  • Only the highest-confidence bottle per frame is tracked.
  • A simple exponential moving average (EMA) smooths jitter in X/Y.
  • The arm is updated at most every ARM_UPDATE_INTERVAL_S seconds to
    avoid flooding the controller.
  • If the bottle disappears for more than LOST_TIMEOUT_S seconds the
    tracker resets and prints a "LOST" message.

Usage
-----
python main_v3.py                        # auto-detect camera
python main_v3.py --webcam               # force laptop webcam
python main_v3.py --camera /dev/video2   # pick device path manually
python main_v3.py --smooth 0.6           # EMA alpha (0=max smooth, 1=raw)
python main_v3.py --interval 0.5         # arm update interval in seconds

Install
-------
pip install numpy opencv-python ultralytics
"""

import sys
import argparse
import time
import os
import json
import glob
import numpy as np
import cv2
from ultralytics import YOLO
import cv2.aruco


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH        = "yolo11n.pt"
CONFIDENCE_THRESH = 0.35          # slightly lower so bottle is caught early

ROBOT_CAM_WIDTH   = 640
ROBOT_CAM_HEIGHT  = 480

ZOOM_DEFAULT = 1.0
ZOOM_MIN     = 1.0
ZOOM_MAX     = 5.0
ZOOM_STEP    = 0.25

# Target class — only bottle is tracked
TARGET_CLASS      = "bottle"
BOTTLE_WIDTH_M    = 0.07   # ~standard 500 ml PET bottle

FALLBACK_WIDTH_M  = 0.07
FOCAL_LENGTH_PX   = 600.0  # overridden by camera_calibration.json

# Fixed gripper approach height above the detected object surface (metres)
GRIPPER_APPROACH_Z_OFFSET = 0.05   # 5 cm above estimated surface

# Tracking / update rate parameters (overridable via CLI)
EMA_ALPHA_DEFAULT         = 0.5    # exponential moving average weight [0,1]
ARM_UPDATE_INTERVAL_S     = 0.4    # minimum seconds between arm commands
LOST_TIMEOUT_S            = 1.5    # seconds before declaring bottle lost

# Calibration files — resolved relative to this script's directory
_DIR                 = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH           = os.path.join(_DIR, MODEL_PATH)   # resolve next to script
CAMERA_CALIB_FILE    = os.path.join(_DIR, "camera_calibration.json")
WORKSPACE_CALIB_FILE = os.path.join(_DIR, "workspace_calibration.json")


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_camera_calibration():
    """Returns (cam_matrix, dist_coeffs, fx) or (None, None, FOCAL_LENGTH_PX)."""
    if not os.path.exists(CAMERA_CALIB_FILE):
        print(f"[CALIB] {CAMERA_CALIB_FILE} not found — using default fx={FOCAL_LENGTH_PX}")
        return None, None, FOCAL_LENGTH_PX
    with open(CAMERA_CALIB_FILE) as f:
        data = json.load(f)
    cam_matrix  = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"],   dtype=np.float64)
    fx = cam_matrix[0, 0]
    print(f"[CALIB] Camera intrinsics loaded — fx={fx:.1f} px")
    return cam_matrix, dist_coeffs, fx


def load_workspace_calibration():
    """Returns (pixel_polygon, homography, workspace_wh_m) or (None, None, None)."""
    if not os.path.exists(WORKSPACE_CALIB_FILE):
        print(f"[CALIB] {WORKSPACE_CALIB_FILE} not found — workspace mapping DISABLED.")
        return None, None, None
    with open(WORKSPACE_CALIB_FILE) as f:
        data = json.load(f)
    pixel_poly = np.array(data["pixel_corners"], dtype=np.float32)
    homography = np.array(data["homography"],    dtype=np.float64)
    ws_m       = data["workspace_m"]              # [width_m, height_m]
    print(f"[CALIB] Workspace loaded — "
          f"{ws_m[0]*100:.0f} cm × {ws_m[1]*100:.0f} cm  "
          f"(homography ready)")
    return pixel_poly.reshape(-1, 1, 2).astype(np.int32), homography, ws_m


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════════════════════════════

def find_video_devices():
    """Return list of (path, w, h) for all usable /dev/videoX nodes.
    Skips metadata nodes (e.g. /dev/video1, /dev/video3) by verifying
    that an actual frame can be grabbed.
    """
    found = []
    for path in sorted(glob.glob("/dev/video*")):
        digits = ''.join(filter(str.isdigit, str(path).split('/')[-1]))
        idx = int(digits) if digits else 0
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ROBOT_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROBOT_CAM_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Attempt a real grab to filter out metadata-only nodes
        ok = cap.grab()
        cap.release()
        if w > 0 and h > 0 and ok:
            found.append((path, w, h))
    return found


def find_robot_camera():
    """Return path of first non-/dev/video0 device, or None."""
    print("[CAM] Scanning /dev/video* for robot camera ...")
    for path, w, h in find_video_devices():
        if path != "/dev/video0":
            print(f"[CAM] Robot camera found: {path} ({w}x{h})")
            return path
    print("[CAM] No robot camera found.")
    return None


def open_camera(path, robot_mode):
    digits = ''.join(filter(str.isdigit, str(path).split('/')[-1]))
    idx = int(digits) if digits else 0

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ROBOT_CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROBOT_CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Warm up — flush stale buffered frames
    for _ in range(20):
        cap.grab()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode = "Robot cam" if robot_mode else "Webcam"
    print(f"[CAM] {mode} opened: {path}  {w}x{h}")
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  3D COORDINATE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_world_xy(px, py, homography):
    """
    Apply the workspace homography to map a pixel (px, py)
    to real-world (X, Y) in metres on the table surface.
    """
    pt_h    = homography @ np.array([px, py, 1.0], dtype=np.float64)
    X_world = pt_h[0] / pt_h[2]
    Y_world = pt_h[1] / pt_h[2]
    return float(X_world), float(Y_world)


def estimate_z(bbox_width_px, fx):
    """
    Pinhole depth estimate:  Z = (real_width_m × fx) / bbox_width_px
    Returns estimated distance from camera (metres).
    """
    if bbox_width_px < 1:
        return float("inf")
    return (BOTTLE_WIDTH_M * fx) / bbox_width_px


def is_inside_workspace(px, py, polygon):
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  EMA TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class EMATracker:
    """
    Exponential Moving Average tracker for (X, Y, Z) world coordinates.
    Smooths noisy detections before sending to the arm.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha   # 0 = very smooth, 1 = raw measurements
        self._x = self._y = self._z = None

    def update(self, x: float, y: float, z: float):
        if self._x is None:
            self._x, self._y, self._z = x, y, z
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
            self._z = self.alpha * z + (1 - self.alpha) * self._z
        return self._x, self._y, self._z

    def reset(self):
        self._x = self._y = self._z = None

    @property
    def has_value(self):
        return self._x is not None


# ══════════════════════════════════════════════════════════════════════════════
#  VISION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def crop_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    cw, ch = int(w / zoom), int(h / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
#  ARM BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def send_to_arm(x_m: float, y_m: float, z_m: float):
    """
    Called whenever the smoothed bottle position changes enough to warrant
    an arm update (rate-limited by ARM_UPDATE_INTERVAL_S).

    Args
    ----
    x_m  — workspace X in metres (from left ArUco edge)
    y_m  — workspace Y in metres (from top  ArUco edge)
    z_m  — height above workspace surface in metres

    Replace the body with your arm control call, e.g.:

        import subprocess
        subprocess.Popen([
            "python", "move_to_3d_location.py",
            "--port",      "/dev/ttyACM0",
            "--id",        "my_lamp",
            "--urdf-path", "/path/to/so100.urdf",
            f"--target-x={x_m:.4f}",
            f"--target-y={y_m:.4f}",
            f"--target-z={z_m:.4f}",
            "--duration",  "0.4",
        ])
    """
    pass   # ← plug arm call in here


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V3 — bottle follower with EMA smoothing")
    parser.add_argument("--webcam",   action="store_true",
                        help="Force laptop webcam (/dev/video0)")
    parser.add_argument("--camera",   default=None,
                        help="Device path, e.g. /dev/video2")
    parser.add_argument("--smooth",   type=float, default=EMA_ALPHA_DEFAULT,
                        help=f"EMA alpha [0-1], default {EMA_ALPHA_DEFAULT}. "
                             "Lower = smoother but laggier.")
    parser.add_argument("--interval", type=float, default=ARM_UPDATE_INTERVAL_S,
                        help=f"Min seconds between arm updates "
                             f"(default {ARM_UPDATE_INTERVAL_S})")
    args = parser.parse_args()

    alpha    = float(np.clip(args.smooth, 0.01, 1.0))
    interval = max(0.05, args.interval)

    # -- Load model -----------------------------------------------------------
    print(f"[INFO] Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # Find YOLO class id for 'bottle'
    bottle_ids = [cid for cid, name in model.names.items()
                  if name == TARGET_CLASS]
    if bottle_ids:
        print(f"[INFO] Model ready — tracking: {TARGET_CLASS}  "
              f"(class id {bottle_ids})")
    else:
        print(f"[WARN] '{TARGET_CLASS}' not found in model class names — "
              "detecting ALL classes and filtering in post.")
        bottle_ids = None

    # ── Load calibrations ─────────────────────────────────────────────────────
    cam_matrix, dist_coeffs, focal_px = load_camera_calibration()
    ws_polygon, homography, ws_m      = load_workspace_calibration()

    # ── Pick camera ───────────────────────────────────────────────────────────
    if args.camera is not None:
        cam_path   = args.camera
        robot_mode = True
    elif args.webcam:
        cam_path   = "/dev/video0"
        robot_mode = False
        print("[CAM] Webcam mode forced.")
    else:
        robot_cam  = find_robot_camera()
        cam_path   = robot_cam if robot_cam is not None else "/dev/video0"
        robot_mode = robot_cam is not None
        if robot_cam is None:
            print("[CAM] Falling back to webcam (/dev/video0).")

    cap = open_camera(cam_path, robot_mode)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zoom = ZOOM_DEFAULT

    # ── Tracker & timing state ────────────────────────────────────────────────
    tracker        = EMATracker(alpha=alpha)
    last_arm_time  = 0.0
    last_seen_time = None       # time of last valid detection
    tracking_state  = "SEARCHING"   # "SEARCHING" | "TRACKING" | "LOST"

    mode_label = "ROBOT CAM" if robot_mode else "WEBCAM"
    print(f"\n[INFO] Running V3 BOTTLE FOLLOWER — {mode_label}")
    print(f"       EMA alpha={alpha}  arm interval={interval}s")
    print("       Press 'q' quit  '+'/'-' zoom  'r' reset tracker\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Frame grab failed -- retrying ...")
                time.sleep(0.05)
                continue

            frame = crop_zoom(frame, zoom)
            if cam_matrix is not None:
                frame = cv2.undistort(frame, cam_matrix, dist_coeffs)

            now = time.time()

            # ── YOLO inference ────────────────────────────────────────────────
            results = model.predict(
                source=frame,
                imgsz=max(fw, fh),
                conf=CONFIDENCE_THRESH,
                classes=bottle_ids,   # None = all, list = filtered
                verbose=False,
            )

            best_box  = None
            best_conf = 0.0

            for result in results:
                for box in result.boxes:
                    cls_name = model.names[int(box.cls)]
                    if cls_name != TARGET_CLASS:
                        continue
                    conf = float(box.conf)
                    if conf > best_conf:
                        best_conf = conf
                        best_box  = box.xyxy[0].cpu().numpy()

            # ── 3D coordinate computation ─────────────────────────────────────
            raw_target = None
            inside     = False

            if best_box is not None:
                x1, y1, x2, y2 = best_box

                # Bottom-centre of bbox = contact point with surface
                obj_px = (x1 + x2) / 2.0
                obj_py = y2

                inside = (ws_polygon is None or
                          is_inside_workspace(obj_px, obj_py, ws_polygon))

                z_from_cam = estimate_z(x2 - x1, focal_px)

                if homography is not None:
                    x_m, y_m = pixel_to_world_xy(obj_px, obj_py, homography)
                    z_m      = GRIPPER_APPROACH_Z_OFFSET
                else:
                    x_m, y_m = float("nan"), float("nan")
                    z_m      = z_from_cam

                raw_target = dict(x_m=x_m, y_m=y_m, z_m=z_m,
                                  z_depth=z_from_cam, inside=inside)
                last_seen_time = now

            # ── Tracking state machine ────────────────────────────────────────
            if raw_target is not None and raw_target["inside"]:
                x_s, y_s, z_s = tracker.update(
                    raw_target["x_m"], raw_target["y_m"], raw_target["z_m"])

                prev_state = tracking_state
                tracking_state = "TRACKING"
                if prev_state != "TRACKING":
                    print(f"[TRACK] Bottle acquired — "
                          f"X={x_s:.3f} m  Y={y_s:.3f} m  Z={z_s:.3f} m")

                # Rate-limited arm update
                if now - last_arm_time >= interval:
                    if not np.isnan(x_s):
                        print(f"[ARM]   → X={x_s:.4f}  Y={y_s:.4f}  Z={z_s:.4f}  "
                              f"(raw conf={best_conf:.2f})")
                    send_to_arm(x_s, y_s, z_s)
                    last_arm_time = now

            elif raw_target is not None and not raw_target["inside"]:
                # Detected but outside workspace — don't move arm
                tracking_state = "OUTSIDE"

            else:
                # No detection this frame
                if last_seen_time is not None and \
                        (now - last_seen_time) > LOST_TIMEOUT_S:
                    if tracking_state not in ("SEARCHING", "LOST"):
                        print(f"[TRACK] Bottle LOST (not seen for "
                              f"{LOST_TIMEOUT_S:.1f}s) — resetting tracker.")
                    tracking_state = "SEARCHING"
                    tracker.reset()
                    last_seen_time = None
                elif last_seen_time is None:
                    tracking_state = "SEARCHING"

            # ── Display ───────────────────────────────────────────────────────
            display = frame.copy()

            # Workspace boundary
            if ws_polygon is not None:
                cv2.polylines(display, [ws_polygon], isClosed=True,
                              color=(255, 200, 0), thickness=2)
                if ws_m is not None:
                    corners_px = ws_polygon.reshape(-1, 2)
                    labels = ["(0,0)", f"({ws_m[0]:.2f},0)",
                              f"({ws_m[0]:.2f},{ws_m[1]:.2f})",
                              f"(0,{ws_m[1]:.2f})"]
                    for (cx, cy), lbl in zip(corners_px, labels):
                        cv2.putText(display, lbl, (int(cx)+4, int(cy)-4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (255, 200, 0), 1)

            # Detection box
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                box_color = (0, 220, 0) if inside else (0, 0, 220)

                cv2.rectangle(display,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              box_color, 2)

                lbl1 = f"bottle {best_conf:.2f}"
                cv2.putText(display, lbl1,
                            (int(x1), max(int(y1) - 20, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                if raw_target is not None and not np.isnan(raw_target["x_m"]) \
                        and tracker.has_value:
                    xs, ys, zs = tracker._x, tracker._y, tracker._z
                    lbl2 = f"X={xs:.3f}m  Y={ys:.3f}m  Z={zs:.3f}m"
                elif raw_target is not None:
                    lbl2 = f"depth~{raw_target['z_depth']:.2f}m"
                else:
                    lbl2 = ""
                cv2.putText(display, lbl2,
                            (int(x1), max(int(y1) - 5, 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1)

                if not inside:
                    cv2.putText(display, "OUTSIDE WORKSPACE",
                                (int(x1), int(y2) + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 220), 1)

                # Cross-hair at bottom-centre
                cx_px = int((x1 + x2) / 2)
                cy_px = int(y2)
                cv2.drawMarker(display, (cx_px, cy_px), box_color,
                               cv2.MARKER_CROSS, 12, 2)

            # Tracking state badge
            state_colors = {
                "TRACKING":  (0, 220, 0),
                "OUTSIDE":   (0, 165, 255),
                "LOST":      (0, 0, 220),
                "SEARCHING": (200, 200, 0),
            }
            state_color = state_colors.get(tracking_state, (200, 200, 200))
            cv2.putText(display, tracking_state,
                        (fw // 2 - 40, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

            # Mode / zoom HUD
            cv2.putText(display,
                        f"{mode_label}  zoom {zoom:.2f}x",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255) if robot_mode else (100, 100, 255), 1)

            # Calibration status
            cal_status = ("CAM+WS"   if cam_matrix is not None and homography is not None
                          else "CAM only" if cam_matrix is not None
                          else "NO CALIB")
            cv2.putText(display, cal_status,
                        (fw - 80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 220, 120), 1)

            # EMA smoothing indicator
            cv2.putText(display, f"EMA a={alpha:.2f}",
                        (6, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (180, 180, 180), 1)

            cv2.imshow("LAMP V3 — Bottle Follower", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                zoom = min(zoom + ZOOM_STEP, ZOOM_MAX)
                print(f"[ZOOM] {zoom:.2f}x")
            elif key == ord("-"):
                zoom = max(zoom - ZOOM_STEP, ZOOM_MIN)
                print(f"[ZOOM] {zoom:.2f}x")
            elif key == ord("r"):
                tracker.reset()
                last_seen_time = None
                tracking_state = "SEARCHING"
                print("[TRACK] Tracker manually reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
