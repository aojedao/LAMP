"""
Book / Object Detection — YOLO11n  (V2: full 3D world coordinates)
===================================================================
Camera  : SO-100 robot camera (UVC), falls back to laptop webcam
Model   : yolo11n.pt
Output  : Real-world (X, Y, Z) in metres, ready for move_to_3d_location.py

Coordinate system
-----------------
  X  — workspace horizontal (metres from left ArUco marker edge)
  Y  — workspace vertical   (metres from top  ArUco marker edge)
  Z  — height above workspace surface (metres, estimated from depth)

  X and Y are computed via the homography in workspace_calibration.json.
  Z is estimated with the pinhole formula:
      Z = (known_object_width_m × fx) / bbox_width_px

Usage
-----
python main_v2.py                  # auto-detect camera
python main_v2.py --webcam         # force laptop webcam
python main_v2.py --camera 2       # pick camera index manually

Install
-------
pip install numpy opencv-python ultralytics
"""

import sys
import argparse
import time
import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
import cv2.aruco


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH        = "yolo11n.pt"
CONFIDENCE_THRESH = 0.40
TARGET_CLASSES    = ["book", "bottle"]

ROBOT_CAM_WIDTH   = 640
ROBOT_CAM_HEIGHT  = 480

ZOOM_DEFAULT = 1.0
ZOOM_MIN     = 1.0
ZOOM_MAX     = 5.0
ZOOM_STEP    = 0.25

# Known real-world widths for depth (Z) estimation — add more as needed
OBJECT_WIDTHS_M = {
    "book":   0.21,   # ~A5 book width
    "bottle": 0.07,   # ~standard 500ml bottle
}
FALLBACK_WIDTH_M  = 0.15
FOCAL_LENGTH_PX   = 600.0   # overridden by camera_calibration.json

# Fixed gripper approach height above the detected object surface (metres)
# The arm will aim for the object centre + this offset so the gripper
# clears the surface before descending.
GRIPPER_APPROACH_Z_OFFSET = 0.05   # 5 cm above estimated surface

# Calibration files — resolved relative to this script's directory
_DIR                 = os.path.dirname(os.path.abspath(__file__))
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

def find_robot_camera():
    print("[CAM] Scanning for robot camera ...")
    for idx in range(1, 6):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0 and frame.max() > 5:
            print(f"[CAM] Robot camera found at index {idx}")
            return idx
    print("[CAM] No robot camera found.")
    return None


def open_camera(index, robot_mode):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}.")
    if robot_mode:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ROBOT_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROBOT_CAM_HEIGHT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode = "Robot cam" if robot_mode else "Webcam"
    print(f"[CAM] {mode} opened at index {index} — {w}×{h}")
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  3D COORDINATE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_world_xy(px, py, homography):
    """
    Apply the workspace homography to map a pixel (px, py)
    to real-world (X, Y) in metres on the table surface.
    """
    pt_h   = homography @ np.array([px, py, 1.0], dtype=np.float64)
    X_world = pt_h[0] / pt_h[2]
    Y_world = pt_h[1] / pt_h[2]
    return float(X_world), float(Y_world)


def estimate_z(bbox_width_px, class_name, fx):
    """
    Pinhole depth estimate:  Z = (real_width_m × fx) / bbox_width_px
    Returns height above the camera (metres).
    """
    if bbox_width_px < 1:
        return float("inf")
    real_w = OBJECT_WIDTHS_M.get(class_name, FALLBACK_WIDTH_M)
    return (real_w * fx) / bbox_width_px


def is_inside_workspace(px, py, polygon):
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  VISION HELPERS
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

def send_to_arm(x_m, y_m, z_m):
    """
    Called every frame a valid in-workspace object is detected.

    Args
    ----
    x_m  — workspace X in metres (from left ArUco edge)
    y_m  — workspace Y in metres (from top  ArUco edge)
    z_m  — height above workspace surface in metres

    Replace the body with your arm control call, e.g.:

        import subprocess
        subprocess.Popen([
            "python", "move_to_3d_location.py",
            "--port", "/dev/ttyACM0",
            "--id",   "my_lamp",
            "--urdf-path", "/path/to/so100.urdf",
            f"--target-x={x_m:.4f}",
            f"--target-y={y_m:.4f}",
            f"--target-z={z_m:.4f}",
            "--duration", "3.0",
        ])
    """
    pass   # ← plug arm call in here


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam",    action="store_true",
                        help="Force laptop webcam")
    parser.add_argument("--camera",   type=int, default=None,
                        help="Manually specify a camera index")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    yolo_class_ids = [cid for cid, name in model.names.items()
                      if name in TARGET_CLASSES]
    print(f"[INFO] Model ready — tracking: {TARGET_CLASSES}  "
          f"(YOLO IDs: {yolo_class_ids})")

    # ── Load calibrations ─────────────────────────────────────────────────────
    cam_matrix, dist_coeffs, focal_px = load_camera_calibration()
    ws_polygon, homography, ws_m      = load_workspace_calibration()

    # ── Pick camera ───────────────────────────────────────────────────────────
    if args.camera is not None:
        cam_index  = args.camera
        robot_mode = True
    elif args.webcam:
        cam_index  = 0
        robot_mode = False
        print("[CAM] Webcam mode forced.")
    else:
        robot_cam = find_robot_camera()
        cam_index  = robot_cam if robot_cam is not None else 0
        robot_mode = robot_cam is not None
        if robot_cam is None:
            print("[CAM] Falling back to webcam (index 0).")

    cap = open_camera(cam_index, robot_mode)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zoom = ZOOM_DEFAULT

    mode_label = "ROBOT CAM" if robot_mode else "WEBCAM"
    print(f"\n[INFO] Running — {mode_label} — 'q' quit · '+'/'-' zoom\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            frame = crop_zoom(frame, zoom)
            if cam_matrix is not None:
                frame = cv2.undistort(frame, cam_matrix, dist_coeffs)

            # ── YOLO inference ────────────────────────────────────────────────
            results = model.predict(
                source=frame,
                imgsz=max(fw, fh),
                conf=CONFIDENCE_THRESH,
                classes=yolo_class_ids,
                verbose=False,
            )

            best_box   = None
            best_conf  = 0.0
            best_class = None

            for result in results:
                for box in result.boxes:
                    cls_name = model.names[int(box.cls)]
                    if cls_name in TARGET_CLASSES:
                        conf = float(box.conf)
                        if conf > best_conf:
                            best_conf  = conf
                            best_box   = box.xyxy[0].cpu().numpy()
                            best_class = cls_name

            # ── 3D coordinate computation ─────────────────────────────────────
            target    = None   # dict with x_m, y_m, z_m if valid detection
            inside    = False

            if best_box is not None:
                x1, y1, x2, y2 = best_box

                # Centre of bounding box bottom edge — closest point to table
                obj_px = (x1 + x2) / 2.0
                obj_py = y2           # bottom of bbox → table contact point

                inside = (ws_polygon is None or
                          is_inside_workspace(obj_px, obj_py, ws_polygon))

                # Z: depth from camera → height above surface
                z_from_cam = estimate_z(x2 - x1, best_class, focal_px)

                if homography is not None:
                    # Map bottom-centre pixel → real-world (X, Y) metres
                    x_m, y_m = pixel_to_world_xy(obj_px, obj_py, homography)
                    # Z above the table: camera_height - object_depth ≈ 0 for
                    # flat objects; we add an approach offset for the gripper
                    z_m = GRIPPER_APPROACH_Z_OFFSET
                else:
                    # No workspace calibration — use raw depth as Z, no X/Y
                    x_m, y_m = float("nan"), float("nan")
                    z_m = z_from_cam

                target = dict(x_m=x_m, y_m=y_m, z_m=z_m,
                              z_depth=z_from_cam)

                if inside:
                    if not np.isnan(x_m):
                        print(f"[TARGET] {best_class.upper()}  "
                              f"conf={best_conf:.2f}  "
                              f"X={x_m:.3f} m  Y={y_m:.3f} m  "
                              f"Z={z_m:.3f} m  "
                              f"(depth≈{z_from_cam:.2f} m)")
                        print(f"         → move_to_3d_location.py "
                              f"--target-x={x_m:.4f} "
                              f"--target-y={y_m:.4f} "
                              f"--target-z={z_m:.4f}")
                    else:
                        print(f"[TARGET] {best_class.upper()}  "
                              f"conf={best_conf:.2f}  "
                              f"depth≈{z_from_cam:.2f} m  "
                              f"(no workspace calibration — X/Y unavailable)")
                    send_to_arm(x_m, y_m, z_m)

            # ── Display ───────────────────────────────────────────────────────
            display = frame.copy()

            # Workspace boundary
            if ws_polygon is not None:
                cv2.polylines(display, [ws_polygon], isClosed=True,
                              color=(255, 200, 0), thickness=2)

                # World-coordinate grid labels at polygon corners
                if ws_m is not None:
                    corners_px = ws_polygon.reshape(-1, 2)
                    labels = ["(0,0)", f"({ws_m[0]:.2f},0)",
                              f"({ws_m[0]:.2f},{ws_m[1]:.2f})",
                              f"(0,{ws_m[1]:.2f})"]
                    for (cx, cy), lbl in zip(corners_px, labels):
                        cv2.putText(display, lbl, (int(cx)+4, int(cy)-4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (255, 200, 0), 1)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                box_color = (0, 220, 0) if inside else (0, 0, 220)

                cv2.rectangle(display,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              box_color, 2)

                # Label line 1: class + confidence
                lbl1 = f"{best_class} {best_conf:.2f}"
                cv2.putText(display, lbl1,
                            (int(x1), max(int(y1) - 20, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                # Label line 2: world coords
                if target is not None and not np.isnan(target["x_m"]):
                    lbl2 = (f"X={target['x_m']:.3f}m  "
                            f"Y={target['y_m']:.3f}m  "
                            f"Z={target['z_m']:.3f}m")
                else:
                    lbl2 = f"depth≈{target['z_depth']:.2f}m" if target else ""
                cv2.putText(display, lbl2,
                            (int(x1), max(int(y1) - 5, 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1)

                if not inside:
                    cv2.putText(display, "OUTSIDE WORKSPACE",
                                (int(x1), int(y2) + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 220), 1)

                # Cross-hair at bottom-centre (table contact point)
                cx_px = int((x1 + x2) / 2)
                cy_px = int(y2)
                cv2.drawMarker(display, (cx_px, cy_px), box_color,
                               cv2.MARKER_CROSS, 12, 2)

            # Mode / zoom HUD
            cv2.putText(display,
                        f"{mode_label}  zoom {zoom:.2f}x",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255) if robot_mode else (100, 100, 255), 1)

            # Calibration status
            cal_status = ("CAM+WS" if cam_matrix is not None and homography is not None
                          else "CAM only" if cam_matrix is not None
                          else "NO CALIB")
            cv2.putText(display, cal_status,
                        (fw - 80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 220, 120), 1)

            cv2.imshow("LAMP — Object Detection V2", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                zoom = min(zoom + ZOOM_STEP, ZOOM_MAX)
                print(f"[ZOOM] {zoom:.2f}x")
            elif key == ord("-"):
                zoom = max(zoom - ZOOM_STEP, ZOOM_MIN)
                print(f"[ZOOM] {zoom:.2f}x")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
