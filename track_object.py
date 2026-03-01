#!/usr/bin/env python
"""
track_object.py — YOLO Object Detection + SO-100 Arm Tracking
==============================================================
Built directly on main_v2.py (single-threaded loop, same camera
open sequence).  Arm control is added inline; if loading fails
the script continues as a plain vision tool.

Usage
-----
# Vision only
python track_object.py --camera /dev/video2

# Full arm tracking
python track_object.py \
    --camera    /dev/video2 \
    --port      /dev/ttyACM0 \
    --id        my_lamp \
    --urdf-path ~/Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf
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
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Resolved relative to THIS file (LAMP root), not vision/
_DIR                 = os.path.dirname(os.path.abspath(__file__))
VISION_DIR           = os.path.join(_DIR, "vision")
MODEL_PATH           = os.path.join(VISION_DIR, "yolo11n.pt")
CAMERA_CALIB_FILE    = os.path.join(VISION_DIR, "camera_calibration.json")
WORKSPACE_CALIB_FILE = os.path.join(VISION_DIR, "workspace_calibration.json")

ROBOT_CAM_WIDTH  = 640
ROBOT_CAM_HEIGHT = 480

ZOOM_DEFAULT = 1.0
ZOOM_MIN     = 1.0
ZOOM_MAX     = 5.0
ZOOM_STEP    = 0.25

OBJECT_WIDTHS_M = {
    "book":       0.21,
    "bottle":     0.07,
    "cell phone": 0.075,
    "cup":        0.08,
    "remote":     0.05,
    "keyboard":   0.38,
    "mouse":      0.07,
    "scissors":   0.09,
    "vase":       0.12,
    "bowl":       0.15,
}
FALLBACK_WIDTH_M        = 0.15
FOCAL_LENGTH_PX         = 600.0
GRIPPER_APPROACH_Z_OFFSET = 0.05   # metres above workspace surface

# Frame-rotation matrices for arm coordinate transform
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


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION LOADERS  (verbatim from main_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_camera_calibration():
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
    if not os.path.exists(WORKSPACE_CALIB_FILE):
        print(f"[CALIB] {WORKSPACE_CALIB_FILE} not found — workspace mapping DISABLED.")
        return None, None, None
    with open(WORKSPACE_CALIB_FILE) as f:
        data = json.load(f)
    pixel_poly = np.array(data["pixel_corners"], dtype=np.float32)
    homography = np.array(data["homography"],    dtype=np.float64)
    ws_m       = data["workspace_m"]
    print(f"[CALIB] Workspace loaded — "
          f"{ws_m[0]*100:.0f} cm x {ws_m[1]*100:.0f} cm  (homography ready)")
    return pixel_poly.reshape(-1, 1, 2).astype(np.int32), homography, ws_m


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA  (verbatim from main_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

def find_video_devices():
    found = []
    for path in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ROBOT_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ROBOT_CAM_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            found.append((path, w, h))
    return found


def find_robot_camera():
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
    for _ in range(20):
        cap.grab()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode = "Robot cam" if robot_mode else "Webcam"
    print(f"[CAM] {mode} opened: {path}  {w}x{h}")
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  3D COORDINATE HELPERS  (verbatim from main_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_world_xy(px, py, homography):
    pt_h = homography @ np.array([px, py, 1.0], dtype=np.float64)
    return float(pt_h[0] / pt_h[2]), float(pt_h[1] / pt_h[2])


def estimate_z(bbox_width_px, class_name, fx):
    if bbox_width_px < 1:
        return float("inf")
    real_w = OBJECT_WIDTHS_M.get(class_name, FALLBACK_WIDTH_M)
    return (real_w * fx) / bbox_width_px


def is_inside_workspace(px, py, polygon):
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


def crop_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    cw, ch = int(w / zoom), int(h / zoom)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(frame[y0:y0+ch, x0:x0+cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
#  ARM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_current_ee_pos(kinematics, joints):
    T = kinematics.forward_kinematics(joints)
    return FRAME_R_RAW_TO_DISPLAY @ T[:3, 3]


def ik_step(kinematics, joints, target_disp, ee_rot):
    target_raw = FRAME_R_DISPLAY_TO_RAW @ target_disp
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
    """Build action dict: first N-2 joints from IK, last 2 locked to home."""
    n = len(motor_names) - 2
    action = {f"{motor_names[i]}.pos": float(joints[i]) for i in range(n)}
    action[motor_names[-2] + ".pos"] = float(home_last2[0])
    action[motor_names[-1] + ".pos"] = float(home_last2[1])
    return action


def find_urdf_auto():
    candidates = [
        Path.home() / "Documents/NYU/ITP/SO-ARM100/Simulation/SO100/so100.urdf",
        Path.home() / "SO-ARM100/Simulation/SO100/so100.urdf",
        Path.home() / "SO-ARM100-main/Simulation/SO100/so100.urdf",
        Path("/opt/placo/models/so100/so100.urdf"),
    ]
    for p in candidates:
        if p.exists():
            print(f"[URDF] Found: {p}")
            return str(p)
    return "so100"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  (structure identical to main_v2.py, arm control added inline)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam",     action="store_true")
    parser.add_argument("--camera",     default=None)
    parser.add_argument("--port",       default=None,
                        help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--id",         default=None,
                        help="Robot calibration ID, e.g. my_lamp")
    parser.add_argument("--urdf-path",  default=None)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--conf",       type=float, default=0.40)
    parser.add_argument("--max-step",   type=float, default=0.02,
                        help="Max EE movement per frame (m)")
    parser.add_argument("--deadzone",   type=float, default=0.005,
                        help="Min target shift to trigger move (m)")
    parser.add_argument("--approach-z", type=float,
                        default=GRIPPER_APPROACH_Z_OFFSET)
    args = parser.parse_args()

    target_classes = list(OBJECT_WIDTHS_M.keys())

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    yolo_class_ids = [cid for cid, name in model.names.items()
                      if name in target_classes]
    if yolo_class_ids:
        print(f"[INFO] Model ready -- tracking: {target_classes}")
    else:
        print("[WARN] Falling back to detecting ALL classes")
        yolo_class_ids = None

    # ── Calibrations ──────────────────────────────────────────────────────────
    cam_matrix, dist_coeffs, focal_px = load_camera_calibration()
    ws_polygon, homography, ws_m      = load_workspace_calibration()

    # ── Camera (identical to main_v2.py) ──────────────────────────────────────
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

    mode_label = "ROBOT CAM" if robot_mode else "WEBCAM"

    # ── Arm setup (lazy — failure keeps vision running) ───────────────────────
    arm_ok         = False
    robot          = None
    kinematics     = None
    current_joints = None
    motor_names    = None
    initial_ee_rot = None
    home_last2     = None

    arm_requested = args.port is not None and args.id is not None
    if arm_requested:
        try:
            from lerobot.model.kinematics import RobotKinematics
            from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

            urdf_path = (str(Path(args.urdf_path).expanduser())
                         if args.urdf_path else find_urdf_auto())
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
            current_joints = np.array(
                [float(obs[f"{m}.pos"]) for m in motor_names
                 if f"{m}.pos" in obs],
                dtype=float,
            )
            print(f"[ARM] Joints: {current_joints}")

            kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="jaw",
                joint_names=motor_names,
            )
            T0             = kinematics.forward_kinematics(current_joints)
            initial_ee_rot = T0[:3, :3].copy()
            home_last2     = current_joints[-2:].copy()  # wrist/gripper locked here
            arm_ok         = True
            print("[ARM] Ready.")
            print(f"[ARM] Wrist/gripper locked at: {home_last2}")

        except Exception as e:
            print(f"[ARM] Setup failed: {e}")
            print("[ARM] Vision-only mode.")
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:
                    pass
            robot = None
    else:
        print("[INFO] No --port/--id — vision-only mode.")

    arm_label = "ARM OK" if arm_ok else ("ARM ERR" if arm_requested else "NO ARM")
    print(f"\n[INFO] Running -- {mode_label} {arm_label} -- 'q' quit  '+'/'-' zoom\n")

    joint_refresh_counter = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP  (single-threaded, identical to main_v2.py)
    # ══════════════════════════════════════════════════════════════════════════
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

            # ── YOLO ──────────────────────────────────────────────────────────
            results = model.predict(
                source=frame,
                imgsz=max(fw, fh),
                conf=args.conf,
                classes=yolo_class_ids,
                verbose=False,
            )

            best_box   = None
            best_conf  = 0.0
            best_class = None

            for result in results:
                for box in result.boxes:
                    cls_name = model.names[int(box.cls)]
                    if cls_name not in target_classes:
                        continue
                    conf = float(box.conf)
                    if conf > best_conf:
                        best_conf  = conf
                        best_box   = box.xyxy[0].cpu().numpy()
                        best_class = cls_name

            # ── 3D coords ──────────────────────────────────────────────────────
            target = None
            inside = False

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                obj_px = (x1 + x2) / 2.0
                obj_py = y2

                inside = (ws_polygon is None or
                          is_inside_workspace(obj_px, obj_py, ws_polygon))

                z_from_cam = estimate_z(x2 - x1, best_class, focal_px)

                if homography is not None:
                    x_m, y_m = pixel_to_world_xy(obj_px, obj_py, homography)
                    z_m = args.approach_z
                else:
                    x_m, y_m = float("nan"), float("nan")
                    z_m = z_from_cam

                target = dict(x_m=x_m, y_m=y_m, z_m=z_m, z_depth=z_from_cam)

                if inside and not np.isnan(x_m):
                    print(f"[TARGET] {best_class.upper()}  conf={best_conf:.2f}  "
                          f"X={x_m:.3f} m  Y={y_m:.3f} m  Z={z_m:.3f} m")

            # ── Arm control ────────────────────────────────────────────────────
            if arm_ok:
                # Periodically refresh joints from robot to prevent drift
                joint_refresh_counter += 1
                if joint_refresh_counter >= 30:
                    joint_refresh_counter = 0
                    try:
                        obs = robot.get_observation()
                        current_joints = np.array(
                            [float(obs[f"{m}.pos"]) for m in motor_names
                             if f"{m}.pos" in obs], dtype=float)
                    except Exception:
                        pass

                if (target is not None and inside
                        and not np.isnan(target["x_m"])):
                    tgt = np.array([target["x_m"], target["y_m"], args.approach_z])
                    ee  = get_current_ee_pos(kinematics, current_joints)
                    d   = tgt - ee
                    dist = np.linalg.norm(d)
                    if dist > args.max_step:
                        d = d / dist * args.max_step
                    sol = ik_step(kinematics, current_joints, ee + d, initial_ee_rot)
                    if sol is not None:
                        # Singularity guard: skip if any joint jumps > 45 deg
                        max_delta = float(np.max(np.abs(sol - current_joints)))
                        if max_delta > 45.0:
                            print(f"[IK] Singularity guard: {max_delta:.1f}deg jump — skipped.")
                            robot.send_action(build_action(current_joints, motor_names, home_last2))
                        else:
                            robot.send_action(build_action(sol, motor_names, home_last2))
                            current_joints = sol
                            print(f"[TRACK] {best_class.upper()}  "
                                  f"XY=({target['x_m']:.3f},{target['y_m']:.3f})m  "
                                  f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})m  "
                                  f"dist={dist:.3f}m")
                    else:
                        # IK failed — hold position
                        robot.send_action(build_action(current_joints, motor_names, home_last2))
                else:
                    # No valid target — hold position
                    robot.send_action(build_action(current_joints, motor_names, home_last2))

            # ── Display (identical to main_v2.py) ──────────────────────────────
            display = frame.copy()

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

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                box_color = (0, 220, 0) if inside else (0, 0, 220)

                cv2.rectangle(display,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              box_color, 2)

                lbl1 = f"{best_class} {best_conf:.2f}"
                cv2.putText(display, lbl1,
                            (int(x1), max(int(y1) - 20, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                if target is not None and not np.isnan(target["x_m"]):
                    lbl2 = (f"X={target['x_m']:.3f}m  "
                            f"Y={target['y_m']:.3f}m  "
                            f"Z={target['z_m']:.3f}m")
                else:
                    lbl2 = f"depth~{target['z_depth']:.2f}m" if target else ""
                cv2.putText(display, lbl2,
                            (int(x1), max(int(y1) - 5, 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1)

                if not inside:
                    cv2.putText(display, "OUTSIDE WORKSPACE",
                                (int(x1), int(y2) + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 220), 1)

                cx_px = int((x1 + x2) / 2)
                cy_px = int(y2)
                cv2.drawMarker(display, (cx_px, cy_px), box_color,
                               cv2.MARKER_CROSS, 12, 2)

            arm_label = "ARM OK" if arm_ok else ("ARM ERR" if arm_requested else "NO ARM")
            cv2.putText(display,
                        f"{mode_label}  zoom {zoom:.2f}x  {arm_label}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255) if robot_mode else (100, 100, 255), 1)

            cal_status = ("CAM+WS" if cam_matrix is not None and homography is not None
                          else "CAM only" if cam_matrix is not None
                          else "NO CALIB")
            cv2.putText(display, cal_status,
                        (fw - 80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 220, 120), 1)

            cv2.imshow("LAMP — Track Object", display)
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
        if robot is not None:
            try:
                print("[ARM] Disconnecting ...")
                robot.disconnect()
                print("[ARM] Disconnected.")
            except Exception:
                pass
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
