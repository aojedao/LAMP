"""
Workspace Calibration — 4 ArUco Markers
=========================================
Detects 4 ArUco markers (DICT_4X4_50, IDs 0-3) placed at the corners
of your workspace and saves the pixel ↔ real-world mapping.

Marker layout (looking down at the table):

    [ID 0] ──────────── [ID 1]
      │                    │
      │     workspace      │
      │     20 × 20 in     │
      │                    │
    [ID 3] ──────────── [ID 2]

Prerequisites
-------------
1. Print four 4x4 ArUco markers (IDs 0, 1, 2, 3)
   — use generate_markers.py or https://chev.me/arucogen/
2. Tape them at the four corners of your workspace
3. Run  python calibrate_camera.py  first (produces camera_calibration.json)

Usage
-----
python calibrate_workspace.py                      # auto-detect camera
python calibrate_workspace.py --camera /dev/video2  # force device path

Output
------
workspace_calibration.json
"""

import cv2
import cv2.aruco
import numpy as np
import json
import glob
import argparse
import os

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these to match YOUR workspace
# ══════════════════════════════════════════════════════════════════════════════

# Physical workspace dimensions (distance between marker centres)
WORKSPACE_W_M = 0.508   # 20 inches in metres
WORKSPACE_H_M = 0.508   # 20 inches in metres

# ArUco dictionary and expected marker IDs (top-left, top-right, bottom-right, bottom-left)
ARUCO_DICT    = cv2.aruco.DICT_4X4_50
MARKER_IDS    = [0, 1, 2, 3]   # clockwise from top-left

# Camera resolution request
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# Files
CAMERA_CALIB_FILE    = "camera_calibration.json"
WORKSPACE_CALIB_FILE = "workspace_calibration.json"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_camera_calibration():
    """Load intrinsics from camera_calibration.json if available."""
    if not os.path.exists(CAMERA_CALIB_FILE):
        print(f"[WARN] {CAMERA_CALIB_FILE} not found — skipping undistortion.")
        return None, None
    with open(CAMERA_CALIB_FILE) as f:
        data = json.load(f)
    cam_matrix  = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    print(f"[CALIB] Loaded camera intrinsics from {CAMERA_CALIB_FILE}")
    return cam_matrix, dist_coeffs


def scan_cameras():
    """Return list of (path, width, height) for usable /dev/videoX nodes."""
    found = []
    for path in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            found.append((path, w, h))
    return found


def pick_camera(cameras):
    """Let the user choose when multiple cameras are available."""
    if not cameras:
        raise RuntimeError("No video devices found under /dev/video*")
    if len(cameras) == 1:
        path, w, h = cameras[0]
        print(f"[CAM] Only one device found: {path} ({w}x{h})")
        return path

    print("\n[CAM] Available cameras:")
    for i, (path, w, h) in enumerate(cameras):
        note = "(built-in webcam)" if path == "/dev/video0" else "(external / robot cam)"
        print(f"  {i+1})  {path}  {w}x{h}  {note}")

    while True:
        try:
            choice = int(input(f"\nChoose camera [1-{len(cameras)}]: "))
            if 1 <= choice <= len(cameras):
                return cameras[choice - 1][0]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def marker_centre(corner):
    """Return the (x, y) centre of a detected marker corner array."""
    return corner[0].mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=None,
                        help="Device path, e.g. /dev/video2")
    args = parser.parse_args()

    # ── Camera ────────────────────────────────────────────────────────────────
    if args.camera is not None:
        cam_path = args.camera
        print(f"[CAM] Using {cam_path} (from --camera flag)")
    else:
        print("[CAM] Scanning /dev/video* ...")
        cameras  = scan_cameras()
        cam_path = pick_camera(cameras)

    cap = cv2.VideoCapture(cam_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {cam_path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Opened {cam_path} at {w}x{h}")

    # ── Camera calibration (optional, for undistortion) ───────────────────────
    cam_matrix, dist_coeffs = load_camera_calibration()

    # ── ArUco setup ───────────────────────────────────────────────────────────
    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Real-world coordinates of the 4 marker centres (metres, origin = top-left)
    world_points = np.array([
        [0.0,           0.0],            # ID 0 — top-left
        [WORKSPACE_W_M, 0.0],            # ID 1 — top-right
        [WORKSPACE_W_M, WORKSPACE_H_M],  # ID 2 — bottom-right
        [0.0,           WORKSPACE_H_M],  # ID 3 — bottom-left
    ], dtype=np.float32)

    saved = False

    print("\n" + "=" * 55)
    print("  Place 4 ArUco markers (IDs 0-3) at workspace corners")
    print(f"  Workspace size: {WORKSPACE_W_M*100:.1f} cm × {WORKSPACE_H_M*100:.1f} cm")
    print(f"                  ({WORKSPACE_W_M/0.0254:.0f}\" × {WORKSPACE_H_M/0.0254:.0f}\")")
    print()
    print("  's'  — save calibration when all 4 markers visible")
    print("  'q'  — quit")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Optional undistortion
        if cam_matrix is not None:
            frame = cv2.undistort(frame, cam_matrix, dist_coeffs)

        corners, ids, _ = detector.detectMarkers(frame)
        display = frame.copy()

        found_markers = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in MARKER_IDS:
                    centre = marker_centre(corners[i])
                    found_markers[int(marker_id)] = centre

        all_found = all(mid in found_markers for mid in MARKER_IDS)

        # Draw workspace outline if all 4 found
        if all_found:
            pts = np.array([found_markers[mid] for mid in MARKER_IDS], dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            status = "All 4 markers found — press 's' to save"
            colour = (0, 255, 0)
        else:
            missing = [mid for mid in MARKER_IDS if mid not in found_markers]
            status = f"Markers found: {len(found_markers)}/4 — missing IDs: {missing}"
            colour = (0, 100, 255)

        cv2.putText(display, status, (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cv2.imshow("Workspace Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            if not all_found:
                print("[WARN] Not all 4 markers visible — can't save yet.")
                continue

            # Build pixel points in same order as world_points
            pixel_points = np.array(
                [found_markers[mid] for mid in MARKER_IDS], dtype=np.float32
            )

            # Compute homography: pixel → world
            H, _ = cv2.findHomography(pixel_points, world_points)

            calib = {
                "workspace_m":   [WORKSPACE_W_M, WORKSPACE_H_M],
                "marker_ids":    MARKER_IDS,
                "pixel_corners": pixel_points.tolist(),
                "world_corners": world_points.tolist(),
                "homography":    H.tolist(),
                "frame_size":    [w, h],
                "notes": (
                    "homography maps pixel (x,y,1) → world (X,Y,1) in metres. "
                    "Origin is marker ID 0 (top-left)."
                ),
            }
            with open(WORKSPACE_CALIB_FILE, "w") as f:
                json.dump(calib, f, indent=2)

            print(f"\n[SAVED] {WORKSPACE_CALIB_FILE}")
            print(f"  Workspace: {WORKSPACE_W_M*100:.1f} cm × {WORKSPACE_H_M*100:.1f} cm")
            print(f"  Pixel corners: {pixel_points.tolist()}")
            print("\n[OK] Done. You can now run:  python main.py")
            saved = True
            break

    cap.release()
    cv2.destroyAllWindows()
    if not saved:
        print("[INFO] Quit — nothing saved.")


if __name__ == "__main__":
    main()
