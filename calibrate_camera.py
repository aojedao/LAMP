"""
Camera Calibration — Standard Checkerboard
============================================
Uses your existing 4x5 inner corner checkerboard (3.5cm squares).

How to use
----------
1. Run this script
2. Hold the checkerboard in front of the camera
3. Press 'c' to capture each pose
4. Move to a different angle/distance between captures
5. Aim for 15-20 captures covering:
     - straight on
     - tilted left / right
     - tilted up / down
     - closer / further away
     - slight rotations
6. Press 's' to run calibration and save

Tips for 32x32 camera
----------------------
- Good even lighting — no shadows across the board
- Hold the board steady when pressing 'c'
- Fill as much of the frame as possible
- The board should be flat, not bent

Output
------
camera_calibration.json  — loaded by calibrate_workspace.py and main.py

Usage
-----
python calibrate_camera.py
"""

import cv2
import numpy as np
import json
import argparse

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — matches your physical board
# ══════════════════════════════════════════════════════════════════════════════

# Inner corners — NOT the number of squares, but the intersections inside
# A board with 5 columns and 4 rows of squares has 4x3 inner corners... 
# but you measured 4x5 inner corners so we use that directly.
CORNERS_COLS    = 4       # inner corners horizontally
CORNERS_ROWS    = 5       # inner corners vertically

SQUARE_SIZE_M   = 0.035   # 3.5 cm in metres

CAM_WIDTH       = 640
CAM_HEIGHT      = 480
MIN_CAPTURES    = 15      # minimum for reliable calibration
OUTPUT_FILE     = "camera_calibration.json"


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKERBOARD SETUP
# ══════════════════════════════════════════════════════════════════════════════

# 3D object points for one board pose — same for every capture
# (0,0,0), (1,0,0), (2,0,0) ... scaled by square size
objp = np.zeros((CORNERS_COLS * CORNERS_ROWS, 3), np.float32)
objp[:, :2] = np.mgrid[0:CORNERS_COLS, 0:CORNERS_ROWS].T.reshape(-1, 2)
objp        *= SQUARE_SIZE_M


def find_checkerboard(gray):
    """
    Attempt to find checkerboard corners in a grayscale frame.
    Returns (True, corners) or (False, None).
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +   # handles uneven lighting
        cv2.CALIB_CB_NORMALIZE_IMAGE  +  # normalise before thresholding
        cv2.CALIB_CB_FAST_CHECK           # quick rejection of non-board frames
    )
    found, corners = cv2.findChessboardCorners(
        gray, (CORNERS_COLS, CORNERS_ROWS), flags
    )
    if found:
        # Subpixel refinement — improves accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
    return found, corners


def run_calibration(obj_points, img_points, frame_size):
    """Run OpenCV camera calibration from collected point pairs."""
    print(f"\n[CALIB] Running calibration on {len(obj_points)} captures ...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, frame_size, None, None
    )
    return ret, camera_matrix, dist_coeffs


def save_calibration(camera_matrix, dist_coeffs, frame_size, reprojection_error):
    calib = {
        "frame_size":         list(frame_size),
        "camera_matrix":      camera_matrix.tolist(),
        "dist_coeffs":        dist_coeffs.tolist(),
        "reprojection_error": float(reprojection_error),
        "checkerboard": {
            "cols":        CORNERS_COLS,
            "rows":        CORNERS_ROWS,
            "square_size": SQUARE_SIZE_M,
        },
        "notes": (
            "camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  |  "
            "dist_coeffs: [k1, k2, p1, p2, k3]"
        )
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\n[SAVED] {OUTPUT_FILE}")
    print(f"  fx={camera_matrix[0,0]:.2f}  fy={camera_matrix[1,1]:.2f}  "
          f"cx={camera_matrix[0,2]:.2f}  cy={camera_matrix[1,2]:.2f}")
    print(f"  Reprojection error: {reprojection_error:.4f} px  "
          f"({'great' if reprojection_error < 0.5 else 'good' if reprojection_error < 1.0 else 'acceptable — try more captures'})")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def scan_cameras(max_index=10):
    """Return a list of (index, width, height) for every camera that can be opened."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        found.append((idx, w, h))
    return found


def pick_camera(cameras):
    """Let the user choose when multiple cameras are available."""
    if not cameras:
        raise RuntimeError("No cameras detected.")
    if len(cameras) == 1:
        idx, w, h = cameras[0]
        print(f"[CAM] Only one camera found — using index {idx} ({w}x{h})")
        return idx

    print("\n[CAM] Multiple cameras detected:")
    for i, (idx, w, h) in enumerate(cameras):
        label = "(likely laptop webcam)" if idx == 0 else "(likely external / robot cam)"
        print(f"  {i+1})  index {idx}  —  {w}x{h}  {label}")

    while True:
        try:
            choice = int(input(f"\nChoose camera [1-{len(cameras)}]: "))
            if 1 <= choice <= len(cameras):
                return cameras[choice - 1][0]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index — skip auto-detection and use this directly")
    args = parser.parse_args()

    # ── Pick camera ───────────────────────────────────────────────────────────
    if args.camera is not None:
        cam_idx = args.camera
        print(f"[CAM] Using camera index {cam_idx} (from --camera flag)")
    else:
        print("[CAM] Scanning for available cameras ...")
        cameras = scan_cameras()
        cam_idx = pick_camera(cameras)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cam_idx}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    print(f"[CAM] Opened camera {cam_idx} at {w}x{h}")
    print("=" * 55)
    print(f"  Board: {CORNERS_COLS}x{CORNERS_ROWS} inner corners, "
          f"{SQUARE_SIZE_M*100:.0f}cm squares")
    print(f"  Target: {MIN_CAPTURES} captures minimum")
    print()
    print(f"  'c'  — capture current pose")
    print(f"  's'  — finish and save calibration")
    print(f"  'q'  — quit without saving")
    print("=" * 55 + "\n")

    obj_points     = []   # 3D world points per capture
    img_points     = []   # 2D image points per capture
    capture_count  = 0
    force_save     = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_checkerboard(gray)

        # ── Display ───────────────────────────────────────────────────────────
        display = frame.copy()

        if found:
            cv2.drawChessboardCorners(
                display, (CORNERS_COLS, CORNERS_ROWS), corners, found
            )
            status_text  = f"Board found! Press 'c' to capture  [{capture_count}/{MIN_CAPTURES}]"
            status_color = (0, 255, 0)
        else:
            status_text  = f"No board detected  [{capture_count}/{MIN_CAPTURES}]"
            status_color = (0, 100, 255)

        cv2.putText(display, status_text, (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        # Progress bar at bottom
        bar_max = w - 8
        bar_w   = int(bar_max * min(capture_count / MIN_CAPTURES, 1.0))
        bar_y   = h - 5
        cv2.rectangle(display, (4, bar_y - 7), (4 + bar_w, bar_y),
                      (0, 200, 100), -1)

        cv2.imshow("Camera Calibration — Checkerboard", display)
        key = cv2.waitKey(1) & 0xFF

        # ── Capture ───────────────────────────────────────────────────────────
        if key == ord("c"):
            if found:
                obj_points.append(objp)
                img_points.append(corners)
                capture_count += 1
                print(f"[CAPTURE {capture_count:02d}]  Board detected — saved.")
                if capture_count == MIN_CAPTURES:
                    print(f"\n[INFO] {MIN_CAPTURES} captures reached — "
                          f"press 's' to calibrate or keep going for better accuracy.")
            else:
                print("[WARN] Board not visible — adjust position and try again.")

        # ── Calibrate and save ────────────────────────────────────────────────
        elif key == ord("s"):
            if capture_count < MIN_CAPTURES and not force_save:
                print(f"[WARN] Only {capture_count}/{MIN_CAPTURES} captures. "
                      f"Press 's' again to force save, or keep capturing.")
                force_save = True
            elif capture_count < 4:
                print("[ERROR] Need at least 4 captures — keep going.")
            else:
                try:
                    error, cam_matrix, dist = run_calibration(
                        obj_points, img_points, frame_size
                    )
                    save_calibration(cam_matrix, dist, frame_size, error)
                    print("\n[OK] Done. Next step:")
                    print("     python calibrate_workspace.py")
                    break
                except Exception as e:
                    print(f"[ERROR] Calibration failed: {e}")
                    print("        Try more captures with varied angles.")

        elif key == ord("q"):
            print("[INFO] Quit — nothing saved.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()