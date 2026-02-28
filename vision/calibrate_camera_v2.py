"""
Camera Calibration — Linux, Checkerboard 4x5 inner corners, 3.5 cm squares
===========================================================================

Controls
--------
  c  — capture current board pose
  s  — run calibration and save
  q  — quit without saving

Target: 15+ varied poses (tilted, rotated, close/far, corners of frame)

Output
------
  camera_calibration.json  (same directory as this script)
"""

import cv2
import numpy as np
import json
import glob
import os
import argparse

# ─── Configuration ────────────────────────────────────────────────────────────

COLS         = 4        # inner corners horizontally
ROWS         = 5        # inner corners vertically
SQUARE_M     = 0.035    # square size in metres (3.5 cm)
MIN_CAPTURES = 15
CAM_W, CAM_H = 640, 480

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "camera_calibration.json")

# Reference 3-D points for one board (Z = 0 plane)
_objp = np.zeros((COLS * ROWS, 3), np.float32)
_objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_M

# ─── Camera discovery ─────────────────────────────────────────────────────────

def find_cameras():
    """Return list of (path, width, height) for usable /dev/videoX nodes."""
    found = []
    for path in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            found.append((path, w, h))
    return found


def choose_camera(args_camera):
    """Return device path string."""
    if args_camera:
        return args_camera

    cams = find_cameras()
    if not cams:
        raise RuntimeError("No video devices found under /dev/video*")
    if len(cams) == 1:
        path, w, h = cams[0]
        print(f"[cam] Only one device found: {path} ({w}x{h})")
        return path

    print("\nAvailable cameras:")
    for i, (path, w, h) in enumerate(cams):
        print(f"  {i+1})  {path}  {w}x{h}")
    while True:
        try:
            idx = int(input(f"Choose [1-{len(cams)}]: ")) - 1
            if 0 <= idx < len(cams):
                return cams[idx][0]
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def open_camera(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] Opened {path}  {w}x{h}")
    return cap, w, h

# ─── Board detection ──────────────────────────────────────────────────────────

def detect(gray):
    flags  = (cv2.CALIB_CB_ADAPTIVE_THRESH |
              cv2.CALIB_CB_NORMALIZE_IMAGE  |
              cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), flags)
    if found:
        crit    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
    return found, corners

# ─── Calibrate & save ─────────────────────────────────────────────────────────

def calibrate_and_save(obj_pts, img_pts, frame_size):
    print(f"\n[calib] Running calibration on {len(obj_pts)} captures ...")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, frame_size, None, None)

    data = {
        "frame_size":         list(frame_size),
        "camera_matrix":      K.tolist(),
        "dist_coeffs":        dist.tolist(),
        "reprojection_error": float(rms),
        "checkerboard": {
            "cols":        COLS,
            "rows":        ROWS,
            "square_size": SQUARE_M,
        },
        "notes": (
            "camera_matrix: [[fx,0,cx],[0,fy,cy],[0,0,1]] | "
            "dist_coeffs: [k1,k2,p1,p2,k3]"
        ),
    }
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    quality = ("great"      if rms < 0.5
               else "good"  if rms < 1.0
               else "acceptable (try more varied captures next time)")
    print(f"[saved] {OUTPUT}")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  RMS reprojection error: {rms:.4f} px  -> {quality}")
    return rms

# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=None,
                        help="Device path, e.g. /dev/video2")
    args = parser.parse_args()

    path       = choose_camera(args.camera)
    cap, w, h  = open_camera(path)
    frame_size = (w, h)

    print("\n" + "="*52)
    print(f"  Board  : {COLS}x{ROWS} inner corners, "
          f"{SQUARE_M*100:.0f} cm squares")
    print(f"  Target : {MIN_CAPTURES}+ captures")
    print("  c  -- capture    s  -- save    q  -- quit")
    print("="*52 + "\n")

    obj_pts    = []
    img_pts    = []
    n          = 0
    force_next = False   # allow saving with < MIN_CAPTURES on second 's' press

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect(gray)
        display        = frame.copy()

        if found:
            cv2.drawChessboardCorners(display, (COLS, ROWS), corners, True)
            label = f"Board visible -- press c to capture  [{n}/{MIN_CAPTURES}]"
            color = (0, 220, 0)
        else:
            label = f"No board detected  [{n}/{MIN_CAPTURES}]"
            color = (0, 80, 255)

        # text with dark outline for readability
        cv2.putText(display, label, (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(display, label, (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # progress bar at bottom
        fill = int((w - 8) * min(n / MIN_CAPTURES, 1.0))
        cv2.rectangle(display, (4, h-14), (4+fill, h-4), (0, 200, 80), -1)
        cv2.rectangle(display, (4, h-14), (w-4,    h-4), (180,180,180), 1)

        cv2.imshow("Camera Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if found:
                obj_pts.append(_objp.copy())
                img_pts.append(corners)
                n += 1
                print(f"  [capture {n:02d}] saved")
                if n == MIN_CAPTURES:
                    print(f"  Reached {MIN_CAPTURES} captures -- "
                          "press 's' to save or keep going.")
            else:
                print("  Board not visible -- reposition and try again.")

        elif key == ord("s"):
            if n < 4:
                print("  Need at least 4 captures.")
            elif n < MIN_CAPTURES and not force_next:
                print(f"  Only {n}/{MIN_CAPTURES} captures -- "
                      "press 's' again to save anyway.")
                force_next = True
            else:
                try:
                    calibrate_and_save(obj_pts, img_pts, frame_size)
                    print("\n  Done -- next step: python calibrate_workspace.py")
                    break
                except Exception as exc:
                    print(f"  Calibration failed: {exc}")
                    print("  Capture more varied poses and try again.")

        elif key == ord("q"):
            print("  Quit -- nothing saved.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
