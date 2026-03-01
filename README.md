# LAMP 💡🤖

**A robotic desk lamp that sees objects and moves to point at them.**

Built at the [NYU ITP Stupid Hackathon](https://stupidshit.lol/) — February 28, 2026.

---

## What is this?

LAMP is a vision-guided robotic arm that acts like a curious desk lamp. Place a book or bottle on the workspace and LAMP swivels to look at it — like Pixar's Luxo Jr., but real and arguably stupider.

## How it works

1. **Camera** — A USB camera on an SO-100 robot arm looks down at a 20" × 20" workspace
2. **Workspace boundary** — Four ArUco markers define the edges of the workspace; objects outside are ignored
3. **Object detection** — YOLO11n runs on every frame and detects books and bottles
4. **Arm control** — Detected object coordinates are sent to the robot arm, which physically moves to point at the object

## Setup

### Requirements

- Python 3.11+
- SO-100 robot arm with USB camera
- 4 printed ArUco markers (4×4 dictionary, IDs 0–3)
- A checkerboard for camera calibration (4×5 inner corners, 3.5 cm squares)

### Install

```bash
python -m venv env
env\Scripts\activate        # Windows
pip install numpy opencv-python ultralytics
```

### Calibrate

**1. Camera intrinsics** — hold a checkerboard in front of the camera at various angles:

```bash
python calibrate_camera.py
```

**2. Workspace boundary** — place 4 ArUco markers at the corners of your workspace:

```bash
python calibrate_workspace.py
```

### Run

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `q` | Quit |

Objects inside the workspace get a **green** bounding box. Objects outside get a **red** box with `[OUTSIDE]`.

## Project structure

```
LAMP/
├── main.py                    # Detection loop + arm bridge
├── calibrate_camera.py        # Checkerboard camera calibration
├── calibrate_workspace.py     # ArUco workspace boundary calibration
├── camera_calibration.json    # Camera intrinsics (generated)
├── workspace_calibration.json # Workspace mapping (generated)
├── yolo11n.pt                 # YOLO model weights (not in git)
├── 1-4aruco.pdf               # Printable ArUco markers
└── .gitignore
```

## Team

Built with questionable judgment at NYU ITP Stupid Hackathon 2026.