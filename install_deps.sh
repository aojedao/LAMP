#!/usr/bin/env bash
# install_deps.sh — Add YOLO11n vision dependencies to the existing 'lerobot' conda env.
#
# What this script does:
#   1. Replaces opencv-python-headless (already in lerobot) with
#      opencv-contrib-python at the SAME version — contrib adds cv2.aruco
#      which main.py needs, while headless lacks it.
#   2. Installs ultralytics (YOLO) while keeping the already-installed
#      torch / torchvision untouched (pip will skip them if compatible).
#
# numpy is already present in lerobot — no change needed.

set -e

ENV_NAME="lerobot"

# ── Check conda is available ──────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Make sure conda is initialised in your shell."
    exit 1
fi

if ! conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "[ERROR] Conda env '${ENV_NAME}' not found."
    exit 1
fi

# ── Step 1: swap opencv flavour ───────────────────────────────────────────────
# Detect the currently installed opencv-python-headless version so we pin
# opencv-contrib-python to the exact same version (avoids any API mismatch).
echo "==> Detecting installed OpenCV version in '${ENV_NAME}'..."
OPENCV_VER=$(conda run -n "${ENV_NAME}" pip show opencv-python-headless 2>/dev/null \
             | grep -i "^Version:" | awk '{print $2}')

if [ -z "${OPENCV_VER}" ]; then
    echo "    opencv-python-headless not found — will install opencv-contrib-python directly."
    OPENCV_PIN="opencv-contrib-python"
else
    echo "    Found opencv-python-headless==${OPENCV_VER} — will replace with contrib at same version."
    conda run -n "${ENV_NAME}" pip uninstall -y opencv-python-headless
    OPENCV_PIN="opencv-contrib-python==${OPENCV_VER}"
fi

echo "==> Installing ${OPENCV_PIN} into '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" pip install "${OPENCV_PIN}"

# ── Step 2: install ultralytics (reuses existing torch/torchvision) ───────────
echo ""
echo "==> Installing ultralytics into '${ENV_NAME}'..."
# pip will see torch 2.10.0 and torchvision 0.25.0 already satisfy ultralytics'
# requirements and skip reinstalling them.
conda run -n "${ENV_NAME}" pip install ultralytics

echo ""
echo "==> Done! All vision dependencies are installed in '${ENV_NAME}'."
echo ""
echo "NOTE: The YOLO model file (yolo11n.pt) will be downloaded automatically"
echo "      on the first run of vision/main.py (~6 MB)."
echo ""
echo "Run with:"
echo "  conda activate ${ENV_NAME}"
echo "  python vision/main.py            # auto-detect robot camera"
echo "  python vision/main.py --webcam   # force laptop webcam (YOLO test only)"
