#!/usr/bin/env bash
# install.sh — Linux/RubikPi setup script
# Avoids leaving conflicting OpenCV wheels installed after dependency setup.
#
# Usage: bash install.sh

set -e

echo "Installing requirements..."
python3 -m pip install --break-system-packages -r requirements.txt

echo ""
echo "Removing OpenCV wheel variants that can conflict with the system cv2 build..."
python3 -m pip uninstall --break-system-packages -y opencv-contrib-python opencv-contrib-python-headless opencv-python opencv-python-headless 2>/dev/null || true

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ Installation complete"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "TO RUN:"
echo ""
echo "  On Rubik Pi 3 3 (no display):" 
echo "    cd src"
echo "    python main.py --mode pipeline --camera-backend pi --verbose"
echo ""
echo "  On Rubik Pi 3 3 (with display):" 
echo "    cd src"
echo "    python main.py --mode display --camera-backend pi --verbose"
echo ""
echo "  On PC with webcam and display available:"
echo "    cd src"
echo "    python main.py --mode display --verbose"
echo ""
echo "  On PC with webcam but no display:"
echo "    cd src"
echo "    python main.py --mode pipeline --verbose"
echo ""
echo "PRODUCTION DEPLOYMENT:"
echo "  To run optimized for Rubik Pi 3 production:"
echo "    bash deploy.sh"
