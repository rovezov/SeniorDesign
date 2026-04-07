#!/usr/bin/env bash
# install.sh — Linux/RubikPi setup script
# Handles the opencv-python-headless conflict introduced by insightface/albumentations.
#
# Usage: bash install.sh

set -e

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo ""
echo "Force-reinstalling full OpenCV (removes headless version if present)..."
python -m pip uninstall opencv-python-headless -y 2>/dev/null || true
python -m pip install --force-reinstall opencv-contrib-python==4.13.0.92

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
