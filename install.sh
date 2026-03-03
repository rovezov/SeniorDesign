#!/usr/bin/env bash
# install.sh — Linux/RubikPi setup script
# Handles the opencv-python-headless conflict introduced by insightface/albumentations.
#
# Usage: bash install.sh

set -e

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Force-reinstalling full OpenCV (removes headless version if present)..."
pip uninstall opencv-python-headless -y 2>/dev/null || true
pip install --force-reinstall opencv-contrib-python==4.13.0.92

echo ""
echo "Done. Run: python src/integrated_demo.py"
