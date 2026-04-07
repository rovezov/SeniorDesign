#!/usr/bin/env bash
# deploy.sh — Production deployment script for Rubik Pi 3
# Runs the detection pipeline with optimized production settings
#
# Usage: bash deploy.sh
# Configuration: 30 FPS, Rubik Pi 3 camera, helmet + vest detection

echo ""
echo "========================================"
echo "Starting Production Deployment - Rubik Pi 3"
echo "========================================"
echo ""

echo "Configuration:"
echo "  Mode: Pipeline (headless)"
echo "  Camera Backend: Rubik Pi 3"
echo "  FPS: 30"
echo "  PPE Requirements: helmet, vest"
echo ""
echo "Logs will be saved to: src/Output/session_log.txt"
echo ""

echo "Starting pipeline..."
python ./src/main.py --mode pipeline --fps 60 --camera-backend pi --ppe helmet vest --verbose

echo ""
echo "Pipeline stopped."
