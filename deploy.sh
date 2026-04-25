#!/usr/bin/env bash
# deploy.sh — Production deployment script for Rubik Pi 3
# Runs the detection pipeline with H.265 UDP streaming to Windows/Client
#
# Usage: bash deploy.sh [WINDOWS_IP] [PORT]
# Examples:
#   bash deploy.sh                           # Local pipeline only (no streaming)
#   bash deploy.sh 192.168.1.178             # Stream to Windows at 192.168.1.178:9001
#   bash deploy.sh 192.168.1.178 5555        # Stream to Windows at 192.168.1.178:5555
#
# Note: Replace 192.168.1.178 with your Windows machine's actual IP address
# Find Windows IP: Open PowerShell and run 'ipconfig' (look for IPv4 Address)

WINDOWS_IP="${1:-}"
STREAM_PORT="${2:-9001}"

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

if [ -z "$WINDOWS_IP" ]; then
    echo "  Streaming: DISABLED (local processing only)"
    PIPELINE_ARGS=""
else
    echo "  Streaming: ENABLED"
    echo "  Stream Host: $WINDOWS_IP"
    echo "  Stream Port: $STREAM_PORT"
    PIPELINE_ARGS="--stream-host $WINDOWS_IP --stream-port $STREAM_PORT"
fi

echo ""
echo "Logs will be saved to: src/Output/session_log.txt"
echo ""

if [ -n "$WINDOWS_IP" ]; then
    echo "On Windows:"
    echo "  1. Open VLC Media Player"
    echo "  2. Media → Open Network Stream"
    echo "  3. Enter: udp://@:$STREAM_PORT"
    echo "  4. Click Play"
    echo ""
fi

echo "Starting pipeline..."
python3 ./src/main.py --mode pipeline --fps 30 --camera-backend pi --ppe helmet vest --face-workers 1 $PIPELINE_ARGS

echo ""
echo "Pipeline stopped."
