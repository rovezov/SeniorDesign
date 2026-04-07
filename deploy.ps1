# deploy.ps1 — Production deployment script for Rubik Pi 3
# Runs the detection pipeline with optimized production settings
#
# Usage: .\deploy.ps1
# Configuration: 30 FPS, Rubik Pi 3 camera, helmet + vest detection

Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting Production Deployment - Rubik Pi 3" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nConfiguration:" -ForegroundColor Cyan
Write-Host "  Mode: Pipeline (headless)" -ForegroundColor White
Write-Host "  Camera Backend: Rubik Pi 3" -ForegroundColor White
Write-Host "  FPS: 30" -ForegroundColor White
Write-Host "  PPE Requirements: helmet, vest" -ForegroundColor White
Write-Host "`nLogs will be saved to: src/Output/session_log.txt" -ForegroundColor Yellow

Write-Host "`nStarting pipeline..." -ForegroundColor Cyan

python ./src/main.py --mode display --fps 60 --camera-backend webcam --ppe helmet vest --verbose

Write-Host "`nPipeline stopped." -ForegroundColor Green
