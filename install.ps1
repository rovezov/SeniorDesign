# install.ps1 — Windows setup script
# Handles the opencv-python-headless conflict introduced by insightface/albumentations.
#
# Usage: .\install.ps1

Set-StrictMode -Version Latest

Write-Host "Installing requirements..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "`nForce-reinstalling full OpenCV (removes headless version)..." -ForegroundColor Cyan
# Suppress errors — pip writes warnings to stderr which PowerShell treats as errors
$ErrorActionPreference = "SilentlyContinue"
python -m pip uninstall opencv-python-headless -y 2>&1 | Out-Null
$ErrorActionPreference = "Continue"
python -m pip install --force-reinstall opencv-contrib-python==4.13.0.92

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Installation complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nTO RUN:" -ForegroundColor Cyan
Write-Host "`n  On Rubik Pi 3 3:" -ForegroundColor Yellow
Write-Host "    cd src" -ForegroundColor White
Write-Host "    python main.py --mode pipeline --camera-backend pi --verbose" -ForegroundColor White
Write-Host "`n  On PC with webcam and display available:" -ForegroundColor Yellow
Write-Host "    cd src" -ForegroundColor White
Write-Host "    python main.py --mode display --verbose" -ForegroundColor White
Write-Host "`n  On PC with webcam but no display:" -ForegroundColor Yellow
Write-Host "    cd src" -ForegroundColor White
Write-Host "    python main.py --mode pipeline --verbose" -ForegroundColor White
Write-Host "`n" -ForegroundColor Cyan
Write-Host "PRODUCTION DEPLOYMENT:" -ForegroundColor Cyan
Write-Host "  To run optimized for Rubik Pi 3 production:" -ForegroundColor Yellow
Write-Host "    .\deploy.ps1" -ForegroundColor White
