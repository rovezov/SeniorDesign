# install.ps1 — Windows setup script
# Avoids leaving conflicting OpenCV wheels installed after dependency setup.
#
# Usage: .\install.ps1

Set-StrictMode -Version Latest

Write-Host "Installing requirements..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "`nRemoving OpenCV wheel variants that can conflict with the system cv2 build..." -ForegroundColor Cyan
# Suppress errors — pip writes warnings to stderr which PowerShell treats as errors
$ErrorActionPreference = "SilentlyContinue"
python -m pip uninstall opencv-contrib-python opencv-contrib-python-headless opencv-python opencv-python-headless -y 2>&1 | Out-Null
$ErrorActionPreference = "Continue"

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
