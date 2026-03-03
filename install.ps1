# install.ps1 — Windows setup script
# Handles the opencv-python-headless conflict introduced by insightface/albumentations.
#
# Usage: .\install.ps1

Set-StrictMode -Version Latest

Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`nForce-reinstalling full OpenCV (removes headless version)..." -ForegroundColor Cyan
# Suppress errors — pip writes warnings to stderr which PowerShell treats as errors
$ErrorActionPreference = "SilentlyContinue"
pip uninstall opencv-python-headless -y 2>&1 | Out-Null
$ErrorActionPreference = "Continue"
pip install --force-reinstall opencv-contrib-python==4.13.0.92

Write-Host "`nDone. Run: python src/integrated_demo.py" -ForegroundColor Green
