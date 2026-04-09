# Setup Guide

## Hardware Requirements
- Rubik Pi 3
- USB/CSI camera module

## Software Requirements
- Python 3.12.3
- Required Python packages (see requirements.txt)

## Setup Steps
1. Connect camera to Rubik Pi 3
2. Create and activate a Python 3.12.3 virtual environment:
   - Windows:
     - py -3.12 -m venv .venv
     - .\.venv\Scripts\Activate.ps1
   - Linux/Rubik Pi:
     - python3.12 -m venv .venv
     - source .venv/bin/activate
3. Install dependencies using the project scripts:
   - Windows: .\install.ps1
   - Linux/Rubik Pi: bash install.sh
4. Run modules in src/ for testing
