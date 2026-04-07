# How to Run main.py

## Overview

`main.py` is the master control script that orchestrates the detection pipeline with optional real-time display. It supports two execution modes and multiple camera backends for flexibility across different hardware configurations.

---

## Quick Start

### Default (Development Mode)
```bash
python main.py
```
Runs in **display mode** with the **default webcam** — perfect for development and testing on your local machine.

### Headless Mode (Rubik Pi 3)
```bash
python main.py --mode pipeline --camera-backend pi --verbose
```
Runs the detection pipeline with **no display output** — ideal for embedded deployment on Rubik Pi 3 where no monitor is connected.

---

## Execution Modes

### 1. Display Mode (Default)
```bash
python main.py --mode display
```

**Purpose:** Real-time video display with detection overlays
- Shows live camera feed with bounding boxes
- Displays person names, IDs, and PPE compliance status
- Shows FPS counter and detection statistics
- Best for development, debugging, and monitoring

**Controls:**
- Press `q` or `ESC` to quit

**Use Cases:**
- Testing detection accuracy
- Debugging PPE violations
- Monitoring live detections
- Development and prototyping

---

### 2. Pipeline Mode
```bash
python main.py --mode pipeline
```

**Purpose:** Pure detection processing without display
- Runs detections in background
- Saves cropped person images to output directory
- No graphical output (headless operation)
- Better performance (no rendering overhead)
- Best for production/embedded systems

**Use Cases:**
- Rubik Pi 3 3 deployment (no monitor)
- Server-based batch processing
- 24/7 recording and analysis
- Resources-constrained environments

---

## Camera Backends

The camera backend determines how video is captured from hardware:

### Webcam Backend (Default)
```bash
python main.py --camera-backend webcam
```

- Uses standard OpenCV VideoCapture
- Works on **any OS** (Windows, macOS, Linux)
- Supports USB cameras, built-in webcams, and standard video devices
- Device ID typically 0 for primary camera
- **Default choice for development**

### Rubik Pi 3 Backend
```bash
python main.py --camera-backend pi
```

- Uses optimized GStreamer pipeline (qtiqmmfsrc)
- **Only works on Rubik Pi 3 3 or later**
- Provides hardware-optimized video capture
- Better performance with Rubik Pi 3 native camera module
- Configures NV12 format and 1280x720 resolution
- **Required for Pi deployment**

---

## Complete Command Reference

### Syntax
```bash
python main.py [OPTIONS]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--mode`, `-m` | `pipeline`, `display` | `display` | Execution mode (with or without display) |
| `--camera-backend`, `-b` | `webcam`, `pi` | `webcam` | Camera hardware backend |
| `--camera`, `-c` | 0, 1, 2, ... | `0` | Camera device ID |
| `--fps`, `-f` | 1-120 | `60` | Frames per second |
| `--duration`, `-d` | seconds (float) | `None` | Run duration; `None` = indefinite |
| `--edge-margin`, `-e` | pixels | `0` | Filter detections within N pixels of frame edge |
| `--verbose`, `-v` | flag | `off` | Enable detailed logging output |
| `--ppe` | item names | `vest` | Required PPE items (space-separated list) |

---

## Usage Examples

### Development Scenarios

#### 1. Basic Display (Local Webcam)
```bash
python main.py
```
- Display mode (real-time video)
- Default webcam (device 0)
- 60 FPS
- Vest required for PPE compliance

#### 2. Display with Verbose Output
```bash
python main.py --verbose
```
- Display mode with detailed logging
- Useful for debugging and understanding pipeline behavior
- Shows frame processing times, tracking updates, person departures

#### 3. Display with Custom FPS
```bash
python main.py --fps 30
```
- Lower FPS for slower hardware or network cameras
- Reduces CPU usage

#### 4. Multiple Camera Sources
```bash
python main.py --camera 0  # Primary camera
python main.py --camera 1  # Secondary camera (if available)
```

#### 5. Set PPE Requirements
```bash
python main.py --ppe helmet vest
python main.py --ppe helmet
```
- `helmet` and `vest` both required
- Person marked non-compliant if either is missing

#### 6. Timed Recording
```bash
python main.py --duration 300 --verbose
```
- Run for 300 seconds (5 minutes)
- Verbose output for monitoring

---

### Production/Deployment Scenarios

#### 7. Rubik Pi 3 Headless Mode
```bash
python main.py --mode pipeline --camera-backend pi
```
- No display output
- Optimized for Pi hardware
- Saves captured person images to `src/Output/`

#### 8. Rubik Pi 3 with Verbose Logging
```bash
python main.py --mode pipeline --camera-backend pi --verbose
```
- Pipeline mode with detailed console logging
- Useful for monitoring Pi deployment
- Log departures, tracking updates, save locations

#### 9. Rubik Pi 3 with Custom FPS
```bash
python main.py --mode pipeline --camera-backend pi --fps 15 --verbose
```
- Lower FPS for reduced Pi CPU load
- 15 FPS is typical for Rubik Pi 3 3
- Verbose logging shows processing status

#### 10. Rubik Pi 3 with Custom PPE
```bash
python main.py --mode pipeline --camera-backend pi --ppe helmet vest
```
- Requires both helmet and vest
- Pipeline runs continuously
- Marks non-compliance when items missing

---

## Output Behavior

### Display Mode Output
- **Console:** FPS counter, frame count, detection statistics
- **Window:** Real-time video with overlays:
  - Green bounding boxes = PPE compliant
  - Red bounding boxes = PPE violation
  - Person name and ID labels
  - FPS and detection counts
- **Files:** Cropped person images saved to `src/Output/`

### Pipeline Mode Output
- **Console:** Frame processing info (with `--verbose`)
- **Window:** None (headless)
- **Files:** Cropped person images saved to `src/Output/`
  - Format: `{name}_{person_id}_compliant.jpg` (PPE OK)
  - Format: `{name}_{person_id}_missing_{items}.jpg` (PPE violation)

---

## Output Directory

All detected and identified persons are automatically saved as cropped images in:
```
src/Output/
```

Filenames follow the pattern:
- **Compliant:** `John_Doe_5_compliant.jpg`
- **Non-compliant:** `John_Doe_5_missing_helmet_vest.jpg`
- **Unknown:** `unknown_person_8_compliant.jpg`

The script automatically clears old images when it starts.

---

## Troubleshooting

### "No module named 'camera'" or "No module named 'detection_pipeline'"
**Solution:** Run from the `src/` directory:
```bash
cd src
python main.py
```

### "Camera not found" or "Cannot open camera"
**Solution:** Check available cameras:
- Try different device IDs: `--camera 0`, `--camera 1`, etc.
- On Windows, use Device Manager to find camera
- On Linux, use `ls /dev/video*` to list cameras
- Ensure USB camera is connected before running

### "GStreamer not available" (Rubik Pi 3 mode)
**Solution:** This error means:
1. You're not on a Rubik Pi 3, OR
2. GStreamer is not installed on your Pi
3. Use `--camera-backend webcam` instead, or
4. Install GStreamer on Pi: `sudo apt-get install gstreamer1.0-tools`

### "ImportError: cannot import name 'PPEDetector'" 
**Solution:** Ensure you're in the correct directory:
```bash
cd src
python main.py
```

### Slow Performance
**Solution:** Try these optimizations:
```bash
# Lower FPS
python main.py --fps 15

# Pipeline mode (no display overhead)
python main.py --mode pipeline

# For Rubik Pi 3 specifically
python main.py --mode pipeline --camera-backend pi --fps 15
```

### Person images not saving
**Solution:** Check:
1. `src/Output/` directory is writable
2. Script has permission to create files
3. No disk space issues
4. Script ran for at least 0.5 seconds per person (min_tracking_time)

---

## Performance Tuning

### For Local Development (Webcam)
```bash
python main.py --fps 30 --verbose
```
- 30 FPS balances quality and CPU usage
- Good for debugging on laptop

### For Rubik Pi 3
```bash
python main.py --mode pipeline --camera-backend pi --fps 15 --verbose
```
- 15 FPS is proven stable on Pi 3
- Keep console logging with `--verbose` for monitoring
- No display saves ~40% CPU overhead

### For High-Performance Server
```bash
python main.py --mode pipeline --camera-backend webcam --fps 60
```
- Higher FPS for better accuracy
- Pipeline mode for efficiency
- No verbose logging in production to save I/O

---

## Next Steps

1. **Test locally first:**
   ```bash
   python main.py --verbose
   ```

2. **Deploy to Rubik Pi 3:**
   ```bash
   python main.py --mode pipeline --camera-backend pi --verbose
   ```

3. **Monitor with logs:**
   ```bash
   python main.py --mode pipeline --camera-backend pi --verbose > pi_logs.txt 2>&1 &
   ```

4. **Check output:**
   ```bash
   ls -la src/Output/
   ```

---

## Support Matrix

| Scenario | Command | Works? |
|----------|---------|--------|
| Windows development | `main.py --mode display` | ✅ Yes |
| macOS development | `main.py --mode display` | ✅ Yes |
| Linux development | `main.py --mode display` | ✅ Yes |
| Rubik Pi 3 display | `main.py --camera-backend pi` | ⚠️ Only if monitor connected |
| Rubik Pi 3 headless | `main.py --mode pipeline --camera-backend pi` | ✅ Yes |
| USB camera on Pi | `main.py --camera-backend webcam` | ✅ Yes |
| Network camera | (camera property dependent) | ⚠️ May work with webcam backend |

---

## See Also

- **Architecture:** See [architecture.md](architecture.md)
- **API Reference:** See [api_spec.md](api_spec.md)
- **Setup Guide:** See [setup_guide.md](setup_guide.md)
