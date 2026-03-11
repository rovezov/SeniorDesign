# PPE Test Images

Place test images here before running `test_ppe_detector.py`.

Supported formats: `.jpg`, `.jpeg`, `.png`

Suggested image types for thorough testing:
- Person wearing both helmet and vest (expect: all_present = True)
- Person wearing only a helmet (expect: missing vest)
- Person wearing only a vest (expect: missing helmet)
- Person wearing no PPE (expect: both missing)
