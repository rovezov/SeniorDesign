# Verbose / logging flags for each module.
# Set a module's flag to True to enable its debug output.

VERBOSE                 = True  # Master switch to enable all verbose output. Overrides individual flags below.
VERBOSE_CAMERA          = False
VERBOSE_FACE_DETECTION  = False
VERBOSE_FACE_MATCHING   = False
VERBOSE_HUMAN_DETECTION = False
VERBOSE_PPE_DETECTION   = True
VERBOSE_STORAGE         = False


_MODULE_FLAGS: dict[str, bool] = {
    "camera":          VERBOSE_CAMERA,
    "face_detection":  VERBOSE_FACE_DETECTION,
    "face_matching":   VERBOSE_FACE_MATCHING,
    "human_detection": VERBOSE_HUMAN_DETECTION,
    "ppe_detection":   VERBOSE_PPE_DETECTION,
    "storage":         VERBOSE_STORAGE,
}


def log(module: str, *args, **kwargs) -> None:
    """Print a log message if verbose is enabled for *module*.

    Usage:
        from utils.verbose import log
        log("ppe_detection", "Detected labels:", labels)

    Args:
        module: One of the module name keys defined in _MODULE_FLAGS.
        *args:  Passed directly to print().
        **kwargs: Passed directly to print().
    """
    if _MODULE_FLAGS.get(module, False) and VERBOSE:
        print(f"[{module}]", *args, **kwargs)
