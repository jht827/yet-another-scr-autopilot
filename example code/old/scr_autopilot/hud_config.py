from scr_autopilot.config import HudRois, PixelRoi, load_config


_CONFIG = load_config()
WINDOW_TITLE = _CONFIG.window.title
REFERENCE_SIZE = _CONFIG.window.reference_size
HUD_ROIS = _CONFIG.rois


def format_roi(roi: PixelRoi) -> str:
    return f"PixelRoi(x={roi.x}, y={roi.y}, width={roi.width}, height={roi.height})"


def format_rois(rois: HudRois) -> str:
    return (
        "HudRois(\\n"
        f"    speed={format_roi(rois.speed)},\\n"
        f"    limit={format_roi(rois.limit)},\\n"
        f"    distance={format_roi(rois.distance)},\\n"
        ")"
    )
