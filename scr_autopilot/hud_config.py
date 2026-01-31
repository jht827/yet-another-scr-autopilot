from dataclasses import dataclass


@dataclass(frozen=True)
class PixelRoi:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class HudRois:
    speed: PixelRoi
    limit: PixelRoi
    distance: PixelRoi


WINDOW_TITLE = "Roblox"
REFERENCE_SIZE = (1920, 1080)
HUD_ROIS = HudRois(
    speed=PixelRoi(x=1002, y=623, width=93, height=49),
    limit=PixelRoi(x=1016, y=706, width=62, height=37),
    distance=PixelRoi(x=-13, y=712, width=54, height=21),
)


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
