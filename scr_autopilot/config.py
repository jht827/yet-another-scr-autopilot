from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import tomllib


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


@dataclass(frozen=True)
class WindowConfig:
    title: str
    reference_size: tuple[int, int]
    refresh_seconds: float


@dataclass(frozen=True)
class OcrTesseractConfig:
    threshold: Optional[int]
    invert: bool
    scale: float
    whitelist: str
    lang: str
    psm: int
    oem: int


@dataclass(frozen=True)
class OcrOpenCvConfig:
    threshold: Optional[int]
    invert: bool
    min_area: int
    min_height: int
    speed_template_dir: Path


@dataclass(frozen=True)
class OcrConfig:
    speed_backend: str
    tesseract: OcrTesseractConfig
    opencv: OcrOpenCvConfig


@dataclass(frozen=True)
class ReadoutConfig:
    interval: float
    debug: bool
    debug_dir: Path
    debug_interval: float
    opencv_debug: bool


@dataclass(frozen=True)
class AccelCurveConfig:
    interval: float
    smooth_window: int
    output_csv: Path


@dataclass(frozen=True)
class AppConfig:
    window: WindowConfig
    rois: HudRois
    ocr: OcrConfig
    readout: ReadoutConfig
    accel_curve: AccelCurveConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "scr_autopilot.toml"


def load_config(path: Optional[Path] = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.exists():
        data = tomllib.loads(config_path.read_text())

    def section(*keys: str) -> Mapping[str, Any]:
        current: Mapping[str, Any] = data
        for key in keys:
            value = current.get(key, {})
            if not isinstance(value, Mapping):
                return {}
            current = value
        return current

    def maybe_threshold(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        return None if value < 0 else value

    default_rois = HudRois(
        speed=PixelRoi(x=1002, y=623, width=93, height=49),
        limit=PixelRoi(x=1016, y=706, width=62, height=37),
        distance=PixelRoi(x=-13, y=712, width=54, height=21),
    )
    rois_section = section("rois")
    rois = HudRois(
        speed=_parse_roi(rois_section.get("speed"), default_rois.speed),
        limit=_parse_roi(rois_section.get("limit"), default_rois.limit),
        distance=_parse_roi(rois_section.get("distance"), default_rois.distance),
    )

    window_section = section("window")
    window = WindowConfig(
        title=str(window_section.get("title", "Roblox")),
        reference_size=(
            int(window_section.get("reference_width", 1920)),
            int(window_section.get("reference_height", 1080)),
        ),
        refresh_seconds=float(window_section.get("refresh_seconds", 1.0)),
    )

    ocr_section = section("ocr")
    tesseract_section = section("ocr", "tesseract")
    opencv_section = section("ocr", "opencv")
    ocr = OcrConfig(
        speed_backend=str(ocr_section.get("speed_backend", "tesseract")),
        tesseract=OcrTesseractConfig(
            threshold=maybe_threshold(_maybe_int(tesseract_section.get("threshold"), 160)),
            invert=bool(tesseract_section.get("invert", False)),
            scale=float(tesseract_section.get("scale", 2.0)),
            whitelist=str(
                tesseract_section.get("whitelist", "0123456789abcdefghijklmnopqrstuvwxyz")
            ),
            lang=str(tesseract_section.get("lang", "eng")),
            psm=int(tesseract_section.get("psm", 7)),
            oem=int(tesseract_section.get("oem", 1)),
        ),
        opencv=OcrOpenCvConfig(
            threshold=maybe_threshold(_maybe_int(opencv_section.get("threshold"), 160)),
            invert=bool(opencv_section.get("invert", False)),
            min_area=int(opencv_section.get("min_area", 30)),
            min_height=int(opencv_section.get("min_height", 10)),
            speed_template_dir=Path(opencv_section.get("speed_template_dir", "templates/speed_digits")),
        ),
    )

    readout_section = section("readout")
    readout = ReadoutConfig(
        interval=float(readout_section.get("interval", 0.05)),
        debug=bool(readout_section.get("debug", False)),
        debug_dir=Path(readout_section.get("debug_dir", "hud_debug")),
        debug_interval=float(readout_section.get("debug_interval", 1.0)),
        opencv_debug=bool(readout_section.get("opencv_debug", False)),
    )

    accel_section = section("accel_curve")
    accel_curve = AccelCurveConfig(
        interval=float(accel_section.get("interval", 0.1)),
        smooth_window=int(accel_section.get("smooth_window", 5)),
        output_csv=Path(accel_section.get("output_csv", "accel_curve.csv")),
    )

    return AppConfig(
        window=window,
        rois=rois,
        ocr=ocr,
        readout=readout,
        accel_curve=accel_curve,
    )


def _parse_roi(value: Any, default: PixelRoi) -> PixelRoi:
    if not isinstance(value, Mapping):
        return default
    try:
        return PixelRoi(
            x=int(value.get("x", default.x)),
            y=int(value.get("y", default.y)),
            width=int(value.get("width", default.width)),
            height=int(value.get("height", default.height)),
        )
    except (TypeError, ValueError):
        return default


def _maybe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
