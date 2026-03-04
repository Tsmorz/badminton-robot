"""Tennis Ball Tracker."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def padded(self, pad: int, frame_w: int, frame_h: int) -> BoundingBox:
        return BoundingBox(
            max(0, self.x1 - pad),
            max(0, self.y1 - pad),
            min(frame_w, self.x2 + pad),
            min(frame_h, self.y2 + pad),
            self.confidence,
            self.label,
        )


@dataclass
class DetectionResult:
    """All detections for a single frame."""

    players: list[BoundingBox] = field(default_factory=list)
    ball: BoundingBox | None = None


@dataclass
class VideoMeta:
    """Metadata extracted from an opened video capture."""

    width: int
    height: int
    fps: float
    total_frames: int
