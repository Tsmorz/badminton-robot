"""Tennis Ball Tracker."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import torch
from loguru import logger
from numpy.typing import NDArray

from badminton_robot.base_classes import BoundingBox, DetectionResult, VideoMeta
from badminton_robot.definitions import DEFAULT_YOLO_MODEL_PATH
from badminton_robot.detector import ObjectDetector, YOLODetector


class FrameRenderer(ABC):
    """Renders annotations onto a frame."""

    @abstractmethod
    def render(
        self,
        frame: NDArray,
        result: DetectionResult,
        trail: deque,
        frame_idx: int,
        total: int,
        ball_found: int,
    ) -> NDArray: ...


class VideoSource(ABC):
    """Yields frames from a video source."""

    @abstractmethod
    def meta(self) -> VideoMeta: ...

    @abstractmethod
    def frames(self) -> Iterator[NDArray]: ...

    @abstractmethod
    def close(self) -> None: ...


class VideoSink(ABC):
    """Writes annotated frames to an output sink."""

    @abstractmethod
    def write(self, frame: NDArray) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class TrailTracker(ABC):
    """Maintains a positional trail for a tracked object."""

    @abstractmethod
    def update(self, box: BoundingBox | None) -> deque: ...


class CvVideoSource(VideoSource):
    """Reads frames from a file using OpenCV."""

    def __init__(self, path: Path) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

    def meta(self) -> VideoMeta:
        c = self._cap
        return VideoMeta(
            width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=c.get(cv2.CAP_PROP_FPS) or 30.0,
            total_frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[NDArray]:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def close(self) -> None:
        self._cap.release()


class CvVideoSink(VideoSink):
    """Writes frames to a file using OpenCV."""

    def __init__(self, path: Path, meta: VideoMeta) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            path, fourcc, meta.fps, (meta.width, meta.height)
        )

    def write(self, frame: NDArray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class BallTrailTracker(TrailTracker):
    """Maintains a fixed-length deque of recent ball centre positions."""

    def __init__(self, max_length: int) -> None:
        self._trail: deque[tuple[int, int] | None] = deque(maxlen=max_length)

    def update(self, box: BoundingBox | None) -> deque:
        self._trail.append(box.centre if box else None)
        return self._trail


class TennisFrameRenderer(FrameRenderer):
    """Renders player boxes, ball box, motion trail, and HUD onto a frame."""

    _PLAYER_COLOUR = (255, 100, 0)
    _BALL_COLOUR = (0, 255, 0)
    _DOT_COLOUR = (0, 0, 255)
    _HUD_COLOUR = (220, 220, 220)
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: NDArray,
        result: DetectionResult,
        trail: deque,
        frame_idx: int,
        total: int,
        ball_found: int,
    ) -> NDArray:
        out = frame.copy()
        h, w = out.shape[:2]
        self._draw_players(out, result.players)
        self._draw_ball(out, result.ball, w, h)
        self._draw_trail(out, trail)
        self._draw_hud(out, frame_idx, total, ball_found)
        return out

    def _draw_players(self, frame: NDArray, players: list[BoundingBox]) -> None:
        for box in players:
            cv2.rectangle(
                frame, (box.x1, box.y1), (box.x2, box.y2), self._PLAYER_COLOUR, 2
            )
            cv2.putText(
                frame,
                f"{box.label} {box.confidence:.2f}",
                (box.x1, max(0, box.y1 - 8)),
                self._FONT,
                0.55,
                self._PLAYER_COLOUR,
                2,
            )

    def _draw_ball(
        self, frame: NDArray, ball: BoundingBox | None, w: int, h: int
    ) -> None:
        if ball is None:
            return
        padded = ball.padded(pad=8, frame_w=w, frame_h=h)
        cv2.rectangle(
            frame, (padded.x1, padded.y1), (padded.x2, padded.y2), self._BALL_COLOUR, 2
        )
        cv2.putText(
            frame,
            f"{ball.label} {ball.confidence:.2f}",
            (padded.x1, max(0, padded.y1 - 8)),
            self._FONT,
            0.55,
            self._BALL_COLOUR,
            2,
        )
        cv2.circle(frame, ball.centre, 5, self._DOT_COLOUR, -1)

    @staticmethod
    def _draw_trail(
        frame: NDArray,
        trail: deque,
        color_start: tuple = (0, 255, 255),
        color_end: tuple = (0, 100, 255),
    ) -> None:
        pts = [p for p in trail if p is not None]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = tuple(
                int(s + (e - s) * alpha) for s, e in zip(color_start, color_end)
            )
            cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(4 * alpha)))

    def _draw_hud(
        self, frame: NDArray, frame_idx: int, total: int, ball_found: int
    ) -> None:
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total}",
            (10, 30),
            self._FONT,
            0.65,
            self._HUD_COLOUR,
            2,
        )
        cv2.putText(
            frame,
            f"Ball detections: {ball_found}",
            (10, 58),
            self._FONT,
            0.65,
            self._BALL_COLOUR,
            2,
        )


class TrackingPipeline:
    """Orchestrates reading, detecting, rendering, and writing.
    Depends entirely on abstract interfaces — concrete implementations are
    injected at construction time (Dependency Inversion).
    """

    def __init__(
        self,
        source: VideoSource,
        sink: VideoSink,
        detector: ObjectDetector,
        renderer: FrameRenderer,
        trail_tracker: TrailTracker,
        show: bool = False,
    ) -> None:
        self._source = source
        self._sink = sink
        self._detector = detector
        self._renderer = renderer
        self._trail_tracker = trail_tracker
        self._show = show
        self._log_interval: int = 100

    def run(self) -> None:
        meta = self._source.meta()
        frame_idx = 0
        ball_found = 0
        trail: deque = deque()

        logger.info(f"Processing {meta.total_frames} frames at {meta.fps:.1f} fps …")

        try:
            for frame in self._source.frames():
                frame_idx += 1

                result = self._detector.detect(frame)
                trail = self._trail_tracker.update(result.ball)
                ball_found += result.ball is not None

                out = self._renderer.render(
                    frame, result, trail, frame_idx, meta.total_frames, ball_found
                )
                self._sink.write(out)

                if self._show:
                    cv2.imshow("Tennis Ball Tracker", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Interrupted by user.")
                        break

                if frame_idx % self._log_interval == 0:
                    logger.info(f"  {frame_idx}/{meta.total_frames} frames processed …")

        finally:
            self._source.close()
            self._sink.close()
            if self._show:
                cv2.destroyAllWindows()

        rate = ball_found / max(frame_idx, 1) * 100
        logger.info(
            f"Done! Ball detected in {ball_found}/{frame_idx} frames ({rate:.1f}%)"
        )


def main() -> None:
    """Run the application."""
    parser = argparse.ArgumentParser(description="Tennis ball tracker using YOLOv8.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL_PATH, help="YOLO weights")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--trail", type=int, default=40, help="Trail length in frames")
    parser.add_argument(
        "--ball-conf", type=float, default=0.10, help="Ball confidence threshold"
    )
    parser.add_argument(
        "--player-conf", type=float, default=0.50, help="Player confidence threshold"
    )
    parser.add_argument(
        "--device", default=None, help="cpu or cuda (auto-detected if omitted)"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = CvVideoSource(path=args.input)
    sink = CvVideoSink(path=args.output, meta=source.meta())
    detector = YOLODetector(args.model, device, args.ball_conf, args.player_conf)
    renderer = TennisFrameRenderer()
    trail = BallTrailTracker(args.trail)

    pipeline = TrackingPipeline(
        source=source,
        sink=sink,
        detector=detector,
        renderer=renderer,
        trail_tracker=trail,
        show=args.show,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
