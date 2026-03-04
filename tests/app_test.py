"""Test the main program."""

from pathlib import Path

from badminton_robot.app import (
    BallTrailTracker,
    CvVideoSink,
    CvVideoSource,
    TennisFrameRenderer,
    TrackingPipeline,
    YOLODetector,
)
from badminton_robot.definitions import DEFAULT_YOLO_MODEL_PATH


def test_main():
    """Test the main function."""
    device = "cpu"
    source = CvVideoSource(path=Path("data", "data", "tennis.mov"))
    sink = CvVideoSink(
        path=Path("data", "results", "tennis-result.mov"), meta=source.meta()
    )
    detector = YOLODetector(
        DEFAULT_YOLO_MODEL_PATH, device, ball_conf=0.1, player_conf=0.5
    )
    renderer = TennisFrameRenderer()
    trail = BallTrailTracker(40)

    pipeline = TrackingPipeline(
        source=source,
        sink=sink,
        detector=detector,
        renderer=renderer,
        trail_tracker=trail,
        show=False,
    )
    pipeline.run()
