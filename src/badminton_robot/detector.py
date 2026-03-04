from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO

from badminton_robot.base_classes import BoundingBox, DetectionResult
from badminton_robot.definitions import PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID


class ObjectDetector(ABC):
    """Detects objects in a single BGR frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult: ...


class YOLODetector(ObjectDetector):
    """Detects players and sports balls using a YOLOv8 models."""

    def __init__(
        self, model_path: Path, device: str | None, ball_conf: float, player_conf: float
    ) -> None:
        self._model = YOLO(model_path)
        self._device = device
        self._ball_conf = ball_conf
        self._player_conf = player_conf
        self._min_conf = min(ball_conf, player_conf)
        logger.info(f"Loaded YOLO model: {model_path}  device={device or 'auto'}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        raw = self._model(
            frame,
            device=self._device,
            verbose=False,
            conf=self._min_conf,
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
        )[0]

        result = DetectionResult()
        for box in raw.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            if cls == PERSON_CLASS_ID and conf >= self._player_conf:
                result.players.append(BoundingBox(x1, y1, x2, y2, conf, "Player"))

            elif cls == SPORTS_BALL_CLASS_ID and conf >= self._ball_conf:
                # Keep only the highest-confidence ball per frame
                candidate = BoundingBox(x1, y1, x2, y2, conf, "Ball")
                if result.ball is None or conf > result.ball.confidence:
                    result.ball = candidate

        return result
