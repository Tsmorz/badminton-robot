# src/shuttlecock_detector.py
from pathlib import Path, PurePosixPath

import cv2
from loguru import logger
from ultralytics import YOLO


class ShuttlecockDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.25, iou=0.45, device=""):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        self.model = YOLO(self.model_path)
        return self.model

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        return self.image

    def detect(self, image):
        # Run detection
        results = self.model(image)

        # Draw results on image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # Show result
        cv2.imshow("Shuttlecock Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    image_path = PurePosixPath(
        "data", "data", "testimage2.jpg"
    )  # Update this to your test image path
    logger.info(image_path)
    detector = ShuttlecockDetector()
    image = detector.load_image(image_path)
    result = detector.detect(image)


if __name__ == "__main__":
    main()
