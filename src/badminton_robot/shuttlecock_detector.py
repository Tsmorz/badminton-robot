"""file detects the shuttlecock."""

from ultralytics import YOLO


class ShuttlecockDetector:
    """detector to find shuttlecock in an image."""

    def __init__(self, model_type="yolov8n.pt"):
        # This loads the AI brain
        self.model = YOLO(model_type)

    def get_location(self, image_path):
        """YOLO used on an image and returns a list of (x, y) coordinates."""
        results = self.model(image_path)

        centers = []
        for result in results:
            for box in result.boxes:
                # xywh[0] gives: center_x, center_y, width, height
                coords = box.xywh[0].tolist()
                x_center = int(coords[0])
                y_center = int(coords[1])
                centers.append((x_center, y_center))

        return centers
