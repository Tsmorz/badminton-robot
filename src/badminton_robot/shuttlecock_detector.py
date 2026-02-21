# src/shuttlecock_detector.py
from ultralytics import YOLO
import cv2

class ShuttlecockDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.25, iou=0.45, device=''):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = None

    def load_model(self, model_path: str = None):
        path = model_path or self.model_path
        self.model = YOLO(path)
        return self.model

    def draw_boxes(self, image, boxes, label_map=None):
        if label_map is None:
            label_map = {0: 'shuttlecock'}
        for b in boxes:
            x1, y1, x2, y2 = map(int, b['xyxy'])
            conf = b.get('conf', 0.0)
            cls = b.get('cls', 0)
            text = f"{label_map.get(cls, cls)} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def webcam_stream(self, cam_index=0, show=True, output_path=None, max_frames=None, label_map=None):
        """
        Run streaming inference using Ultralytics' stream=True API backed by the webcam.
        - cam_index: 0 for default laptop webcam (or set device path / URL)
        - yields (boxes, vis_image, result) per frame and also displays/saves if requested.
        """
        if self.model is None:
            self.load_model()

        source = cam_index
        stream = self.model.predict(source=source, conf=self.conf, iou=self.iou, device=self.device, stream=True)

        writer = None
        frame_count = 0
        try:
            for res in stream:
                img = res.orig_img  # BGR numpy array
                boxes = []
                if hasattr(res, 'boxes') and res.boxes is not None:
                    xyxy_arr = res.boxes.xyxy.cpu().numpy()
                    conf_arr = res.boxes.conf.cpu().numpy()
                    cls_arr = res.boxes.cls.cpu().numpy()
                    for xyxy, cf, cl in zip(xyxy_arr, conf_arr, cls_arr):
                        boxes.append({'xyxy': [float(x) for x in xyxy], 'conf': float(cf), 'cls': int(cl)})

                vis = self.draw_boxes(img.copy(), boxes, label_map=label_map)

                # lazy writer init
                if output_path and writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = getattr(res, 'fps', 25) or 25
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if writer:
                    writer.write(vis)

                if show:
                    cv2.imshow('shuttlecock-detect', vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                frame_count += 1
                yield boxes, vis, res

                if max_frames and frame_count >= max_frames:
                    break
        finally:
            if writer:
                writer.release()
            cv2.destroyAllWindows()