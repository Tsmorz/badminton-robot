import cv2
from ultralytics import YOLO

# Load the standard model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# The 'Sports Hack': 32 is 'sports ball', 14 is 'bird' (sometimes shuttlecocks look like birds to AI!)
TARGET_CLASSES = [14, 32]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run detection only on our target classes
    results = model(frame, classes=TARGET_CLASSES, conf=0.3)

    for r in results:
        for box in r.boxes:
            # Get coordinates of the box: x1, y1 (top left) and x2, y2 (bottom right)
            x1, y1, x2, y2 = box.xyxy[0]

            # Calculate the Center Point (This is what the robot needs!)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Print coordinates to the terminal
            print(f"TARGET DETECTED -> X: {center_x} Y: {center_y}")

            # Draw a small circle at the center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Show the video
    cv2.imshow("Robot Vision - Target Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
