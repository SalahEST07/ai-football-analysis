from ultralytics import YOLO
from src.features.geometry import get_center

import cv2

model = YOLO("models/best.pt")

cap = cv2.VideoCapture("data/raw/videos/match.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Football Detection", annotated)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
