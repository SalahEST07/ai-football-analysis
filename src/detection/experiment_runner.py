import time
import mlflow
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

video_path = "data/raw/videos/match.mp4"


def run_experiment(conf=0.25, imgsz=640):
    cap = cv2.VideoCapture(video_path)

    frames = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        model(frame, conf=conf, imgsz=imgsz)
        frames += 1

        if frames >= 200:  # test first 200 frames
            break

    end = time.time()

    fps = frames / (end - start)

    cap.release()

    return fps


# =========================
# MLflow Tracking
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Football_Video_Analysis")

with mlflow.start_run(run_name="Hyperparameter_Test"):

    conf = 0.25
    imgsz = 640

    fps = run_experiment(conf, imgsz)

    mlflow.log_param("confidence", conf)
    mlflow.log_param("img_size", imgsz)

    mlflow.log_metric("fps", fps)

    print("FPS:", fps)
