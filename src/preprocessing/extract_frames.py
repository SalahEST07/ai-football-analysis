import cv2
import os

video_path = "data/raw/videos/match.mp4"
output_folder = "data/processed/frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 30 == 0:  # save every 30 frames
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)

    count += 1

cap.release()
print("Frames extracted!")
