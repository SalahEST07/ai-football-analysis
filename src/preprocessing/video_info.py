import cv2
import os

video_path = "data/raw/videos/match.mp4"

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: File not found at {video_path}")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)

print(f"Attempting to open: {video_path}")
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"Raw values - FPS: {fps}, Width: {width}, Height: {height}, Frames: {frames}")

# Check for zero FPS
if fps == 0 or fps is None:
    print("Error: Could not retrieve FPS from video file")
    print("Possible reasons:")
    print("1. The video file may be corrupted")
    print("2. The video codec might not be supported by OpenCV")
    print("3. The file might not be a valid video file")
    
    # Try to read the first frame to test
    ret, frame = cap.read()
    if ret:
        print("Successfully read first frame, but FPS is still 0")
        print(f"Frame shape: {frame.shape}")
    else:
        print("Could not read first frame")
    
    exit(1)

# Calculate duration only if FPS is valid
duration = frames / fps

print("\nVideo Information:")
print(f"FPS: {fps}")
print(f"Resolution: {int(width)} x {int(height)}")
print(f"Total Frames: {int(frames)}")
print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

cap.release()