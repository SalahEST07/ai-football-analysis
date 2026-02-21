from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

# Test image
results = model("data/raw/test.jpg", show=True)

print("Detection done!")
