import os
from ultralytics import YOLO

model_path = r"D:\Notes\Degree\MITWPU HACKATHON\Smart Tracking Violation Detection System\yolo_models\helmet_yolov8_best.pt"

# ✅ Check if the file exists
if not os.path.exists(model_path):
    print(f"🚨 Model file NOT FOUND: {model_path}")
else:
    print(f"✅ Model file found at: {model_path}")

    # Try loading the model
    try:
        model = YOLO(model_path)
        print("✅ YOLO Model loaded successfully!")
    except Exception as e:
        print(f"🚨 Error loading YOLO Model: {e}")
