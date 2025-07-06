# Real-Time Object Detection using Webcam in Jupyter
from ultralytics import YOLO
import os

print("🔄 Loading object detection model...")

# Load the saved model (local path)
model_path = 'object_detection_model/object_detection.pt'  # Adjust this path if your model is elsewhere

# Check alternative paths if main path doesn't exist
if not os.path.exists(model_path):
    alt_paths = [
        'object_detection/weights/best.pt',
        'object_detection/weights/last.pt'
    ]
    for path in alt_paths:
        if os.path.exists(path):
            model_path = path
            break

if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
else:
    print("❌ Model not found!")
    raise Exception("Model not found!")

# Start Webcam Stream for Live Detection
print("\n🎥 Starting webcam for real-time object detection...")

# Run prediction using webcam stream (source=0 refers to default webcam)
model.predict(
    source=0,         # 0 = Default webcam
    show=True,        # Display the live predictions in a pop-up window
    conf=0.25,        # Confidence threshold
    save=False,       # Do not save video frames
    show_labels=True, # Show class labels
    show_conf=True    # Show confidence scores
)

print("\n✨ Webcam detection ended! ✨")
