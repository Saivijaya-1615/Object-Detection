# Simple Object Detection Model Testing in Jupyter
from ultralytics import YOLO
import os
from IPython.display import Image, display

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
    exit()

# Test image path (update to your Desktop image)
test_image_path = 'C:/Users/Vijaya/Desktop/img4.jpeg'
print(f"🖼 Test image: {test_image_path}")

# Check if test image exists
if not os.path.exists(test_image_path):
    print(f"❌ Test image not found: {test_image_path}")
    exit()

# Run prediction
print("🔍 Running prediction...")
results = model.predict(
    source=test_image_path,
    save=True,
    conf=0.25,
    project='.',  # Save results in the current working directory
    name='test_prediction1',
    show_labels=True,
    show_conf=True
)

# Get results
result = results[0]

# Display results
print("\n📊 PREDICTION RESULTS")
print("=" * 40)

if result.boxes is not None and len(result.boxes) > 0:
    print(f"🎯 Objects detected: {len(result.boxes)}")

    # Get detection details
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    class_names = model.names

    # Show each detection
    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
        class_name = class_names[int(cls)]
        print(f"  {i + 1}. {class_name}: {conf:.2f} confidence")

else:
    print("❌ No objects detected")

print("=" * 40)

# Display original image
print("\n📸 Original Image:")
display(Image(filename=test_image_path))

# Display prediction result
predicted_image_dir = './test_prediction1'  # Local path
if os.path.exists(predicted_image_dir):
    predicted_files = [f for f in os.listdir(predicted_image_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.avif'))]
    if predicted_files:
        predicted_image_path = os.path.join(predicted_image_dir, predicted_files[0])
        print("\n🎯 Prediction Result (with bounding boxes and labels):")
        display(Image(filename=predicted_image_path))
        print(f"✅ Result saved at: {predicted_image_path}")
    else:
        print("❌ No predicted image found")
else:
    print("❌ Prediction directory not found")

print("\n✨ Testing completed! ✨")
