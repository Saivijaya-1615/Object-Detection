🚦 Real-Time Object Detection using YOLOv8
This project focuses on detecting traffic lights and speed limit signs in real time using a custom-trained YOLOv8 model.
We built the model from scratch, collected our own dataset, and achieved high accuracy and real-time performance.

📌 What is Object Detection?
Object Detection is a computer vision technique that identifies and locates objects within an image or video.
In this project, we trained our model to detect:

🚦 Traffic Lights

🚫 Speed Limit Signs

⚙️ How We Built It
📂 Collected images from Google and Kaggle.

🏷️ Labeled datasets using Roboflow.

⚡ Used YOLOv8 (You Only Look Once version 8) for fast, real-time detection.

🛠️ Trained the model with our custom dataset.

🔍 Tested it on both static images and live webcam streams.

🎯 Project Results
✅ Successfully detects traffic lights and speed signs in real time.

🎯 Achieved ~92% accuracy and precision.

📸 Works on both test images and live video.

🚧 Challenges Faced
Difficult to find a diverse, high-quality dataset with different:

Camera angles

Lighting conditions

Real-world traffic scenarios

Solved by carefully collecting images from multiple sources and improving dataset diversity.

🚀 Why YOLOv8?
We chose YOLOv8 because:

⚡ It’s fast and supports real-time detection.

🎯 It’s accurate and lightweight.

🧩 It works well with custom datasets.

📂 Project Structure
text
Copy
Edit
.
├── object_detection_model/
│    └── object_detection.pt       # Trained YOLOv8 model
├── image_test_detection.py        # Image-based detection script
├── real_time_detection.py         # Real-time webcam detection script
└── README.md                      # Project documentation
🛠️ Requirements
Install the required libraries:

bash
Copy
Edit
pip install ultralytics opencv-python ipython
💻 How to Run
🖼️ Static Image Detection:
bash
Copy
Edit
python image_test_detection.py
Update the image path inside the script.

🎥 Real-Time Webcam Detection:
bash
Copy
Edit
python real_time_detection.py
Make sure your webcam is connected.

📚 Tools Used
YOLOv8 - Ultralytics YOLO

Roboflow - For dataset labeling

Python

OpenCV

✨ Final Notes
This project shows that custom real-time object detection is achievable with careful dataset preparation and the right model.
The same method can be extended to detect:

Vehicles

Helmets

Lane markings

Pedestrians

🙌 Feel free to fork, use, and improve this project!
