# 🕵️‍♂️ Real-Time Face Anonymizer Using MediaPipe & OpenCV

[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceDetection-green)](https://google.github.io/mediapipe/solutions/face_detection)

A privacy-preserving tool that automatically detects and anonymizes faces using computer vision. This project detects and **automatically blurs faces** in images, video files, or live webcam feeds. It's ideal for anonymizing visual data quickly and efficiently using **MediaPipe's face detection** model and **OpenCV**.

🎬 **Demo Comparison:**  
See the real-time face detection and blurring in action below:

| Before | After |
|--------|-------|
| ![Before](./before.gif) | ![After](./demo.gif) |


---

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenCV** – for frame capture and image processing
- **MediaPipe** – for fast and accurate face detection

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- A webcam (for live demo)
- A sample image or video (if not using webcam)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abdelrahman-Awd/face-anonymizer.git
   cd face-anonymizer
    ```

2. Install required libraries:
    ```bash 
    pip install opencv-python mediapipe
    ```

### Usage
    ``` bash
    # Live webcam anonymization (press Q to quit)
    python main.py --mode webcam

    # Process an image (saves to output/output.png)
    python main.py --mode image --filePath input.jpg

    # Process a video (saves to output/output.mp4)
    python main.py --mode video --filePath input.mp4
    ```

---

## 🧠 How It Works
1. MediaPipe detects faces in each frame using a pretrained model.
2. The relative bounding boxes are converted to pixel coordinates.
3. Each detected face region is blurred using OpenCV’s cv2.blur() function.
4. The modified image or video is saved or shown live.

---

## 📁 File Structure
    ```plaintext
    .
    ├── main.py            # Main processing script with all modes
    ├── output/            # Directory containing processed outputs
    │   ├── output.png     # Blurred image output
    │   └── output.mp4     # Blurred video output
    ├── before.gif         # GIF showing the original unblurred video
    ├── demo.gif           # GIF showing the anonymized (blurred) video
    └── README.md          # Project documentation 
    ```

---

## 💡 Key Features

- Multi-face detection in single frames
- Three processing modes: Image, Video, and Live Webcam
- Configurable blur intensity (via code modification)
- Preserved resolution in output files

> Press Q to exit live webcam mode.