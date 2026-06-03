# Real-time-Digit-Recognition-System
A real-time digit recognition system that uses a CNN model trained on MNIST and uses OpenCV to detect handwritten digits from webcam input.
<h1 align="center">🔢 Real-time Digit Recognition System</h1>

<p align="center">
 A real-time digit recognition system that uses a CNN model trained on MNIST and uses OpenCV to detect handwritten digits from webcam input. 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📌 Project Overview

This project is a **real-time handwritten digit recognition system** built using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. It captures live webcam input, applies a series of image preprocessing techniques via **OpenCV**, and predicts the handwritten digit (0–9) along with a confidence score.

The system comes with **two interface options**:
- 🖥️ **Tkinter Desktop App** — A dark-themed native GUI that shows the live camera feed, detected digit, and confidence score.

---

## ✨ Features

- 🎥 **Live Webcam Feed** — Captures frames in real-time directly from your webcam
- 🧠 **CNN-based Digit Classification** — Pre-trained on the MNIST dataset to classify digits 0–9 with high accuracy
- 🔲 **Bounding Box Detection** — Automatically draws a bounding box around the largest detected digit contour
- 🧹 **Advanced Image Preprocessing** — Applies Gaussian blur, adaptive thresholding, and morphological operations for clean digit isolation
- 📊 **Confidence Score Display** — Shows prediction confidence in real-time; uncertain predictions below the threshold are filtered out
- 🔄 **Prediction Smoothing** — Uses a rolling history buffer (`deque`) to stabilize predictions and reduce flickering
- ⚡ **FPS Tracking** — Built-in frame rate calculation for performance monitoring
- 🎛️ **Dual Interface** — Run as a desktop app (Tkinter) or a web app (Streamlit)

---

## 🛠️ Tech Stack

| Category           | Technology                          |
|--------------------|--------------------------------------|
| Language           | Python 3.12+                        |
| Deep Learning      | TensorFlow / Keras                  |
| Computer Vision    | OpenCV (`cv2`)                       |
| Dataset            | MNIST (handwritten digits)          |
| Desktop UI         | Tkinter               |
| Numerical Computing| NumPy                               |
| Model Format       | `.keras` (Keras v3 native format)   |

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
Grayscale Conversion
     │
     ▼
Gaussian Blur  (5×5 kernel)
     │
     ▼
Adaptive Thresholding  (ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV)
     │
     ▼
Morphological Opening  (noise removal)
     │
     ▼
Contour Detection  (largest contour selected, area > 500px)
     │
     ▼
Digit Cropped → Padded to Square → Resized to 28×28
     │
     ▼
Normalized → Reshaped to (1, 28, 28, 1)
     │
     ▼
CNN Model Prediction
     │
     ▼
Confidence Check  (threshold: 60%)
     │
     ▼
Rolling History Buffer (deque, size 20) → Stable Prediction
     │
     ▼
Display: Digit + Confidence Score + Bounding Box
```

---

## 📁 Project Structure

```
Real-time Digit Recognition System/
│
├── main.py                        # Tkinter desktop GUI application
├── webcam_digit_recognition.py    # Core logic: model loading, preprocessing, prediction, FPS
├── digit_cnn_model.keras          # Pre-trained CNN model (MNIST)
├── .gitignore                     # Python gitignore
└── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

### ✅ Prerequisites

Make sure you have the following installed:

- Python **3.12** or higher
- A working **webcam**
- `pip` package manager

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ashfak-Uzzaman-Khan/-Real-time-Digit-Recognition-System.git
cd Real-time-Digit-Recognition-System
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install tensorflow opencv-python pillow numpy streamlit
```

---

## ▶️ Running the Application

### 🖥️ Option 1 — Tkinter Desktop App

```bash
python main.py
```

- A dark-themed GUI window  will open.
- Click **"Start Camera"** to begin live digit recognition.
- Click **"Stop Camera"** to stop.
- The predicted digit and confidence score will be displayed in real-time below the video feed.


## 📸 Screenshots

> _Screenshots will be added soon._

<!-- 
| Tkinter Desktop App | Streamlit Web App |
|---|---|
| ![Tkinter UI]() | ![Streamlit UI]() |
-->

---

## 🔍 Implementation Highlights

### Image Preprocessing Pipeline
The raw webcam frame goes through several OpenCV transformations before being fed to the model:
- **Gaussian Blur** smooths out noise in the frame
- **Adaptive Thresholding** binarizes the image based on local pixel intensity, making it robust to lighting changes
- **Morphological Opening** removes small noise blobs while preserving the digit shape
- The largest contour (by area) is selected, cropped, padded to a square, and resized to the MNIST-standard **28×28 pixels**

### Prediction Stability
To avoid flickering predictions frame-to-frame, the system maintains a **rolling buffer** using Python's `deque` (size = 20 frames). Only predictions with confidence **above 60%** are added to this buffer. The final displayed digit is the **most frequent prediction** in the buffer, ensuring visual stability.

### Model
The CNN model (`digit_cnn_model.keras`) was trained on the **MNIST dataset** — 60,000 training images of handwritten digits (0–9) at 28×28 pixels. It was saved in Keras's native `.keras` format for efficient loading.

---

<p align="center">
  If you found this project helpful, please consider giving it a ⭐ on GitHub!
</p>
