# 🧠 Real-Time Face Detection and Classification Pipeline

A deep learning-based system that combines **Object Detection**, **Image Classification**, and **Multiprocessing** for efficient real-time inference. Built with **Python**, **OpenCV**, **TensorFlow/Keras**, and `.tflite` model quantization.

---

## 📌 Project Overview

This project demonstrates how to integrate object detection and image classification models into a single modular pipeline. It uses **frame optimization**, **database mapping**, and **multiprocessing** to deliver fast, scalable, and real-time predictions from video streams.

---

## 🔍 Key Components

### 📁 Instance Attributes

- `objdet_path`:  
  * Path to the object detection model file (e.g., `haarcascade_frontalface.xml`).

- `imgcls_path`:  
  * Path to the image classification model file (e.g., `.h5`, `.keras`, or `.tflite` model).

---

### ⚙️ Core Methods

- `load_objectdetection_model()`:  
  * Loads the object detection model from the specified `objdet_path`.

- `load_imageclassification_model()`:  
  * Loads the image classification model from the specified `imgcls_path`.

- `load_database()`:  
  * Loads a JSON-based database (`db.json`) to map predicted class indices to actual names/labels.

- `process_video(source_path, objectdetmodel, imgclassmodel, db)`:  
  * Processes video frame-by-frame: detects faces, classifies them, and displays annotated results in real time.
  * Includes preprocessing such as:
      - Frame downsampling  
      - Grayscale conversion  
      - FPS limiting to 15

- `run(video_path, model1, model2, database)`:  
  * A wrapper that calls `process_video()` with all the required components.

---

## 🔄 Version-wise Development

### ✅ Version 1: Initial Pipeline Structure
- Built the base object-oriented class structure with instance attributes:
  - `imgclspath`, `objpath`
- Implemented core methods:
  - `load_objectdetection_model()`
  - `load_imageclassification_model()`
  - `process_video()`
  - `run()` method executes the full pipeline.

---


### 🔄 Version 2: Object Detection Integration
- Integrated the object detection model directly with `process_video`.
- Initial latency observed when processing video frames with detection model.

---


### ⚙️ Version 3: Performance Optimization
- Optimizations applied to reduce latency:
  - Downsampled frame resolution by 50%.
  - Converted frames to **grayscale** before inference.
  - Reduced video FPS to **15**.
- Result: Noticeable improvement in frame processing speed.



---

### 🧠 Version 4: Image Classification Integration
- Integrated the classification model into the optimized pipeline.
- Added label prediction on detected faces.
- Issue: Classification model introduced additional latency.
- Solution:
  - Quantized classification model to `.tflite`
  - Converted weights to `float16`

---

### 🔗 Version 5: Database Mapping
- Mapped classification output index to a custom label **database**.
- Reintroduction of latency due to Keras model being used again.
- Next step: Optimize Keras classification model.

---
### 🚀 Version 6: Final Optimization with Quantization
- Applied **Post-Training Quantization (float16)** using TensorFlow.
- Converted Keras model to efficient `.tflite` format.
- Result: Smooth real-time inference with accurate object classification.

---

### 🧵 Version 7: Multiprocessing for Concurrent Video Classification
- Introduced **Python multiprocessing** to classify multiple video files simultaneously.
- Used `multiprocessing.Process` to spawn parallel processes—each handling its own video source.
- Result: Significant speedup in batch video inference, improved throughput for multi-stream setups


---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Computer Vision**: OpenCV
- **Deep Learning**: TensorFlow, Keras
- **Model Optimization**: TFLite, Float16 Quantization
- **Parallelism**: Python Multiprocessing
- **Data Format**: JSON (for class label mapping)

---


## 📁 Folder Structure


├── objdet_weights

│    └── haarcascade_frontalface.xml       # Pretrained Haar cascade for face detection

├── imgcls_weights

│   ├── mobnetv3.keras                    # Quantized image classification model (MobileNetV3, VGG19, ResNet)

│   ├── vgg19.keras  

│   └── resnet.keras

├── videos

│   └── trimmed.mp4                       # Sample input video(s) for testing

├── db.json                               # Label mapping (class index → name)

├── module.py                             # Core pipeline logic (class, model loading, video processing)

├── run.py                                # Main entry point for executing video classification (multiprocessing)

└──  README.md                             # Project documentation





