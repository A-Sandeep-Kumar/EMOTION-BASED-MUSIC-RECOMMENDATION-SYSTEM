# 🎵 Emotion-Based Music Recommendation System

> An AI-powered system that detects facial emotions in real time and recommends music based on the detected emotional state.

## 🧠 Overview

The **Emotion-Based Music Recommendation System** combines computer vision and deep learning to recognize a user's facial emotion through a webcam and recommend music accordingly.

The system uses **OpenCV** for face detection and a **CNN-based emotion classification model** trained using the **FER-2013 dataset**.

The project explores how AI can connect **human emotions with personalized music recommendations**.

## ✨ Features

- 🎥 Real-time emotion detection using a webcam
- 👤 Face detection using Haar Cascade
- 🧠 CNN-based facial emotion classification
- 🎵 Emotion-based music recommendation
- ⚡ Real-time prediction and interaction
- 🖥️ Web-based application interface

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| OpenCV | Face detection and image processing |
| TensorFlow / Keras | Deep learning model |
| CNN | Emotion classification |
| FER-2013 | Emotion recognition dataset |
| HTML / CSS | Web interface |

## 🔄 How It Works

```text
Webcam
   ↓
Capture Video Frame
   ↓
Detect Face using Haar Cascade
   ↓
Preprocess Facial Image
   ↓
CNN Emotion Classification
   ↓
Identify Emotional State
   ↓
Recommend Suitable Music
