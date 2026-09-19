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
```


## 📂 Project Structure

EMOTION-BASED-MUSIC-RECOMMENDATION-SYSTEM/
│
├── dataset/
├── static/
├── templates/
│
├── app.py
├── train_model.py
├── emotion_model.h5
├── haarcascade_frontalface_default.xml
└── README.md


## 🚀 Getting Started

Prerequisites
Python 3.x
Webcam
Required Python libraries
Installation

Clone the repository:

git clone https://github.com/A-Sandeep-Kumar/EMOTION-BASED-MUSIC-RECOMMENDATION-SYSTEM.git

Navigate to the project directory:

cd EMOTION-BASED-MUSIC-RECOMMENDATION-SYSTEM

Install the required dependencies:

pip install -r requirements.txt
Run the Application
python app.py

Then open the application in your browser and allow webcam access when requested.

## 📊 Dataset

This project uses the FER-2013 (Facial Expression Recognition 2013) dataset for training the emotion classification model.

The dataset is not included in this repository due to its size.

## 🔮 Future Improvements
🎧 Spotify API integration for dynamic music recommendations
📈 Improve emotion classification accuracy
🧠 Experiment with advanced deep learning architectures
🎵 Personalized recommendations based on listening history
☁️ Deploy the application for online access
💡 What I Learned

## Through this project, I gained practical experience in:

Computer vision
Facial expression recognition
CNN-based image classification
OpenCV
Model training and inference
Integrating an AI model into an application
Building an end-to-end AI project

## 👨‍💻 Author

Adirala Sandeep Kumar

[Linkedin](https://www.linkedin.com/in/sandeep-kumar-aiml)



