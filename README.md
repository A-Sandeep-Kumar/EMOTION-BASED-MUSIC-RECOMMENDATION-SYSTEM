# Emotion-Based Music Recommendation System

## Overview
This project is a real-time emotion-based music recommendation system that detects user emotions using facial expressions and suggests songs accordingly.

## Features
- Real-time emotion detection using webcam
- Face detection using Haar Cascade
- Emotion classification using CNN model trained on FER-2013 dataset
- Music recommendation based on detected emotions
- Future enhancement: Spotify API integration

## Technologies Used
- Python
- OpenCV
- Deep Learning (CNN)
- FER-2013 Dataset

## How It Works
1. Capture image from webcam
2. Detect face using Haar Cascade
3. Pass cropped face to CNN model
4. Predict emotion (happy, sad, neutral, etc.)
5. Recommend songs based on emotion

## Dataset
FER-2013 dataset (not included due to size)

## Future Improvements
- Spotify API integration
- Improved accuracy using advanced models
