# Face Verification App

A real-time face verification application built with Kivy and TensorFlow that uses a Siamese neural network to verify identity through webcam input.

## Overview

This application captures live video feed from a webcam and performs face verification by comparing the current frame against a set of reference images stored locally. The system uses a Siamese network architecture to determine if the person in front of the camera matches the authorized user(s).

## Features

- **Real-time Video Processing**: Live webcam feed with 30+ FPS display
- **Centered Face Cropping**: Automatically crops a 250x250 pixel region from the center of the frame
- **AI-Powered Verification**: Uses a trained Siamese neural network for face comparison
- **Instant Results**: Click "Verify" button for immediate identity verification
- **Robust Image Filtering**: Supports multiple image formats and filters out hidden/invalid files

## Technologies Used

### Core Frameworks
- **Kivy**: Cross-platform GUI framework for the user interface
- **TensorFlow/Keras**: Deep learning framework for the Siamese neural network
- **OpenCV**: Computer vision library for webcam capture and image processing

### Key Libraries
- **NumPy**: Numerical computing for array operations and image preprocessing
- **Custom L1Dist Layer**: Custom TensorFlow layer for distance calculation in the Siamese network

### Hardware Requirements
- Webcam (configurable camera index)
- Sufficient processing power for real-time video processing and neural network inference

## Architecture

### Siamese Network
The application uses a Siamese neural network architecture that:
- Takes two images as input (current frame vs reference image)
- Learns to output a similarity score between 0 and 1
- Uses L1 distance as the similarity metric
- Trained to distinguish between same person (high score) vs different person (low score)

### Verification Process
1. **Image Capture**: Captures current frame from webcam
2. **Preprocessing**: Resizes images to 100x100 pixels and normalizes pixel values
3. **Comparison**: Compares input against all reference images in the verification directory
4. **Scoring**: Applies detection threshold (0.8) and verification threshold (0.9)
5. **Decision**: Returns "Verified" or "Unverified" based on the proportion of positive matches

## Configuration

### Camera Settings
- **Resolution**: 1280x720 pixels
- **Camera Index**: 1 (configurable in code)
- **Crop Size**: 250x250 pixels (centered)
- **Frame Rate**: ~33 FPS

### Verification Thresholds
- **Detection Threshold**: 0.8 (minimum similarity score for positive detection)
- **Verification Threshold**: 0.9 (minimum proportion of positive matches required)

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- Case-insensitive extensions

## Installation

1. **Install Dependencies**:
   ```bash
   pip install kivy opencv-python tensorflow numpy
   ```

2. **Set Up Directory Structure**:
   ```bash
   mkdir -p app/application_data/input_image
   mkdir -p app/application_data/verification_images
   ```

3. **Add Reference Images**:
   Place authorized user photos in `app/application_data/verification_images/`

4. **Train/Load Model**:
   Ensure `siamesemodelv2.keras` is in the app directory

## Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Position Yourself**: Stand in front of the webcam with your face centered
3. **Click Verify**: Press the "Verify" button to perform identity verification
4. **Check Result**: The label will display "Verified" or "Unverified"

## Model Training

The Siamese network is trained on pairs of images:
- **Anchor Images**: Base images of authorized users
- **Positive Images**: Different photos of the same authorized users
- **Negative Images**: Photos of unauthorized individuals

Training focuses on minimizing the distance between positive pairs while maximizing distance between negative pairs.

## Security Considerations

- Reference images should be high-quality and represent various angles/lighting conditions
- The system may be vulnerable to photo-based spoofing attacks
- Consider implementing liveness detection for production use
- Adjust thresholds based on your security requirements

## Troubleshooting

### Common Issues
- **Camera not found**: Check camera index and permissions
- **All verifications positive**: Verify reference images exist and are properly formatted
- **Poor accuracy**: Retrain model with more diverse reference images
- **Performance issues**: Reduce frame rate or image resolution

### Debug Logging
The application logs prediction results and threshold counts for debugging purposes.

