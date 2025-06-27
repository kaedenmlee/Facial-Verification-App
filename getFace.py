# get positive and anchor images

# import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid

# import TensorFlow (use only tensorflow.keras)
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Copy images into negative folder from lfw
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)


# Establish webcam - 0 for mac webcam
cap = cv2.VideoCapture(0)
crop_width, crop_height = 250, 250
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Calculate center crop coordinates
    center_x, center_y = width // 2, height // 2
    x1 = center_x - crop_width // 2
    x2 = center_x + crop_width // 2
    y1 = center_y - crop_height // 2
    y2 = center_y + crop_height // 2

    # Ensure crop stays within frame bounds
    x1 = max(0, x1)
    x2 = min(width, x2)
    y1 = max(0, y1)
    y2 = min(height, y2)

    frame = frame[y1:y2, x1:x2]
    cv2.imshow('Image Collection', frame)
    # press a to collect anchor
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # create unique filename
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write image
        cv2.imwrite(imgname, frame)

    # press p to collect positives
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # create unique filename
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write image
        cv2.imwrite(imgname, frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release webcam and close windows
cap.release()
cv2.destroyAllWindows()
