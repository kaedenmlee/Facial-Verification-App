# import kivy dependencies
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# build app and layout


class CamApp(App):
    def build(self):
        # layout components
        self.webcam = Image(size_hint=(1, .8), allow_stretch=True)
        self.button = Button(
            text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(
            text="Verification Uninitiated", size_hint=(1, .1))

        # add widgets to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # load model
        self.model = tf.keras.models.load_model(
            'siamesemodelv2.keras', custom_objects={'L1Dist': L1Dist})

        # video capture
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

        # crop dimensions
        self.crop_width, self.crop_height = 250, 250

        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Get frame dimensions
        height, width, _ = frame.shape

        # Calculate center crop coordinates
        center_x, center_y = width // 2, height // 2
        x1 = center_x - self.crop_width // 2
        x2 = center_x + self.crop_width // 2
        y1 = center_y - self.crop_height // 2
        y2 = center_y + self.crop_height // 2

        # Ensure crop stays within frame bounds
        x1 = max(0, x1)
        x2 = min(width, x2)
        y1 = max(0, y1)
        y2 = min(height, y2)

        # Apply the centered crop
        frame = frame[y1:y2, x1:x2]

        # Convert to texture for Kivy display
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    # image preprocessing
    def preprocess(self, file_path):
        # read in image from file path
        byte_img = tf.io.read_file(file_path)
        # load in the image
        img = tf.io.decode_jpeg(byte_img)

        # preprocess the image
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    # verification function
    def verify(self, *args):
        detection_threshold = 0.8
        verification_threshold = 0.9
        print(self.model.summary())
        print(self.model.get_weights())

        # capture input image
        SAVE_PATH = os.path.join(
            'application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        # Get frame dimensions
        height, width, _ = frame.shape

        # Calculate center crop coordinates
        center_x, center_y = width // 2, height // 2
        x1 = center_x - self.crop_width // 2
        x2 = center_x + self.crop_width // 2
        y1 = center_y - self.crop_height // 2
        y2 = center_y + self.crop_height // 2

        # Ensure crop stays within frame bounds
        x1 = max(0, x1)
        x2 = min(width, x2)
        y1 = max(0, y1)
        y2 = min(height, y2)
        frame = frame[y1:y2, x1:x2]
        cv2.imwrite(SAVE_PATH, frame)

        # build results array
        results = []

        # Define valid image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif',
                            '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP')
        verification_dir = os.path.join(
            'application_data', 'verification_images')

        for image in os.listdir(verification_dir):
            # Skip hidden files and non-image files
            if image.startswith('.') or not image.endswith(image_extensions):
                continue
            input_img = self.preprocess(os.path.join(
                'application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(
                os.path.join(verification_dir, image))
            # Make predictions
            result = self.model.predict(
                list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Detection Threshold : metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: proportion of positive predictions/ total positive samples
        verification = detection / len(os.listdir(verification_dir))
        verified = verification > verification_threshold

        # set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # log out
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))

        return results, verified


if __name__ == '__main__':
    CamApp().run()
