import os
import logging
from ultralytics import YOLO
from training import train_model
import cv2
import matplotlib.pyplot as plt

class ROI:

    model: YOLO

    root_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(root_dir, 'weights.pt')


    def __init__(self):
        # Turn off logging for YOLO
        logging.disable(logging.CRITICAL)

        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path, verbose=False)
        else:
            print("Model file not found. Training the model...")
            train_model(self.model)

    def detect(self, img_path):
        


sample_dir = "Samples"
output_dir = "Output"
output_path = os.path.join(root_dir, output_dir)

all_images = os.listdir(sample_dir)

for img_name in all_images:
    img_path = os.path.join(sample_dir, img_name)

    results = model.predict(img_path)

    # Print coordinates of detected boxes
    for box in results[0].boxes:
        coordinates = box.xyxy.tolist()
        print(f"Image: {img_name}, Coordinates: {coordinates}")