import os
import logging
from ultralytics import YOLO
from .training import train_model
from .helpers import cut_image

class RegionOfInterest:

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
        results = self.model.predict(img_path)

        img_list = []
        for box in results[0].boxes:
            blist = box.xyxy.tolist()
            x1 = int(blist[0][0])
            y1 = int(blist[0][1])
            x2 = int(blist[0][2])
            y2 = int(blist[0][3])
            print(f"Bounding box: {x1}, {y1}, {x2}, {y2}")
            img_list.append(cut_image(img_path, x1, y1, x2, y2))

        return img_list