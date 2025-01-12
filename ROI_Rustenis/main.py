import os
from ultralytics import YOLO
from training import train_model

root_dir = os.path.abspath(os.path.dirname(__file__))

model_path = os.path.join(root_dir, 'weights.pt')

if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
else:
    print("Model file not found. Training the model...")
    train_model()
    model = YOLO(model_path)

sample_dir = "Samples"

all_images = os.listdir(sample_dir)

for img_name in all_images:
    img_path = os.path.join(sample_dir, img_name)
    results = model.predict(img_path)
    print(results)