import os
import xml.etree.ElementTree as ET
from keras.api.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import random

def parse_annotations(annotations_dir, images_dir):
    annotations = []
    for file in os.listdir(annotations_dir):
        if not file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(annotations_dir, file))
        root = tree.getroot()
        
        image_filename = root.find('filename').text
        image_path = os.path.join(images_dir, image_filename)
        
        objects = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            objects.append((xmin, ymin, xmax, ymax))
        
        annotations.append((image_path, objects))
    return annotations

def convert_to_yolo_format(annotations, output_dir, image_size=(416, 416)):
    os.makedirs(output_dir, exist_ok=True)
    for image_path, objects in annotations:
        h, w = image_size
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(output_dir, f"{image_name}.txt")
        
        with open(txt_path, 'w') as f:
            for xmin, ymin, xmax, ymax in objects:
                class_id = 0  # Since only 'license' plates are labeled
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                print(f"{class_id} {x_center} {y_center} {width} {height}")
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def preprocess_image(image_path, target_size=(416, 416)):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")
    
    # Convert from BGR to RGB (optional but commonly done for consistency)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, target_size)

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    return np.array(image, dtype=np.float32)


def data_generator(annotations, input_size=(416, 416), batch_size=8):
    while True:
        batch_images = []
        batch_labels = []
        
        for image_path, objects in annotations[:batch_size]:
            image = preprocess_image(image_path, input_size)
            labels = np.zeros((input_size[0], input_size[1], 5))  # Adjust dimensions
            
            batch_images.append(image)
            batch_labels.append(labels)
        
        yield np.array(batch_images), np.array(batch_labels)

def validation_data_generator(annotations, input_size=(416, 416), batch_size=8):
    while True:
        batch_images = []
        
        # Loop through annotations in batches
        for image_path, _ in annotations[:batch_size]:
            # Preprocess the image
            image = preprocess_image(image_path, input_size)
            batch_images.append(image)
        
        yield np.array(batch_images)


def split_data(annotations, split=0.2):
    random.shuffle(annotations)

    split_index = int(len(annotations) * (1 - split))
    train_annotations = annotations[:split_index]
    val_annotations = annotations[split_index:]

    return train_annotations, val_annotations