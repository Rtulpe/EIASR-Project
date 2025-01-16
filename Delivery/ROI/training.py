import os
from random import shuffle
import glob
from helpers import process_anns_file, bounding_box_in_yolo_format
import yaml
from ultralytics import YOLO

root_dir = os.path.abspath(os.path.dirname(__file__))
train_path = os.path.join(root_dir, 'Training')
val_path = os.path.join(root_dir, 'Validation')

def train_model(model: YOLO):
    images_train_glob = glob.glob('Training/images/*.png')
    annot_train_glob = glob.glob('Training/annotations/*.xml')
    images_val_glob = glob.glob('Validation/images/*.png')
    annot_val_glob = glob.glob('Validation/annotations/*.xml')

    print(len(images_train_glob), len(annot_train_glob), len(images_val_glob), len(annot_val_glob))

    process_anns_file("Training/annotations/Cars0.xml")

    bounding_box_in_yolo_format(134, 128, 262, 160, 400, 248)

    shuffle(images_train_glob)
    shuffle(images_val_glob)
    train_range = range(len(images_train_glob))
    # Starts after the last training image
    val_range = range(len(images_train_glob), len(images_train_glob) + len(images_val_glob))

    for i in train_range:
        x1,y1,x2,y2,w,h = process_anns_file("Training/annotations/Cars{}.xml".format(i))
        x1,y1,w,h = bounding_box_in_yolo_format(x1,y1,x2,y2,w,h)
        with open("Training/labels/Cars{}.txt".format(i), "w") as file:
            file.write("0 {} {} {} {}".format(x1, y1, w, h))

    for i in val_range:
        x1,y1,x2,y2,w,h = process_anns_file("Validation/annotations/Cars{}.xml".format(i))
        x1,y1,w,h = bounding_box_in_yolo_format(x1,y1,x2,y2,w,h)
        with open("Validation/labels/Cars{}.txt".format(i), "w") as file:
            file.write("0 {} {} {} {}".format(x1, y1, w, h))

    data = dict(
        train=train_path,
        val=val_path,
        nc=1,
        names={0: "car_licence_plate"}
    )
    with open("data.yaml", "w") as ymlfile:
        yaml.dump(data, ymlfile, default_flow_style=False)

    model_path = 'weights.pt'

    model=YOLO('yolo11s.pt')
    model.train(data='data.yaml',imgsz=320,epochs=85, amp=True)
    model.save(model_path)