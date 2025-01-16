import os
from random import shuffle
import glob
from helpers import process_anns_file, bounding_box_in_yolo_format
import yaml
from ultralytics import YOLO

root_dir = os.path.abspath(os.path.dirname(__file__))
train_path = os.path.join(root_dir, 'Training')
val_path = os.path.join(root_dir, 'Validation')
model_path = 'weights.pt'

def train_model(model: YOLO = None):
    check_folders()

    images_train_glob = get_globs('Training/images/*.png')
    annot_train_glob = get_globs('Training/annotations/*.xml')
    images_val_glob = get_globs('Validation/images/*.png')
    annot_val_glob = get_globs('Validation/annotations/*.xml')

    if model is None:
        print("Model not found, fetching a new one...")
        model = YOLO('yolo11s.pt')

    print("Found {} training images and {} annotations".format(len(images_train_glob), len(annot_train_glob)))
    print("Found {} validation images and {} annotations".format(len(images_val_glob), len(annot_val_glob)))

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

    model.train(data='data.yaml',imgsz=320,epochs=85, amp=True)
    model.save(model_path)


def check_folders():
    if not os.path.exists(train_path):
        raise FileNotFoundError("Training folder not found")
    if not os.path.exists(val_path):
        raise FileNotFoundError("Validation folder not found")
    
def get_globs(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("Nothing found for pattern: {}".format(pattern))
    return files


train_model()