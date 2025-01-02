from model import yolo_model
from preproc import data_generator, parse_annotations
import os

def train_model():
    TRAIN_ANN_DIR = 'Training/annotations'
    TRAIN_IMG_DIR = 'Training/images'
    VAL_ANN_DIR = 'Validation/annotations'
    VAL_IMG_DIR = 'Validation/images'

    train_annotations = parse_annotations(TRAIN_ANN_DIR, TRAIN_IMG_DIR)
    val_annotations = parse_annotations(VAL_ANN_DIR, VAL_IMG_DIR)

    # If no saved model is found, train the model
    if not os.path.exists('model.h5'):
        model = yolo_model()
        

        train_gen = data_generator(train_annotations)
        val_gen = data_generator(val_annotations)

        model.fit(
        train_gen,
        steps_per_epoch=len(train_annotations) // 8,  # You can adjust the batch size
        epochs=5,
        validation_data=val_gen,
        validation_steps=len(val_annotations) // 8  # Adjust based on your validation set size
        )

        model.save("model.h5")


train_model()
