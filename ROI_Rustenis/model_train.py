from model import yolo_model
from preproc import data_generator, parse_annotations, convert_to_yolo_format
import os
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

def train_model():
    TRAIN_ANN_DIR = 'Training/annotations'
    TRAIN_IMG_DIR = 'Training/images'
    VAL_ANN_DIR = 'Validation/annotations'
    VAL_IMG_DIR = 'Validation/images'
    OUTPUT_DIR = 'Output'
    batch_size = 16

    train_annotations = parse_annotations(TRAIN_ANN_DIR, TRAIN_IMG_DIR)
    convert_to_yolo_format(train_annotations, OUTPUT_DIR)
    val_annotations = parse_annotations(VAL_ANN_DIR, VAL_IMG_DIR)
    convert_to_yolo_format(val_annotations, OUTPUT_DIR)

    # Steps per epoch
    steps_per_epoch = len(train_annotations) // batch_size
    validation_steps = len(val_annotations) // batch_size

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10,  # Stop if no improvement after 10 epochs
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'best_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )

    # If no saved model is found, train the model
    if not os.path.exists('model.h5'):
        model = yolo_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        train_gen = data_generator(train_annotations)
        val_gen = data_generator(val_annotations)

         # Train the model
        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=50,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=[early_stopping, checkpoint]
        )

        model.save("model.h5")


train_model()
