from model import create_faster_rcnn, rpn_loss, detection_loss
import keras
import tensorflow as tf
from dataset import get_dataset
import numpy as np

BATCH_SIZE = 2
dataset = get_dataset()
epochs = 10
model = create_faster_rcnn()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

def train_step(image, class_true, bbox_true):
    with tf.GradientTape() as tape:
        # Forward pass
        cls_out, bounds_out, rpn_cls_logits, rpn_bounds = model(image, training=True)
        
        # Loss
        rpn_cls_loss = rpn_loss(rpn_cls_logits, class_true, rpn_bounds, bbox_true)
        detection_cls_loss = detection_loss(cls_out, class_true, bounds_out, bbox_true)
        total_loss = rpn_cls_loss + detection_cls_loss

    # Compute gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss

# Epoch loop for training
def train_model():
    for epoch in range(epochs):
        for images, bounds in dataset:
            class_true = np.zeros((BATCH_SIZE, 2))
            bbox_true = bounds

            loss = train_step(images, class_true, bbox_true)
            print(f'Epoch {epoch}, Loss: {loss}')

    model.save('model.h5')
