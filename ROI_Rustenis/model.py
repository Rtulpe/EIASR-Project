from keras.api.applications import VGG16
from keras.api.layers import Conv2D, Dense, Flatten, Input
from keras.api.models import Model
import tensorflow as tf

# Configuration
IMG_WIDTH = 800
IMG_HEIGHT = 600

# Faster-RCNN-like model, with VGG16 as the base model
def create_base_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    return base_model


# Region Proposal Network (RPN)
# This takes the output of the base model and generates region proposals
def create_rpn(base_model_output, num_anchors=9):
    rpn_conv = Conv2D(512, (3, 3), padding='same', activation='relu')(base_model_output)
    rpn_cls_logits = Conv2D(num_anchors, (1, 1), padding='same', activation='sigmoid')(rpn_conv)
    # This is for the bounding box regression
    rpn_bounds = Conv2D(num_anchors * 4, (1, 1), padding='same')(rpn_conv)

    return rpn_cls_logits, rpn_bounds

# Pooling for the Region of Interest (ROI)
def roi_align(features, boxes, output_size=(7, 7)):
    return tf.image.crop_and_resize(features, boxes, box_indices=tf.zeros(len(boxes), dtype=tf.int32), crop_size=output_size)


# Object Detection
def create_object_detection_model(rois):
    x = Flatten()(rois)
    x = Dense(1024, activation='relu')(x)

    # Classification, first parameter is the number of classes.
    # We either have license plate or not, so 2 classes
    cls_out = Dense(2, activation='softmax', name='cls_out')(x)

    # Bounding box regression
    bounds_out = Dense(4, activation='linear', name='bounds_out')(x)

    return cls_out, bounds_out

# Combining all the models to have
# Something like Faster-RCNN
def create_faster_rcnn():
    # Feature extraction
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model = create_base_model()
    base_output = base_model(input_img)

    # Region Proposal
    rpn_cls_logits, rpn_bounds = create_rpn(base_output)

    # ROI Pooling
    rois = roi_align(base_output, rpn_bounds)

    # Detection
    cls_out, bounds_out = create_object_detection_model(rois)

    # Final model
    model = Model(inputs=input_img, outputs=[cls_out, bounds_out, rpn_cls_logits, rpn_bounds])
    return model


# Some random loss functions
def rpn_loss(class_logits, class_true, bbox_pred, bbox_true):
    class_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(class_true, class_logits))
    bbox_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(bbox_true, bbox_pred))
    return class_loss + bbox_loss

def detection_loss(class_logits, class_true, bbox_pred, bbox_true):
    class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(class_true, class_logits))
    bbox_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(bbox_true, bbox_pred))
    return class_loss + bbox_loss
