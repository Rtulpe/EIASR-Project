from keras.api import Model
from keras.api.layers import Input, Conv2D, BatchNormalization, LeakyReLU

def yolo_model(input_shape=(416, 416, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (3, 3), strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(num_classes * 5, (1, 1), strides=1, padding="same")(x)  # Adjust output dimensions
    
    model = Model(inputs, x)
    return model
