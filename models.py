from keras.layers import Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l2

from metrics import fmeasure
from metrics import precision
from metrics import recall

from losses import load_myloss

def load_model_from_disk(path,class_weights):
    model = load_model(path,
                       custom_objects={load_myloss(class_weights).__name__: load_myloss(class_weights), "fmeasure": fmeasure, "precision": precision,
                                       "recall": recall})
    return model


def starter_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', input_shape=(120, 180, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004), activation='softmax', padding='same'))

    return model


def model_0():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=2, input_shape=(120, 180, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=4))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=8))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=16))
    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004),
                     activation='softmax', padding='same'))

    return model
