from keras.layers import Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l2

from losses import load_myloss
from metrics import fmeasure
from metrics import precision
from metrics import recall


def starter_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', input_shape=(120, 180, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004), activation='softmax', padding='same'))

    model.name = "starter_model"

    return model


def model_0():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', input_shape=(120, 180, 3), dilation_rate=2))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=4))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004), activation='softmax', padding='same', dilation_rate=8))

    model.name = "model_0"

    return model


def model_1():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(.0004), padding='same', input_shape=(120, 180, 3),
               dilation_rate=2))
    model.add(PReLU())
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=4))
    model.add(PReLU())
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004), activation='softmax', padding='same', dilation_rate=8))

    model.name = "model_1"

    return model


def model_2():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(.0004), padding='same', input_shape=(120, 180, 3),
               dilation_rate=2))
    model.add(ELU())
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=4))
    model.add(ELU())
    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004), activation='softmax', padding='same', dilation_rate=8))

    model.name = "model_2"

    return model


def model_3():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(.0004), padding='same', input_shape=(120, 180, 3),
               dilation_rate=2, activation='relu'))

    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(.0004), padding='same', dilation_rate=4,
                     activation='relu'))

    model.add(BatchNormalization(momentum=.9, scale=False))
    model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004), activation='softmax', padding='same', dilation_rate=8))

    model.name = "model_3"

    return model


def load_model_from_disk(path, class_weights):
    model = load_model(path,
                       custom_objects={load_myloss(class_weights).__name__: load_myloss(class_weights),
                                       "fmeasure": fmeasure, "precision": precision,
                                       "recall": recall})
    return model


def parse_class_weights(path):
    class_weights = []
    if str.find(path, "default") != -1:
        class_weights = [0.25, 0, 1]
    elif str.find(path, "balanced") != -1:
        class_weights = [1, 0, 1]
    elif str.find(path, "reduced") != -1:
        class_weights = [.01, 0, 1]
    return class_weights
