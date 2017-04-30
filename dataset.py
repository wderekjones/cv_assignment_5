import os, glob, random
import numpy as np
import keras
from keras.preprocessing import image


def load_image_and_label(fname):
    fname_gt = fname.replace('input' + os.sep + 'in', 'groundtruth' + os.sep + 'gt').replace('jpg', 'png')

    img = image.load_img(fname, target_size=(120, 180))
    label = image.load_img(fname_gt, target_size=(120, 180), grayscale=True)

    img = image.img_to_array(img) / 255.
    label = image.img_to_array(label)

    # sparse indexing with many classes
    label[label == 50] = 1  # ignore
    label[label == 85] = 1  # ignore
    label[label == 170] = 1  # ignore
    label[label == 255] = 2

    return img, label


def minibatch(file_names, batch_size):
    imgs = []
    labels = []

    for fname in random.sample(file_names, batch_size):
        img, label = load_image_and_label(fname)

        imgs.append(img)
        labels.append(label)

    return np.stack(imgs), np.stack(labels)


def testing(data_dir, batch_size=8):
    '''Import data and form batches of testing images.'''

    file_names = glob.glob(os.path.join(data_dir, '*', 'input', '*.*'))
    file_names = [s for s in file_names if s[-8] != '0']  # filter out training images

    while True:
        yield minibatch(file_names, batch_size)


def training(data_dir, batch_size=8):
    '''Import data and form batches of training images.'''

    file_names = glob.glob(os.path.join(data_dir, '*', 'input', '*.*'))
    file_names = [s for s in file_names if s[-8] == '0']  # include only testing images

    while True:
        yield minibatch(file_names, batch_size)
