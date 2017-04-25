import os, random
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

import dataset
from losses import myloss as loss
import metrics

#
# Define training and testing set generators
#

training_gen = dataset.training(os.path.join('.','baseline'))
testing_gen = dataset.testing(os.path.join('.','baseline'))

#
# Specify the CNN
#
# https://keras.io/layers/convolutional/
#

# TODO experiment with different network architectures
#
# Things to try:
#   - different number of convolutional layers
#   - different kernel shapes
#   - more or fewer filters
#   - different amount of regularization
#   - set the dilation rate
#   - pooling
#   - Use Conv2DTranspose to build an architecture with a "bottleneck"
#   - with or without BatchNormalization
#   - different amount of BatchNormalization momentum
#   - build a "Hypercolumn" architecture using upsampling
#   - different activation functions
#   - different initialization strategies
#   - incorporate pre-trained weights from VGG16 or something similar
#
# Other things you could try:
#   - different image pre processing
#   - different optimizer settings

act = LeakyReLU(alpha=0.1) # activation function

model = Sequential()
model.add(
        Conv2D(32, (3, 3), kernel_regularizer=l2(.0004),
            activation=act, padding='same',dilation_rate=2, input_shape=(120, 180, 3)))
model.add(BatchNormalization(momentum=.9,scale=False))
model.add(
        Conv2D(32, (3, 3), kernel_regularizer=l2(.0004),
            activation=act, padding='same',dilation_rate=4))
model.add(
        Conv2D(32, (3, 3), kernel_regularizer=l2(.0004),
            activation=act, padding='same',dilation_rate=8))
model.add(
        Conv2D(32, (3, 3), kernel_regularizer=l2(.0004),
            activation=act, padding='same',dilation_rate=16))
model.add(BatchNormalization(momentum=.9,scale=False))
model.add(Conv2D(3, (3, 3), kernel_regularizer=l2(.0004),
    activation='softmax', padding='same'))

#
# Define the learning process
#

optimizer = Adam(lr=0.001)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=[metrics.fmeasure]) # TODO add precision and recall

#
# Train the model
#

model.fit_generator(
        training_gen,
        epochs=50, # run this many epochs
        steps_per_epoch=20, # run this many mini batches per epoch
        validation_data=testing_gen,
        validation_steps=10 # run this many mini batches of testing data every epoch
        )

# TODO save the model to the file system (it may be useful to save
# these with different names so you can compare different models and
# training strategies)

# Run evaluation metrics on 100 mini-batches of testing data
# TODO move this to evaluation.py
score = model.evaluate_generator(testing_gen, steps=100)
print('Evaluation on Testing Data')
print('Loss = {:2.3}, F-measure= {:2.3}'.format(score[0],score[1]))

# visualize predictions
# TODO move this to visualize.py
for index, (ims, labels) in enumerate(testing_gen):

    labels_est = model.predict_on_batch(ims)

    plt.figure(1)
    plt.clf()

    plt.subplot(2,2,1)
    plt.imshow(ims[0,:,:,:])
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(labels[0,:,:,:].squeeze(),clim=(0.0,2.0))
    plt.title('Ground Truth Labels')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(labels_est[0,:,:,2].squeeze(),clim=(0.0,1.0))
    plt.title('Probability of Foreground')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(np.argmax(labels_est[0,:,:,:],axis=-1),clim=(0.0,2.0))
    plt.title('Predicted Labels')
    plt.axis('off')

    #plt.pause(.1)

    plt.savefig(str(index)+".png")

    if index == 10:
        break

