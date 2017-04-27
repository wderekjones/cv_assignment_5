import matplotlib.pyplot as plt
import numpy as np
import dataset
import os
from models import *
# visualize predictions

def visualize_model_performance(path):
    testing_gen = dataset.testing(os.path.join('.', 'baseline'))

    model = load_model_from_disk(path)

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

        plt.pause(.1)

        if index == 10:
            break


visualize_model_performance("saved_models/test_0.h5")