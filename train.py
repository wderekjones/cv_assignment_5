import os
import dataset
import metrics

from losses import load_myloss
from keras.optimizers import Adam


'''
contains utilities related to model training process'''

def train_model(model, model_name, num_epochs, class_weights=None):
    training_gen = dataset.training(os.path.join('.', 'baseline'))

    optimizer = Adam(lr=0.001)

    model.compile(
        loss=load_myloss(class_weights),
        optimizer=optimizer,
        metrics=[metrics.fmeasure, metrics.precision,
                 metrics.recall])  # TODO add precision and recall, also extra: accuracy?

    #
    # Train the model
    #

    model.fit_generator(
        training_gen,
        epochs=num_epochs,  # run this many epochs
        steps_per_epoch=20,  # run this many mini batches per epoch
    )

    model_path = "saved_models/" + model_name + ".h5"
    model.save(model_path)

    return model_path
