from keras.optimizers import Adam
import dataset
import os
import metrics
from models import *


def train_model(model,model_name,num_epochs):
    training_gen = dataset.training(os.path.join('.', 'baseline'))

    optimizer = Adam(lr=0.001)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[metrics.fmeasure,metrics.precision,metrics.recall]) # TODO add precision and recall, also extra: accuracy?

    #
    # Train the model
    #

    model.fit_generator(
        training_gen,
        epochs=num_epochs, # run this many epochs
        steps_per_epoch=20, # run this many mini batches per epoch
        )

    model.save("saved_models/"+model_name+".h5")

model = starter_model()

train_model(model,"test_0",1)
