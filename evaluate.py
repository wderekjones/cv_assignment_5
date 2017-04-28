import os

import dataset
from models import load_model_from_disk

testing_gen = dataset.testing(os.path.join('.', 'baseline'))


def evaluate_model(path, class_weights=None):
    model = load_model_from_disk(path, class_weights)
    score = model.evaluate_generator(testing_gen, steps=100)
    print('Evaluation on Testing Data')
    print('Loss = {:2.3}, F-measure= {:2.3}, Precision = {:2.3}, Recall = {:2.3}'.format(score[0], score[1], score[2],
                                                                                         score[3]))
