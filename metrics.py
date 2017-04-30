import keras.backend as K


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.

    Ignores class 1.
    '''

    pred_im = K.cast(K.equal(K.argmax(y_pred), 2), 'float32')
    true_im = K.cast(K.equal(K.squeeze(y_true, -1), 2), 'float32')

    true_positives = K.sum(K.clip(pred_im * true_im, 0, 1))
    predicted_positives = K.sum(pred_im)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.

    Ignores class 1.
    '''

    pred_im = K.cast(K.equal(K.argmax(y_pred), 2), 'float32')
    true_im = K.cast(K.equal(K.squeeze(y_true, -1), 2), 'float32')

    true_positives = K.sum(K.round(K.clip(pred_im * true_im, 0, 1)))
    possible_positives = K.sum(true_im)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fmeasure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    fmeasure = 2 * (p * r) / (p + r + K.epsilon())

    return fmeasure
