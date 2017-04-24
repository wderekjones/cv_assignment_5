import numpy as np
from keras import backend as K

#
# Tuning these will adjust the output of your network
# class_weights[0] = penalty for misclassifying background
# class_weights[1] = penalty for misclassifying unknown 
# class_weights[2] = penalty for misclassifying foreground 
# Setting class_weights = [.25,0,1] seems to do a reasonable job of
# balancing precision and recall, thereby giving a higher f-measure
#
class_weights = [.25,0,1] 
class_weights = np.array(class_weights,dtype=np.float32).reshape((1,1,1,3))

#
# weighted categorical crossentropy
# 
# Necessary because of imbalaced classes and "don't care" labels
#
def myloss(y_true, y_pred):

    y_true = K.squeeze(K.cast(y_true,'int32'),-1)
    y_true = K.one_hot(y_true,num_classes=3)
    loss_prelim = K.categorical_crossentropy(y_true,y_pred)

    # increase penalty on missing foreground and ignore background class
    weight = K.sum(y_true * class_weights, axis=-1)

    # apply weight and average across pixels
    loss_final = K.mean(loss_prelim * weight, axis=[-1,-2])

    return loss_final

