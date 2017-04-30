from evaluate import evaluate_model
from models import *
from train import train_model
from visualize import visualize_model_performance

num_epochs = 50

model0 = model_0()
model1 = model_1()
model2 = model_2()
model3 = model_3()

baseline_model_path = train_model(model0, str(model0.name) + "_p3_baseline_balanced_weights", num_epochs,
                                  class_weights=[1, 0, 1])
prelu_model_path = train_model(model1, str(model1.name) + "_p3_prelu_balanced_weights", num_epochs,
                               class_weights=[1, 0, 1])
elu_model_path = train_model(model2, str(model2.name) + "_p3_elu_balanced_weights", num_epochs, class_weights=[1, 0, 1])
relu_model_path = train_model(model3, str(model3.name) + "_p3_relu_balanced_weights", num_epochs,
                              class_weights=[1, 0, 1])

evaluate_model(baseline_model_path, class_weights=[1, 0, 1])
evaluate_model(prelu_model_path, class_weights=[1, 0, 1])
evaluate_model(elu_model_path, class_weights=[1, 0, 1])
evaluate_model(relu_model_path, class_weights=[1, 0, 1])

visualize_model_performance(baseline_model_path, class_weights=[1, 0, 1])
visualize_model_performance(prelu_model_path, class_weights=[1, 0, 1])
visualize_model_performance(elu_model_path, class_weights=[1, 0, 1])
visualize_model_performance(relu_model_path, class_weights=[1, 0, 1])
