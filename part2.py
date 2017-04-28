from evaluate import evaluate_model
from models import *
from train import train_model
from visualize import visualize_model_performance

num_epochs = 1

model = starter_model()

default_model_path = train_model(model, "default_weights", num_epochs)
evaluate_model(default_model_path)

reduced_background_path = train_model(model, "reduced_background_weights", num_epochs, [.01, 0, 1])
evaluate_model(reduced_background_path, [.01, 0, 1])

balanced_path = train_model(model, "balanced_weights", num_epochs, [1, 0, 1])
evaluate_model(balanced_path, [1, 0, 1])

visualize_model_performance(default_model_path)
visualize_model_performance(reduced_background_path, [.01, 0, 1])
visualize_model_performance(balanced_path, [1, 0, 1])
