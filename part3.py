from evaluate import evaluate_model
from models import *
from train import train_model
from visualize import visualize_model_performance

num_epochs = 10
model = model_0()
model_path = train_model(model, "model_0_default_weights", num_epochs)

evaluate_model(model_path)
visualize_model_performance(model_path)

model_path = train_model(model, "model_0_reduced_background_weights", num_epochs, [.01, 0, 1])
evaluate_model(model_path, [.01, 0, 1])
visualize_model_performance(model_path, [.01, 0, 1])

model_path = train_model(model, "model_0_balanced_weights", num_epochs, [1, 0, 1])
evaluate_model(model_path, [1, 0, 1])
visualize_model_performance(model_path, [1, 0, 1])
