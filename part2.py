from evaluate import evaluate_model
from losses import myloss, myloss2, myloss3
from models import *
from train import train_model
from visualize import visualize_model_performance


num_epochs = 1

model = starter_model()

defualt_model_path = train_model(model, "default_weights", num_epochs, myloss)
evaluate_model(defualt_model_path,myloss)

reduced_background_path = train_model(model,"reduced_background_weights",num_epochs,myloss2)
evaluate_model(reduced_background_path,myloss2)

balanced_path = train_model(model,"balanced_weights",num_epochs,myloss3)
evaluate_model(balanced_path,myloss3)

visualize_model_performance(defualt_model_path)
visualize_model_performance(reduced_background_path)
visualize_model_performance(balanced_path)