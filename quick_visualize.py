import argparse

from visualize import visualize_model_performance

parser = argparse.ArgumentParser(
    description="Provide a list of model files and a setting of class weights to visualize performance")

parser.add_argument('-m', type=str, nargs='+', help="list of paths to the saved model files")

args = parser.parse_args()

def parse_class_weights(path):
    class_weights = []
    if str.find(path,"default") != -1:
        class_weights = [0.25,0,1]
    elif str.find(path,"balanced") != -1:
        class_weights = [1,0,1]
    elif str.find(path,"reduced") != -1:
        class_weights = [.01,0,1]
    return class_weights

for model_path in args.m:
    weights = parse_class_weights(model_path)
    visualize_model_performance(model_path,class_weights=weights)
