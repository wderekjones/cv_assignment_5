import argparse

from models import parse_class_weights
from visualize import visualize_model_performance

parser = argparse.ArgumentParser(
    description="Provide a list of model files and a setting of class weights to visualize performance")

parser.add_argument('-m', type=str, nargs='+', help="list of paths to the saved model files")
parser.add_argument('--s',action='store_true',default=False,help="supply this argument to save visual output")

args = parser.parse_args()

for model_path in args.m:
    weights = parse_class_weights(model_path)
    visualize_model_performance(model_path, class_weights=weights,save_output=args.s)
