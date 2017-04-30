import argparse

from evaluate import evaluate_model
from models import parse_class_weights

parser = argparse.ArgumentParser(
    description="Provide a list of model files and a setting of class weights to visualize performance")

parser.add_argument('-m', type=str, nargs='+', help="list of paths to the saved model files")
parser.add_argument('-wtf', help="boolean for redirecting output to file", action='store_true', default=False)
args = parser.parse_args()

for model_path in args.m:
    weights = parse_class_weights(model_path)
    evaluate_model(model_path, class_weights=weights, write_to_file=args.wtf)
