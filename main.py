import argparse
import json
from src.models import Models


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("ml_step", help="MLOps Step")
    parser.add_argument("input_json", help="JSON input file")
    args = parser.parse_args()
    return args.ml_step, args.input_json

def load_json(json_path):
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    return json_dict

if __name__ == "__main__":
    ml_step, input_json = get_arguments()
    task_info = load_json(input_json)

    if ml_step == "inference":
        models = Models(task_info, ml_step)
        input_data = models.get_data()
        models.load()
        models.predict(input_data)
    else:
        pass