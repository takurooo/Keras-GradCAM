#-------------------------------------------
# import
#-------------------------------------------
import os
import json
from model_utils import get_model, get_model_inputsize
#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))
JSON_PATH = os.path.join(CUR_PATH, 'args.json')


#-------------------------------------------
# private functions
#-------------------------------------------


def get_args():
    with open(JSON_PATH, "r") as f:
        j = json.load(f)
    return j


def main(args):
    target_model_name = args["model"]

    model, _, _ = get_model(
        target_model_name)

    model.summary()


if __name__ == '__main__':
    main(get_args())
