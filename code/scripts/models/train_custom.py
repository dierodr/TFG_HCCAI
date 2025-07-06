#!/usr/bin/env python3

"""
Training entry point for Liver Image classification using CustomCNN.


"""

from pprint import pprint
from code.source.config.categories import DatasetMode
from code.source.CustomCNN import get_hyperparameters, main_loop
from code.source.LiverImg import LiverImg
from datetime import datetime

import argparse

from code.source.config.images import ImageNormalization



ALL = "ALL"
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run liver classification models on different dataset modes.",
        usage = """train_model.py --mode ALL|ORGAN_CLASSIFICATION|CIRRHOTIC_STATE|HEALTHY_LIVERS_OR_NOT
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=[mode.name for mode in DatasetMode.all()] + [ALL],
        help="Dataset mode to use (e.g. CIRRHOTIC_STATE). "
    )
    return parser.parse_args()

def main(dataset_mode):
    print(f"\t + On: {dataset_mode.name}")

    # Set dataset class and training hyperparameters
    dataset_class = LiverImg


    norm_stats=ImageNormalization.SIMPLE_GRAY
    hyperparameters = get_hyperparameters(grayscale=True,img_mean=norm_stats[0],img_std=norm_stats[1])
    print(f"\n--- HYPERPARAMETERS ---")
    pprint(hyperparameters)
    # Run main training loop
    model = main_loop(
        dataset_class=dataset_class,
        dataset_mode=dataset_mode,
        hyperparameters=hyperparameters,
        validation=True,
        test_as_validation=True,
        force_cpu=False,
        root_data_dir="./data/"
    )
    print(" ----------------------- \n")



if __name__ == '__main__':
    print(datetime.now())
 
    args = parse_args()
    if args.mode == ALL:
        modes = DatasetMode.all()
    else:
        modes = [DatasetMode[args.mode]]
    print("Training CustomCNN with:\n")
    for datasetmode in modes:
        main(datasetmode)