#!/usr/bin/env python3
"""
Trains one of more pretrained models in one of more dataset modes.


"""
import os
import sys
sys.path.insert(0, os.path.abspath('./code'))
from source.utils.data_loading import build_transformations, load_data
from source.LiverImg import LiverImg
from code.source.config.categories import DatasetMode, ModelNames
from source.PretrainedModel  import PretrainedModel, PretrainedConfig

import argparse

ALL = "ALL"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run liver classification models on different dataset modes.",
        usage = """
        code/scripts/models/pretrained_classify.py --model ALL --mode ALL
        code/scripts/models/pretrained_classify.py --model ConvNeXt --mode CIRRHOTIC_STATE
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[model.name for model in ModelNames.pretrained()]+ [ALL],
        help="Classifier architecture to use (e.g. EfficientNet, ConvNeXt). If omitted, runs all."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[mode.name for mode in DatasetMode.all()] + [ALL],
        help="Dataset mode to use (e.g. CIRRHOTIC_STATE). If omitted, runs all."
    )

    return parser.parse_args()

def classify(classifier:ModelNames,datasetmode:DatasetMode):
    """
    Trains one pretrained model in one dataset mode.
    Evaluates the model.
    Saves the best one.

    Parameters
    ----------
    classifier:ModelNames
        Name of the pretrained model to train. From ModelNames.

    datasetmode:DatasetMode
        Nane of the dataset mode to use. From DatasetModes.
    """
    num_classes=datasetmode.num_classes
    root_path=datasetmode.directory()
    # Load model
    model = PretrainedModel(classifier,root_path,num_classes,download_weights=True)

    #transforms = model.load(path_to_model)

    transforms = build_transformations(
            grayscale = model.grayscale,
            img_mean = model.img_mean,
            img_std = model.img_mean,
            resize = model.input_size,
            horizontal_flip = None
    )
    datasets,loaders = load_data(
        dataset_class = LiverImg,
        dataset_mode = datasetmode,
        transform = transforms,
        batch_size = PretrainedConfig.BATCH_SIZE,
        train = True,
        validation = False,
        test = True,
        test_size  = 1/3,
        only_bmode = True
    ) 


    train_loader = loaders["train"]
    valid_loader = loaders["test"]
    # Train
    model.fit(train_loader, valid_loader)

    # Evaluate on train and val sets
    print("\n[Train Evaluation]")
    model.evaluate(train_loader, save=False)

    print("\n[Validation Evaluation]")
    model.evaluate(valid_loader, save=True)

if __name__ == '__main__':
    args = parse_args()



    if args.mode == ALL:
        modes = DatasetMode.all()
    else:
        modes = [DatasetMode[args.mode]]        

    if args.model == ALL:
        classifiers = ModelNames.pretrained()
    else:
        classifiers = [ModelNames[args.model]]  


    for datasetmode in modes:
        for classifier in classifiers:
            print(f"\n--- Running {classifier.name} on {datasetmode.name} ---")
            classify(classifier, datasetmode)