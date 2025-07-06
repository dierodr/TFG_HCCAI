#!/usr/bin/env python3
"""
Trains one of more pretrained models in one of more dataset modes.
"""
import os
import sys

import pandas as pd
sys.path.insert(0, os.path.abspath('./code'))
from source.utils.data_loading import build_transformations, load_data
from code.source.utils.files import find_best_model
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

def evaluate(classifier:ModelNames,datasetmode:DatasetMode):
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
    best_model_path = find_best_model(model.get_model_path().parent)
    transforms = model.load(best_model_path)
   


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


    # Evaluate on train and val sets
    print("\n[Train Evaluation]")
    train_result = model.evaluate(train_loader, save=False)

    print("\n[Validation Evaluation]")
    test_result = model.evaluate(valid_loader, save=False)

    print(model.final_path)

    return train_result,test_result

def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats a DataFrame of model evaluation metrics:
    - Converts selected metrics to percentage scale (0-100).
    - Converts per-class torch.Tensor values to readable lists.
    - Leaves 'classifier' unchanged.
    - Returns a transposed DataFrame with classifiers as columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing classifiers and their metrics.

    Returns
    -------
    pd.DataFrame
        Transposed and formatted DataFrame.
    """

    metrics_pct = [
        'accuracy_macro', 'accuracy_weighted', 'accuracy_per_class',
        'f1', 'auroc', 'precision', 'recall', 'specificity'
    ]

    df_formatted = df.copy()

    for col in df_formatted.columns:
        if col == 'classifier':
            continue

        for i in df_formatted.index:
            value = df_formatted.at[i, col]

            if isinstance(value, torch.Tensor):
                # Multiply tensor values by 100, round and convert to list
                value = (value * 100).round(2).tolist()
                df_formatted.at[i, col] = value

            elif isinstance(value, (float, int)):
                if col in metrics_pct:
                    df_formatted.at[i, col] = round(value * 100, 2)
                else:
                    df_formatted.at[i, col] = round(value, 2)

            elif isinstance(value, str) and value.startswith("tensor("):
                # Optional: parse stringified tensor if needed
                pass  # Leave it as is, or implement parsing if needed

    return df_formatted.set_index("classifier").T

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

        train_results=[]
        test_results=[]
        for classifier in classifiers:
            print(f"\n--- Running {classifier.name} on {datasetmode.name} ---")
            train_result,test_result = evaluate(classifier, datasetmode)
            train_result["classifier"]=classifier.name
            test_result["classifier"]=classifier.name

            train_results.append(train_result)
            test_results.append(test_result)
            print(train_result)



        df_train=format_metrics(pd.DataFrame(train_results))
        df_test=format_metrics(pd.DataFrame(test_results))
        df_train.to_csv(f"./results/{datasetmode.name}_pretrained_results_train.csv",index=True)
        df_test.to_csv(f"./results/{datasetmode.name}_pretrained_results_test.csv",index=True)
