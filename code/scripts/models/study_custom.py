#!/usr/bin/env python3

import argparse

from pathlib import Path
import traceback


from torch import optim

import torch  # main torch llibrary


import optuna
from optuna.samplers import TPESampler

from code.source.utils.data_loading import build_transformations, get_input_dimensions, load_data
from code.source.utils.models_common import get_loss_function
from code.source.CustomCNN import CustomCNN, get_hyperparameters
from code.source.LiverImg import LiverImg
from code.source.config.categories import DatasetMode
from code.source.config.images import ImageNormalization, ImageTrim


validation = True
test_as_validation = True

STORAGE = "sqlite:///optuna_study.db"
STUDY_NAME = "cnn_"

MAX_EPOCHS = 60

failed_trials = []


def objective(trial):

    try:

        choice = trial.suggest_categorical(
            "grayscale_and_norm",
            [
                "gray_imagenet",
                "gray_simple",
                "color_imagenet",
                "color_simple"
            ]
        )

        grayscale = "gray" in choice

        if choice == "gray_imagenet":
            norm_stats = ([0.084], [0.1069])
        elif choice == "gray_simple":
            norm_stats = ([0.5], [0.5])
        elif choice == "color_imagenet":
            norm_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:  # "color_simple"
            norm_stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        hyperparameters = get_hyperparameters(
            grayscale=grayscale, img_mean=norm_stats[0], img_std=norm_stats[1], max_epochs=MAX_EPOCHS)
        # OPTIMIZER

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        # SHEDULER

        scheduler_name = trial.suggest_categorical(
            "scheduler", ["ReduceLROnPlateau"])
        # DATA TRANSFORMS

        dataset_class = LiverImg
        dataset_mode = DatasetMode[data_mode]
        print(f"Selected mode: {dataset_mode} - {dataset_mode.name}")

        # -----------------------------
        # Dataset Paths and Loading
        # -----------------------------
        root_data_dir = Path.cwd()/"data/"
        model_directory = root_data_dir / \
            f"model_state/{dataset_mode.name.lower()}/"

        transform = build_transformations(
            grayscale=grayscale, img_mean=norm_stats[0], img_std=norm_stats[1], resize=ImageTrim().resize(2))

        # -----------------------------
        # Load Dataset & Dataloaders
        # -----------------------------
        print(f"Using Test as Validation inside Train: {test_as_validation}")
        dataset, loaders = load_data(
            dataset_class=dataset_class,
            dataset_mode=dataset_mode,
            transform=transform,
            batch_size=hyperparameters["batch_size"],
            # Only pass `validation` if not using test as validation
            validation=not test_as_validation and validation
        )

        if test_as_validation:
            dataset["validation"] = dataset["test"]
            loaders["validation"] = loaders["test"]
        print([f"{dat[0]} size:{len(dat[1])}" for dat in dataset.items()])
        input_dimensions = get_input_dimensions(loaders["train"])
        num_classes = len(dataset["test"].classes)
        print(f"Input_dimensions:{input_dimensions}")
        print("Classes:", dataset["test"].classes)
        print(f"Number of Classes: {num_classes}")

        # -----------------------------
        # Create Model
        # -----------------------------
        data_dir = f"./data/model_state/{dataset_mode.name.lower()}/custom/"
        model = CustomCNN(input_dimensions=input_dimensions,
                          num_classes=num_classes,
                          cv_channels=hyperparameters["cv_layers"],
                          cv_kernels=hyperparameters["cv_kernels"],
                          cv_strides=hyperparameters["cv_strides"],
                          cv_paddings=hyperparameters["cv_paddings"],
                          pool_kernels=hyperparameters["pool_kernels"],
                          pool_strides=hyperparameters["pool_strides"],
                          pool_paddings=hyperparameters["pool_paddings"],
                          hidden_layers=hyperparameters["fc_layers"],
                          batch_norm=hyperparameters["batch_norm"],
                          fcl_dropouts=hyperparameters["fcl_dropouts"],
                          directory=data_dir)

        device = model.send_to_device()

        criterion = get_loss_function(
            use_weights=True,
            num_classes=num_classes,
            dataset=dataset["train"],
            device=device,
        )

        if optimizer_name == "RMSprop":
            optimizer = getattr(optim, optimizer_name)(
                model.parameters(), lr=hyperparameters["learning_rate"])
        else:
            weight_decay = trial.suggest_float(
                'weight_decay', 1e-6, 1e-2, log=True)
            optimizer = getattr(optim, optimizer_name)(
                model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=weight_decay)

        if scheduler_name == "StepLR":
            step_size = trial.suggest_int("step_size", 2, 10)
            gamma = trial.suggest_float("step_gamma", 0.1, 0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "ReduceLROnPlateau":
            factor = trial.suggest_float("plateau_factor", 0.1, 0.9)
            patience = trial.suggest_int("plateau_patience", 2, 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=factor, patience=patience)
        elif scheduler_name == "CosineAnnealingLR":
            T_max = trial.suggest_int("T_max", 5, 50)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max)

        print("Criterion:", criterion)
        print("Scheduler:", scheduler)
        print("Hyperparameters:", hyperparameters)
        # -----------------------------
        # Train Model
        # -----------------------------

        training_losses, validation_losses, validation_accuracies = model.fit(
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparameters=hyperparameters,
            trial=trial,
            validation=validation)
        return validation_accuracies[-1]

    except optuna.exceptions.TrialPruned:
        print(
            f"Trial {trial.number} was pruned.\n---------------------------------\n")

        raise

    except Exception as e:
        print(f"Trial {trial.number} failed due to error: {e}")
        print(traceback.format_exc())
        failed_trials.append((trial.number, str(e)))
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run liver classification models on different dataset modes.",
        usage="""study_custom.py --mode [ORGAN_CLASSIFICATION | CIRRHOTIC_STATE | HEALTHY_LIVERS_OR_NOT]
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=[mode.name for mode in DatasetMode.all()],
        help="Dataset mode to use (e.g. CIRRHOTIC_STATE). "
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_mode = args.mode

    sampler = TPESampler()

    try:
        optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE)
        print(f"Deleted existing study: {STUDY_NAME}")
    except KeyError:
        print(f"Study '{STUDY_NAME}' did not exist.")
    except Exception as e:
        print(f"Unexpected error when trying to delete study: {e}")
    # optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE)
    study = optuna.create_study(study_name=STUDY_NAME,
                                direction="maximize",
                                storage=STORAGE,
                                load_if_exists=False,
                                sampler=sampler)

    try:
        study.optimize(objective)
    except KeyboardInterrupt:
        print("Stopped by user.")
        print("Best trial so far:")
        print(study.best_trial)
    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Accuracy={}".format(
            trial.values[0]))
        print("    Params: {}".format(trial.params))

    if failed_trials:
        print("\nFailed trials summary:")
        for trial_num, err in failed_trials:
            print(f"Trial {trial_num}: {err}")
