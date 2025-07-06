#!/usr/bin/env python3
"""
Continue training from a CustomCNN checkpoint.

Usage:
    python3 retrain_model.py --model-path PATH [--epochs N]

Arguments:
    --model-path PATH   Path to a .pth checkpoint (under data/model_state/...).
    --epochs N          Number of additional epochs to train (default = 0, just evaluates).
"""

import argparse
from pathlib import Path

import os
import sys

# Ensure code/ directory is in sys.path
sys.path.insert(0, os.path.abspath('./code'))

from code.source.config.categories import DatasetMode
from source.LiverImg import LiverImg
from source.CustomCNN import CustomCNN, build_transformations, get_loss_function, load_data



def parse_args():
    p = argparse.ArgumentParser(
        usage="""Usage:
    python3 run_training.py --model-path PATH [--epochs N]

Arguments:
    --model-path PATH   Path to a .pth checkpoint (under data/model_state/...).
    --epochs N          Number of additional epochs to train (default = 0, just evaluates).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the `.pth` checkpoint (under data/model_state/...)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Additional epochs to train (0 = no further training, just evaluate)",
    )
    return p.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"Model checkpoint not found: {model_path}")
        return

    model, start_epoch, last_loss, scheduler, optimizer, params = CustomCNN.load_model(model_path)
    
    device = model.send_to_device()

    num_classes = model.num_classes
    dataset_mode = DatasetMode.from_num_classes(num_classes)
    print(f"DatasetMode: {dataset_mode.name}")


    print(f"Resumed at epoch {start_epoch}, loss = {last_loss:.4f}, device = {device}")

   
    additional_epochs = args.epochs
    total_epochs = start_epoch + additional_epochs
    params["max_epochs"] = additional_epochs
    print(f"Will train for {additional_epochs} more epochs (up to epoch {total_epochs})")


    transform = build_transformations(
        grayscale=params["grayscale"],
        img_mean=params["img_mean"],
        img_std=params["img_std"],
        resize=params["resize"],
        horizontal_flip=params.get("horizontal_flip", 0.0),
    )

    
    test_as_validation = True
    validation = True   
    dataset, loaders = load_data(
        dataset_class=LiverImg,
        dataset_mode=dataset_mode,
        transform=transform,
        batch_size=params["batch_size"],
        validation=not test_as_validation and validation,
    )
    if test_as_validation:
        dataset["validation"] = dataset["test"]
        loaders["validation"] = loaders["test"]
   
    criterion = get_loss_function(
        use_weights=True,
        num_classes=num_classes,
        dataset=dataset["train"],
        device=device,
    )

    
    if additional_epochs > 0:
        print("\nStarting training...")
        training_losses, validation_losses, validation_accuracies = model.fit(
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparameters=params,
            validation=validation
        )
        
        print("Training complete.\n")
    else:
        print("\nSkipping training, moving to evaluation.\n")


    for split in ("train", "test"):
        loader: DataLoader = loaders[split]
        print(f"Evaluating on {split} set ({len(dataset[split])} samples)")
        model.evaluation_loop(loader, criterion)


if __name__ == "__main__":
    main()
