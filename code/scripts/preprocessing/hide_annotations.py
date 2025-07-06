#!/usr/bin/env python3


from pathlib import Path
import cv2
import os
import sys

# Ensure code/ directory is in sys.path
sys.path.insert(0, os.path.abspath('./code'))
from code.source.config.paths import Dirs
from source.utils.preprocessing import hide_body_shape, hide_rectangle


def process_images_in_folder(
    input_dir: str | Path | Dirs.Images = Dirs.Images.HURH,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
) -> None:
    """
    Recursively processes images in the given folder, applying preprocessing steps like 
    `hide_rectangle` and `hide_body_shape`. Only overwrites images if modifications 
    were made.

    Parameters
    ----------
    input_dir : str | Path | ImageDirs, optional
        Root directory containing class-labeled subfolders with images.
        Defaults to ImageDirs.HURH.

    extensions : tuple[str, ...], optional
        Valid image extensions to include. Defaults to (".jpg", ".jpeg", ".png").

    Behavior
    --------
    - Iterates through all subdirectories inside `input_dir`.
    - Loads each image with a matching extension.
    - Applies `hide_body_shape` and `hide_rectangle`.
    - If any modification is detected, the image is overwritten.
    - Logs the number of modified images per category.

    Returns
    -------
    None
    """

    input_dir=Path(input_dir)
    print(f"Input directory: {input_dir}")

    if not input_dir.exists():
        print("Error: Input directory does not exist.")
        return
    
    # List all subdirectories that potentially contain images
    img_dirs = [p for p in Path(input_dir).iterdir() if p.is_dir()]

    if not img_dirs:
        print("No subdirectories found.")
        return
    
   
    # Process each directory
    for img_dir in img_dirs:
        category_processed = 0
        print(f"\tProcessing: {img_dir.name}",end=" -> ")
        for image_path in Path(img_dir).iterdir():
            if image_path.suffix.lower() in extensions:
                image = cv2.imread(image_path)
                _,mod_body_shape = hide_body_shape(image)
                _, mod_rectangle = hide_rectangle(image)
                if mod_rectangle or mod_body_shape:
                    category_processed += 1
                    cv2.imwrite(image_path, image)
        print(category_processed)
                


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hide_annotations.py <path_to_folder>")
    else:
        folder_path = sys.argv[1]
        print(f"Starting processing for folder: {folder_path}")
        process_images_in_folder(input_dir=folder_path)
        print("Processing completed.")
