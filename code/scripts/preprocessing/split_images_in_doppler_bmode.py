#!/usr/bin/env python3

import os
from pathlib import Path
import sys

# Ensure project root path is included when running standalone
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('./code'))

from source.utils.preprocessing import move_color_imgs
from code.source.config.paths import Dirs
from code.source.config.images import  USMode

def split_in_doppler_bmode(input_dir: str | Path = Dirs.Images.HURH):
    """
    Automatically separates ultrasound images into 'doppler' and 'bmode' subfolders
    based on grayscale detection.

    For each subdirectory inside the input folder, this function scans the images
    and moves them into a new 'bmode/' or 'doppler/' folder under the same subdirectory,
    depending on whether they look grayscale or color.

    Parameters
    ----------
    input_dir : str | Path, optional
        Root directory containing category subfolders with ultrasound images. 
        Each subfolder will be processed individually.
        Defaults to ImageDirs.HURH.

    Notes
    -----
    - Uses the `move_color_imgs` function to detect and relocate each image.
    - Creates new folders named `bmode/` and `doppler/` if they don't exist.
    - Assumes classification is based on whether the image is bmode or doppler.
    - If no valid images are found in a subfolder, a warning will be printed.

    Example
    -------
    Given a structure like:

        input_dir/
            category_1/
                image001.jpg
                image002.jpg
            category_2/
                image003.jpg

    After running this function:

        input_dir/
            category_1/
                bmode/
                    image001.jpg
                doppler/
                    image002.jpg
            category_2/
                bmode/
                    image003.jpg
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
        print(f"\tProcessing: {img_dir.name}")
        # Define target directories for split
        img_dir_doppler = img_dir / USMode.DOPPLER.name.lower()
        img_dir_bmode = img_dir / USMode.BMODE.name.lower()
        # Move images based on grayscale detection
        df = move_color_imgs(input_dir=img_dir,
                             doppler_dir=img_dir_doppler, bmode_dir=img_dir_bmode)
        # Warn if no images were processed
        if df is None or df.empty:
            print(f"\t   Warning: No images processed in {img_dir}")


if __name__ == "__main__":
    split_in_doppler_bmode()
