#!/usr/bin/env python3

import os
from pathlib import Path
import sys
from datetime import datetime


if __name__ == '__main__':
    # Add 'code' directory to import path for local module access
    sys.path.insert(0, os.path.abspath('./code'))

import pandas as pd
from code.source.config.paths import Dirs
from code.source.config.images import USMode


from source.utils.files import get_newest_file



def process_images(input_dir: str | Path) -> list[dict]:
    """
    Processes categorized ultrasound image directories and classifies each file
    as B-mode or Doppler based on its folder structure.

    Assumes the following folder structure under `input_dir`:

        input_dir/
        ├── category1/
        │   ├── bmode/
        │   │   ├── image1.jpg
        │   │   └── ...
        │   └── doppler/
        │       ├── image2.jpg
        │       └── ...
        └── category2/
            └── ...

    Parameters
    ----------
    input_dir : str | Path
        Root directory containing multiple category subfolders,
        each with 'bmode' and/or 'doppler' subdirectories.

    Returns
    -------
    list[dict]
        A list of dictionaries, each with:
            - 'directory': Path to image folder as a string (with trailing slash).
            - 'filename': Name of the image file.
            - 'bmode': Boolean indicating whether the image is B-mode (True) or Doppler (False).
    """
    
    input_path = Path(input_dir)
    image_list = []

    bmode = USMode.BMODE.name.lower()
    doppler = USMode.DOPPLER.name.lower()
    
    for category_dir in input_path.iterdir():
        #if not category_dir.is_dir():# Skip non-directories
        if category_dir.is_file():# Skip non-directories
            continue

        # Process both 'doppler' and 'bmode' directories
        for doppler_bmode in [doppler, bmode]:
            dir_path = category_dir / doppler_bmode
            if dir_path.is_dir():
                grayscale_flag = (doppler_bmode == bmode)
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        image_list.append({
                            "directory": str(dir_path) + "/",  # mantiene el formato original
                            "filename": file_path.name,
                            "bmode": grayscale_flag
                        })

    return image_list


def doppler_bmode_to_df(
    input_dir: str | Path | Dirs.Images = Dirs.Images.HURH,
    output_dir: str | Path | Dirs.Images = Dirs.Annotations.HURH
):
    """
    Generates a CSV that merges grayscale (B-mode) detection with existing
    post-classification metadata for HURH ultrasound images.

    This function:
    1. Walks through all categorized image folders.
    2. Determines whether each image is from B-mode or Doppler.
    3. Merges this info with previously saved classification metadata.
    4. Outputs a timestamped CSV to `output_dir`.

    Parameters
    ----------
    input_dir : str | Path | ImageDirs, optional
        Directory containing classified ultrasound image folders (default: ImageDirs.HURH).

    output_dir : str | Path | ImageDirs, optional
        Directory where the output CSV will be saved (default: Dirs.Annotations.HURH).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image data
    image_list = process_images(input_dir)

    if not image_list:
        print("No files to analyze. CSV won't be created.")
        return

    # Create DataFrame
    df_grayscale = pd.DataFrame(data=image_list)

    # Load latest post-manual classification file
    classified_dirs_structure = pd.read_csv(
        get_newest_file(Dirs.Annotations.POST_CLASSIFICATION))
    
    # Merge grayscale/color info with classification info
    df_hurh = pd.merge(left=classified_dirs_structure,
                            right=df_grayscale,
                            how="left", on="filename")
    
    df_hurh["img_path"] = df_hurh.directory+df_hurh.filename

    # Generate timestamped filename
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_file = output_dir / f"hurh_{now}.csv"
    # Save DataFrame
    df_hurh.to_csv(output_file, index=False)
    print(f"CSV created at: {output_file}")


if __name__ == '__main__':
    doppler_bmode_to_df()
