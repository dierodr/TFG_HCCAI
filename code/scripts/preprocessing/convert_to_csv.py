#!/usr/bin/env python3


import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Ensure code/ directory is in sys.path
sys.path.insert(0, os.path.abspath('./code'))
from code.source.config.paths import Dirs

def convert_to_csv(input_dir:str|Path|Dirs.Images = Dirs.Images.CONVERT, output_dir:str|Path|Dirs.Images=Dirs.Annotations.CONVERT ):
    """
    Traverse a nested image directory structure and convert its metadata into a structured CSV file.

    The directory is expected to follow this format:
        input_dir/
            JOB_YYYYMMDD.HHMMSS.MMM/
                YYYYMMDDHHMMSSmmm/
                    image_YYYYMMDDHHMMSSmmm.png

    This function extracts timestamps and identifiers from directory and file names,
    builds a DataFrame with structured metadata, and writes it to a CSV in `output_dir`.

    Parameters
    ----------
    input_dir : str | Path | ImageDirs, optional
        Root directory containing the nested image folder structure. Defaults to `ImageDirs.CONVERT`.

    output_dir : str | Path | ImageDirs, optional
        Destination directory to store the resulting CSV file. Defaults to `Dirs.Annotations.CONVERT`.

    Output CSV Columns
    ------------------
    - directory       : Job-level folder name (e.g., JOB_YYYYMMDD.HHMMSS.MMM)
    - job_datetime    : Parsed datetime from job folder
    - subdirectory    : Echo study folder (e.g., YYYYMMDDHHMMSSmmm)
    - eco_datetime    : Parsed datetime from echo study folder
    - filename        : Image filename
    - image_datetime  : Parsed datetime from filename
    - image_id        : Identifier extracted from filename prefix

    Notes
    -----
    The output CSV will be timestamped and saved as: `convert_YYYY_MM_DD_HH_MM_SS.csv`
    """



    input_dir=Path(input_dir)
    output_dir=Path(output_dir)


    print(f"Target directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get top-level directories
    directories = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"{len(directories)} directories found.")

    # Define DataFrame columns
    columns = ["directory", "subdirectory", "filename"]
    list_of_images = []

    # Traverse the target directory
    for folder in directories:
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                for file in subfolder.iterdir():
                    if file.is_file():
                        list_of_images.append((folder.name, subfolder.name, file.name))

    # Create DataFrame
    df = pd.DataFrame(list_of_images, columns=columns)

    # Define datetime formats
    eco_datetime_format = "%Y%m%d%H%M%S%f"
    job_datetime_format = "%Y%m%d.%H%M%S.%f"

    # Convert subdirectory and directory timestamps
    df["eco_datetime"] = pd.to_datetime(
        df["subdirectory"], format=eco_datetime_format, errors="coerce")
    df["job_datetime"] = pd.to_datetime(
        df["directory"].str[5:], format=job_datetime_format, errors="coerce")

    # Extract timestamp from filename and convert
    image_datetime = df["filename"].str.split("_").str[1].str[:-8]
    image_id = df["filename"].str.split("_").str[0]

    df["image_datetime"] = pd.to_datetime(
        image_datetime, format=eco_datetime_format, errors="coerce")
    df["image_id"] = image_id

    # Define new column order
    new_column_order = ["directory", "job_datetime", "subdirectory",
                        "eco_datetime", "filename", "image_datetime", "image_id"]

    # Save DataFrame to CSV
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_path = output_dir / f"convert_{now}.csv"
    print(f"Saving DataFrame to {output_path}")
    df[new_column_order].to_csv(output_path, index=False)


if __name__ == '__main__':
    convert_to_csv()
