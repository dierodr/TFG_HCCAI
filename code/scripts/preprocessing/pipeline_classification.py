#!/usr/bin/env python3

"""
Image Classification Pipeline Runner.

This script runs a 7-step image preprocessing and annotation pipeline for 
ultrasound image classification.

Steps:
    0. Extracts file metadata from directory structure into a CSV.
    1. Loads manually classified categories and generates a structured CSV.
    2. Removes sensitive/irrelevant rectangular regions from the images.
    3. Splits images into 'doppler' and 'bmode' folders using grayscale detection.
    4. Appends grayscale flag (bmode/doppler) into the CSV metadata.
    5. Preprocesses OneDrive images using the same classification logic.
    6. Merges hospital and OneDrive metadata into a final dataset.


"""

from code.source.utils.preprocessing import join_hurh_od_csv
from code.scripts.preprocessing.classified_structure_to_df import classified_structure_to_df
from code.scripts.preprocessing.split_images_in_doppler_bmode import split_in_doppler_bmode
from code.scripts.preprocessing.doppler_bmode_to_df import doppler_bmode_to_df
from code.scripts.preprocessing.convert_to_csv import convert_to_csv
from code.scripts.preprocessing.hide_annotations import process_images_in_folder
from code.scripts.preprocessing.onedrive_preprocessing import onedrive_preprocessing

def run_pipeline() -> None:
    """Executes the full image preprocessing and metadata pipeline."""
    
    print(" --- Step 0 ---")
    print("Extracting folder metadata to CSV...")
    convert_to_csv()
    print()

    print(" --- Step 1 ---")
    print("Loading manual classifications...")
    classified_structure_to_df()
    print()

    print(" --- Step 2 ---")
    print("Removing visual annotations (hiding rectangles)...")
    process_images_in_folder()
    print()

    print(" --- Step 3 ---")
    print("Splitting images into 'bmode' and 'doppler' folders...")
    split_in_doppler_bmode()
    print()

    print(" --- Step 4 ---")
    print("Adding grayscale flags to metadata...")
    doppler_bmode_to_df()
    print()

    print(" --- Step 5 ---")
    print("Preprocessing OneDrive images...")
    onedrive_preprocessing()
    print()

    print(" --- Step 6 ---")
    print("Merging HURH and OneDrive metadata into final CSV...")
    join_hurh_od_csv()
    print()

    print("Pipeline complete.")


if __name__ == '__main__':
    run_pipeline()