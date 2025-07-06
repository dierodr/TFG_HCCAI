

from datetime import datetime
from pathlib import Path
import pandas as pd
import sys
import os

if __name__ == '__main__':
    # Add 'code' directory to import path for local module access
    sys.path.insert(0, os.path.abspath('./code'))
from source.utils.files import get_newest_file


def fix():
    # DEfine Paths
    data_path = Path.cwd()/"data/"
    images_path = data_path/"images/HURH/"
    csv_classified = data_path/"csv/post_manual_classification/"
    csv_grayscale = data_path/"csv/hurh/"

    
    print("Images Path:", images_path)
    print("CSV Classified:", csv_classified)
    print("CSV Colored:", csv_grayscale)

    classified_list = []
    colors_list = []
    # For all directories
    for category in images_path.iterdir():
        print(f" f{category.name=}")

        # Check if we have more directories
        if category.is_dir():
            # Loop them
            for doppler_dir in category.iterdir():
                print(f" {doppler_dir.name=}")

                # if we have a file outside color o gray add it
                if doppler_dir.is_file():
                    classified_list.append((category.name, doppler_dir.name))
                else:  # is dir
                    # There are 2 folders that we need to taverse
                    for image_file in doppler_dir.iterdir():
                        print(f"    {image_file.name=}")

                        if image_file.is_file():
                            classified_list.append((category.name, image_file.name))
                            colors_list.append((category.name, image_file.name, doppler_dir,None ))

                        else:  # is dir
                            print("Not expected directory found:", image_file)
        else:
            print("Not expected file found:", category)

    df_categories = pd.DataFrame(data=classified_list, columns=["category", "filename"])
    df_colors = pd.DataFrame(data=colors_list, columns=["category", "filename","directory","bmode"])

    df_pre_classification = pd.read_csv(get_newest_file(Path.cwd()/"data/csv/pre_manual_classification"))

    df_categories = pd.merge(df_categories, df_pre_classification, on="filename")
    df_colors = pd.merge(df_colors, df_pre_classification[["job_datetime","subdirectory","eco_datetime","filename","image_datetime","image_id"]], on="filename")
  

    df_categories = df_categories[["category", "filename", "subdirectory","eco_datetime", "image_datetime", "image_id"]]
    df_colors = df_colors[["category", "filename", "subdirectory","eco_datetime", "image_datetime", "image_id","directory","bmode"]]
                        

    # Save DataFrame to CSV
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    df_categories.to_csv(csv_classified/f"classified_structure_{now}.csv",index=False)
    df_colors.to_csv(csv_grayscale/f"grayscale_{now}.csv",index=False)


if __name__ == "__main__":
    fix()

