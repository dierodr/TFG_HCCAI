#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image
import shutil
import hashlib
from collections import defaultdict
import shutil
import sys
import pandas as pd
from datetime import datetime
import re

# Ensure project root path is included when running standalone
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('./code'))
from code.source.config.paths import Dirs
from code.source.config.images import USMode
from code.source.config.categories import Categories
from scripts.preprocessing.hide_annotations import process_images_in_folder
from source.utils.preprocessing import move_color_imgs


def copy_images_by_size(
    directory: str | Path,
    shapes: list[tuple[int, int]] = [(1280, 960)]
) -> None:
    """
    Copies images from a given directory into subfolders based on their resolution.

    Each image that matches one of the specified shapes will be copied into a subfolder 
    named "<width>x<height>" inside the original directory. Images that do not match 
    any target shape are skipped.

    Parameters
    ----------
    directory : str | Path
        Path to the folder containing image files.
    
    shapes : list of tuple[int, int], optional
        List of (height, width) pairs. Only images matching one of these shapes
        will be copied. Defaults to [(960, 1280)].
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    print(f"Copying images in {directory} with {shapes=}")
    print(directory)

    for file in directory.iterdir():
        if file.is_dir():
            continue

        img = Image.open(file)
        if img is None:
            print(f"[WARN] Skipping unreadable file: {file.name}")
            continue
     
        w, h = img.size
        if (w, h) in shapes:
            target_dir = directory / f"{w}x{h}"
            target_dir.mkdir(exist_ok=True)

            dest = target_dir / file.name
            if dest.exists():
                print(f"\tSkipping {file.name} — already exists.")
                continue
            shutil.copy(str(file), dest)
            print(f"\tCopied {file.name} to {target_dir.name}/")
    print("Done.\n")


def hash_file(file_path: Path) -> str:
    """Returns the hash of a given file

    Parameters
    ----------
    file_path : Path
        PAth to the file to hash

    Returns
    -------
    str
        Hash of the file
    """
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def merge_images_by_category(input_dir: str|Path, output_dir: str|Path) -> None:
    """ Combine images by category, avoiding duplicates, even with different names.

    Parameters
    ----------
    root_dir : Path
        Root directory with directories by category.
    output_dir : Path
        Destination directory for images organized without duplicates.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}


    print(f"Merging images in category: {input_dir}")

    # Take all images IN category and hashthem
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)

    for subfolder in input_dir.iterdir():
        
        if subfolder.is_dir():
            print(f"  Subfolder: {subfolder.relative_to(input_dir)}")
            for image_path in subfolder.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in allowed_exts:
                    img_hash = hash_file(image_path)
                    hash_to_paths[img_hash].append(image_path)

    
    #Copy one image per hash
    used_names = set()
    for img_hash, paths in hash_to_paths.items():
        # sort by name. first lower numbers
        paths.sort(key=lambda p: p.name)

        original_path = paths[0]
        try:
            patient_id, sub_id_ext = original_path.name.rsplit("-", 1)
            sub_id = sub_id_ext.rsplit(".", 1)[0]
            extension = original_path.suffix.lower()
        except ValueError:
            print(f"    Wrong name: {original_path.name}")
            continue

        base_num = int(sub_id) if sub_id.isdigit() else 0
        candidate_num = base_num

        while True:
            new_filename = f"{patient_id}-{candidate_num}{extension}"
            destination = output_dir / new_filename

            if new_filename not in used_names and not destination.exists():
                shutil.copy2(original_path, destination)
                used_names.add(new_filename)
                break
            else:
                if destination.exists():
                    existing_hash = hash_file(destination)
                    if existing_hash == img_hash:
                        break
                candidate_num += 1
    print("Done.\n")


def is_corrupted_pillow(image_path: Path) -> bool:
    """
    Verifies if an image is corrupted using Pillow.

    Parameters
    ----------
    image_path : Path
        Path to the imageimage.

    Returns
    -------
    bool
        True if image is corrupted, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Basic integrity check. But sometimes not enough.
        with Image.open(image_path) as img:
            img.transpose(Image.FLIP_LEFT_RIGHT)  # This works
    except Exception:
        return True
    return False


def detect_and_move_corrupted_images(
    output_dir: Path,
    corrupted_dir: Path
) -> None:
    """
    Detects corrupted images in output_dir and moves them to corrupted_dir.

    Parameters
    ----------
    output_dir: Path
    Folder where the images are organized.
    corrupted_dir: Path
    Destination folder for the corrupted images.
    """
    output_dir = Path(output_dir)
    corrupted_dir = Path(corrupted_dir)
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    print("Moving corrupted images...")
    for img_path in output_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in allowed_exts:
            continue

        if is_corrupted_pillow(img_path):
            rel_path = img_path.relative_to(output_dir)
            dest_path = corrupted_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), str(dest_path))
            print(f"\tCorrupted image moved to: {dest_path}")
    print("Done.\n")


def onedrive_preprocessing():
    """
    Loop through all the directroies in the Onedrive folder and preprocess the images that founds.
    Renaming the images.
    Copying to a directory with it size as name.
    Moves the doppler images to one folder and the bmodes to another.
    Generates a csv of the images.
    """
    directories = [
        Dirs.Images.ONEDRIVE/Categories.HCC.value,
        Dirs.Images.ONEDRIVE/Categories.CIRRHOSIS.value,
        Dirs.Images.ONEDRIVE/Categories.HEALTHY_LIVER.value
    ]
    shape = (1280, 960)
    aplio_size = f"{shape[0]}x{shape[1]}"
    dfs = []
    for dir in directories:
        dir = Path(dir)
        merge_images_by_category(dir, dir) 
        detect_and_move_corrupted_images(dir, dir/"corrupted")
        aplio_size_dir = dir/aplio_size
        aplio_size_dir.mkdir(exist_ok=True)
        copy_images_by_size(dir, shapes=[shape])
        process_images_in_folder(input_dir=aplio_size_dir)
        _ = move_color_imgs(input_dir=aplio_size_dir,
                            doppler_dir=aplio_size_dir / USMode.DOPPLER.name.lower(),
                            bmode_dir=aplio_size_dir/USMode.BMODE.name.lower()) 
        list_dic = livers_to_csv(aplio_size_dir)
        dfs += list_dic

    df = pd.DataFrame(dfs)
    if not df.empty:
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        annotations_file = Dirs.Annotations.ONEDRIVE/f"onedrive_{now}.csv"
        df.to_csv(annotations_file, index=False)
        print(f"CSV created in: {annotations_file}")
    else:
        print("Dataframe not saved because it was empty.")


def livers_to_csv(path:str|Path)->list[dict]:
    """Creates a csv of the images in the specified path

    Args:
        path (str|Path): Path to the directory where the images are.

    Raises:
        ValueError: ValueError

    Returns:
        _type_: _description_
    """
    path = Path(path)
    list_csv = []
    category = path.parents[0]
    print("Creating CSV...")
    for ultrasound_mode_dir in path.iterdir():
        if ultrasound_mode_dir.is_file():
            continue
       
        us_mode_name = ultrasound_mode_dir.name.upper()
        try:
            us_mode = USMode[us_mode_name]
        except KeyError:
            raise ValueError(f"Invalid ultrasound mode folder: {ultrasound_mode_dir.name}")

        bmode = us_mode.value
        for file in ultrasound_mode_dir.iterdir():
            list_csv.append({
                "category": category.name,
                "filename": file.name,
                "img_path": file,
                "directory": category,
                "image_id": file.stem.split("-")[-1],
                "bmode": bmode
            })
    return list_csv


if __name__ == "__main__":
    onedrive_preprocessing()
