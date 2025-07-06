from pathlib import Path
import shutil

import numpy as np
from code.source.config.paths import Dirs
from code.source.utils.files import get_newest_file

import pandas as pd 
import cv2

from datetime import datetime

class DepthMeterPosition:
    normal: tuple[int,int] = (1212, 174)
    wide:   tuple[int,int] = (  55, 175)


def join_hurh_od_csv():
    """
    Joins the most recent CSV files from the HURH and OneDrive annotation directories,
    saves the concatenated result as a timestamped CSV in the final annotation directory.

    """
    hurh_csv_dir = Path(Dirs.Annotations.HURH)
    od_csv_dir = Path(Dirs.Annotations.ONEDRIVE)

    hurh_csv = get_newest_file(hurh_csv_dir)
    od_csv = get_newest_file(od_csv_dir)

    df_hurh = pd.read_csv(hurh_csv)
    df_od = pd.read_csv(od_csv)

    df = pd.concat([df_od, df_hurh])

    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    final_csv_dir = Path(Dirs.Annotations.FINAL)
    final_csv_dir.mkdir(parents=True, exist_ok=True)
    final_csv_file = f"final_{now}.csv"
    df.to_csv(final_csv_dir/final_csv_file, index=False)
    print(f"Combined CSV saved to {final_csv_file}")


def hide_body_shape(
    image: np.ndarray,
    x: int = 1051,
    y: int = 737,
    max_height: int = 120,
    max_width: int = 108,
    search_padding: int = 20,
    white_threshold: int = 160
) -> tuple[np.ndarray, bool]:
    """
    Detects and blackens a small rectangular white shape within a defined region of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR format) as a NumPy array.
    x : int, optional
        X-coordinate of the top-left corner of the search region. Default is 1051.
    y : int, optional
        Y-coordinate of the top-left corner of the search region. Default is 737.
    max_height : int, optional
        Maximum height of the target rectangle. Default is 120.
    max_width : int, optional
        Maximum width of the target rectangle. Default is 108.
    search_padding : int, optional
        Padding to extend the search region. Default is 20.
    white_threshold : int, optional
        Threshold to isolate white regions in grayscale (0–255). Default is 160.

    Returns
    -------
    tuple[np.ndarray, bool]
        A tuple containing:
        - The modified image with blackened rectangle(s).
        - A boolean indicating whether any modification was made.
    """
    modified = False

    # Define Region of Interest (ROI)
    roi = image[y - search_padding:y + max_height + search_padding,
                x - search_padding:x + max_width + search_padding]

    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # binary threshold to isolate the white border
    _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    CANDIDATES = [
            (31, 18),
            (13, 11),
            (7, 31),
            (31, 7),
            (82, 21),
            (12, 120),
            (82, 33),
            (7, 29)
        ]


    # Loop through contours and blacken the detected rectangle
    for contour in contours:
        x_roi, y_roi, w, h = cv2.boundingRect(contour)


        # Define a tolerance margin (padding)
        margin = 2

        # checks if (w, h) is within the range of a candidate
        def is_within_margin(candidate, target, margin):
            cw, ch = candidate
            tw, th = target
            return (cw - margin <= tw <= cw + margin) and (ch - margin <= th <= ch + margin)

        # Check each candidate with the margin
        if any(is_within_margin(candidate, (w, h), margin) for candidate in CANDIDATES):

            abs_x = x + x_roi - search_padding
            abs_y = y + y_roi - search_padding

            # Draw a black rectangle
            cv2.rectangle(image, (abs_x-margin, abs_y-margin),
                          (abs_x + w+margin, abs_y + h+margin), (0, 0, 0), -1)
            modified = True

    return image, modified


def hide_rectangle(
    image: np.ndarray,
    x: int = 42,
    y: int = 757,
    max_height: int = 34,
    max_width: int = 219,
    search_padding: int = 25
) -> tuple[np.ndarray, bool]:
    """
    Detects and blackens a bright rectangle in a defined region of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR format) as a NumPy array.
    x : int, optional
        X-coordinate of the top-left corner of the target area. Default is 42.
    y : int, optional
        Y-coordinate of the top-left corner of the target area. Default is 757.
    max_height : int, optional
        Maximum height of the search region. Default is 34.
    max_width : int, optional
        Maximum width of the search region. Default is 219.
    search_padding : int, optional
        Padding to extend the search region. Default is 25.

    Returns
    -------
    tuple[np.ndarray, bool]
        - The modified image.
        - A boolean indicating whether any rectangle was blackened.
    """
    modified = False
    # Define Region of Interest (ROI)
    roi = image[y - search_padding:y + max_height + search_padding,
                x - search_padding:x + max_width + search_padding]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    WHITE_THRESHOLD = 200
    # Apply binary threshold to isolate the white border
    _, binary = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and blacken the detected rectangle
    for contour in contours:
        x_roi, y_roi, w, h = cv2.boundingRect(contour)
        min_w = 150
        max_w = 240
        min_h = 20
        max_h = 90
        if min_w < w < max_w and min_h < h < max_h:

            abs_x = x + x_roi - search_padding
            abs_y = y + y_roi - search_padding

            # Draw a black rectangle
            cv2.rectangle(image, (abs_x, abs_y),
                          (abs_x + w, abs_y + h), (0, 0, 0), -1)
            modified = True
    return image, modified


def move_color_imgs(input_dir: str | Path, doppler_dir: str | Path, bmode_dir: str | Path) -> pd.DataFrame:
    """Processes a directory of ultrasound images, checks if each image is in doppler 
    or bmode, and moves them to separate directories accordingly.

    It uses the `is_grayscale` function to determine the image type (bmode/doppler)
    after optionally cropping with `get_depth_meter`.

    Parameters
    ----------
    input_dir : str or Path
        Path to the directory containing the input images.
    doppler_dir : str or Path
        Path to the directory where doppler images will be moved.
    bmode_dir : str or Path
        Path to the directory where bmode images will be moved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - 'filename': Path to the moved file.
        - 'mean_diff': Mean channel difference used to determine doppler/bmode.
        - 'bmode': True if bmode, False if doppler.
    """

    img_paths = [p for p in Path(input_dir).resolve().iterdir() if p.is_file()]

    # Create output directories if they do not exist
    doppler_dir.mkdir(parents=True, exist_ok=True)
    bmode_dir.mkdir(parents=True, exist_ok=True)
    img_diffs = []

    for img_file in img_paths:

        ori_im = cv2.imread(img_file)
        crop = get_depth_meter(
            ori_im, rect_position=DepthMeterPosition.normal)
        grayscale, mean_diff = is_grayscale(crop)

        if grayscale is None:
            crop = get_depth_meter(
                ori_im, rect_position=DepthMeterPosition.wide)
            grayscale, mean_diff = is_grayscale(crop)

        dest_dir = bmode_dir if grayscale else doppler_dir

        dest_file = Path(dest_dir)/img_file.name

        shutil.move(img_file, dest_file)

        img_diffs.append({
            "filename": dest_file,
            "mean_diff": mean_diff,
            "bmode": grayscale
        })

    return pd.DataFrame(img_diffs)


def get_depth_meter(image: cv2.typing.MatLike, rect_position=(1212, 174), rect_size=(18, 159)) -> cv2.typing.MatLike:
    """
    Extracts the region of an ultrasound image that contains the depth meter.

    Parameters
    ----------
    image : cv2.typing.MatLike
        Input ultrasound image (usually BGR).
    rect_position : tuple of int, optional
        (x, y) position of the top-left corner of the depth meter.
    rect_size : tuple of int, optional
        (width, height) size of the depth meter region.

    Returns
    -------
    cv2.typing.MatLike
        Cropped region containing the depth meter. Returns an empty array if region is invalid.
    """
   
    x_min = rect_position[0]
    x_max = x_min + rect_size[0]
    y_min = rect_position[1]
    y_max = y_min + rect_size[1]


    return image[y_min:y_max, x_min:x_max].copy()


def is_grayscale(
    image: cv2.typing.MatLike,
    diff_min: int = 100,
    unknown_threshold: int = 10
) -> tuple[bool | None, float]:
    """Determines whether an image is grayscale based on average channel differences.

    Parameters
    ----------
    image : cv2.typing.MatLike
        Input image in BGR format (as typically returned by OpenCV).
    diff_min : int, optional
        Threshold above which an image is considered definitely color. Defaults to 100.
    unknown_threshold : int, optional
        Threshold below which image is considered too uniform to decide. Returns None in that case. Defaults to 10.

    Returns
    -------
    tuple[bool | None, float]
        - True if the image is grayscale,
        - False if the image is color,
        - None if the result is inconclusive.
        - Also returns the mean channel difference.
    """

    # Compute BGR channel differences
    #    BG                      B                G
    diff_bg = np.abs(image[:, :, 0] - image[:, :, 1])
    #    RB                      B                R
    diff_rb = np.abs(image[:, :, 0] - image[:, :, 2])
    #    GR                      G                R
    diff_gr = np.abs(image[:, :, 1] - image[:, :, 2])

    mean_diff = np.mean([diff_gr, diff_rb, diff_bg])

    if mean_diff > diff_min:
        return False, mean_diff  # Definitely color
    elif mean_diff < unknown_threshold:
        return None, mean_diff   # Uncertain 
    else:
        return True, mean_diff   # Considered grayscale

