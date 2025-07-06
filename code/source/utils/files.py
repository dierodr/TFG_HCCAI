
import os
from pathlib import Path
import re


def get_newest_file(path: str | Path) -> Path:
    """
    Returns the most recently created file in the specified directory.

    Parameters
    ----------
    path : str | Path
        Directory to search for files.

    Returns
    -------
    Path
        Path to the most recently created file.

    Raises
    ------
    ValueError
        If the directory contains no files.
    """
    files = [p for p in Path(path).iterdir() if p.is_file()]
    if not files:
        raise ValueError(f"No files found in directory: {path}")
    return max(files, key=lambda p: p.stat().st_ctime)


def find_best_model(parent_folder: str | Path) -> Path:
    """
    Searches for the best `.pth` model file based on the highest accuracy and lowest loss.

    A valid model file must follow the naming pattern: `m_aACC_lLOSS.pth`, 
    e.g., `m_a92.5_l0.123.pth`.

    Parameters
    ----------
    parent_folder : str | Path
        Root directory to search for `.pth` model files.

    Returns
    -------
    Path
        Path to the best model file found.

    Raises
    ------
    FileNotFoundError
        If no valid `.pth` model files are found in the directory tree.
    """
    print(parent_folder)
    parent_folder = Path(parent_folder)
    # Regex extracts the float value from filenames
    FILE_PATTERN = re.compile(r"m_a(\d+\.\d+)_l(\d+\.\d+)\.pth")

    print("\n--- Searching for the Smallest Model Across All Subfolders ---\n")

    best_file = None
    best_acc = 0.0
    best_loss = float('inf')
    # Recursively search through all files
    for root, _, files in os.walk(parent_folder):
        for filename in files:
            match = FILE_PATTERN.fullmatch(filename)
            if match:
                acc = float(match.group(1))
                loss = float(match.group(2))
                # Pick model with highest accuracy, then lowest loss
                if acc > best_acc or (acc == best_acc and loss < best_loss):
                    best_acc = acc
                    best_loss = loss
                    best_file = os.path.join(root, filename)

    if best_file:
        print(f"[RESULT] Best file: {best_file} (accuracy={best_acc:.2f}, loss={best_loss:.4f})")
    else:
        print("[INFO] No valid .pth files found.")
        raise FileNotFoundError(f"No valid .pth files found for {parent_folder}")
        

    print("\n--- Search Completed ---")

    return Path(best_file)