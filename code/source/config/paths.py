
# Default image root
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path("./")
CSV_PATH = PROJECT_ROOT/"data/csv/trials.csv"


class Dirs:
    """Grouped dataset, modes and annoations directory constants."""
    @dataclass(frozen=True)
    class Modes:
        MAIN_DIR: Path = PROJECT_ROOT / "data/model_state/"
        CIRRHOTIC_STATE: Path = MAIN_DIR / "cirrhotic_state"
        HEALTHY_LIVERS_OR_NOT: Path = MAIN_DIR / "healthy_livers_or_not"
        ORGAN_CLASSIFICATION: Path = MAIN_DIR / "organ_classification"

    @dataclass(frozen=True)
    class Annotations:
        MAIN_DIR: Path = PROJECT_ROOT / "data/csv/"
        HURH: Path = MAIN_DIR / "hurh/"
        ONEDRIVE: Path = MAIN_DIR / "onedrive/"
        FINAL: Path = MAIN_DIR / "final/"
        CONVERT: Path = MAIN_DIR / "pre_manual_classification"
        POST_CLASSIFICATION: Path = MAIN_DIR / "post_manual_classification"

    @dataclass(frozen=True)
    class Images:
        MAIN_DIR: Path = PROJECT_ROOT / "data/images/"
        CONVERT: Path = MAIN_DIR / "Convert"
        HURH: Path = MAIN_DIR / "HURH"
        ONEDRIVE: Path = MAIN_DIR / "OneDrive"

