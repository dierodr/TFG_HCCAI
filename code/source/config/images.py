from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class ImageTrim:
    LEFT: int = 138
    RIGHT: int = 1142
    TOP: int = 171
    BOT: int = 832

    @property
    def WIDTH(self) -> int:
        return self.RIGHT - self.LEFT

    @property
    def HEIGHT(self) -> int:
        return self.BOT - self.TOP

    def r_width(self, n: int) -> int:
        return round(self.WIDTH / n)

    def r_height(self, n: int) -> int:
        return round(self.HEIGHT / n)

    def resize(self, n: int) -> tuple[int, int]:
        return (self.r_height(n), self.r_width(n))

@dataclass(frozen=True)
class ImageNormalization:
    """
    Holds per-channel mean and standard deviation computed over a dataset.
    ([mean],[std])
    """

    IMAGENET_GRAY =  ([0.0840], [0.1069])
    IMAGENET_COLOR = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    SIMPLE_GRAY =    ([0.5], [0.5])
    SIMPLE_COLOR =   ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


class USMode(Enum):
    """Class for storing Ultrasound modes.
    Currently contains only BMODE and DOPPLER, but there are other Ultrasound modes.
    Like: Amplitude -> AMODE"""
    BMODE = True
    DOPPLER = False