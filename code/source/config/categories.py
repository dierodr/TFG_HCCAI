from enum import Enum

from code.source.config.paths import Dirs

class Categories(Enum):
    SPLEEN = "Bazo"
    GALLBLADDER = "Cálculos_y_pólipos_en_la_vesicula"
    CIRRHOSIS = "Higado_con_cirrosis"
    STEATOSIS = "Higado_con_esteatosis"
    HCC = "Higado_con_hepatocarcinoma"
    HEALTHY_LIVER = "Higado_sano"
    PANCREAS = "Pancreas_normal"
    KIDNEY = "Riñón"
    BENIGN_LIVER_INJURY = "Lesiones_hepaticas_benignas"


    @classmethod
    def all(cls) -> tuple["Categories", ...]:
        """All categories."""
        return (
            cls.SPLEEN, cls.GALLBLADDER, cls.CIRRHOSIS, cls.STEATOSIS,
            cls.HCC, cls.HEALTHY_LIVER, cls.PANCREAS, cls.KIDNEY, cls.BENIGN_LIVER_INJURY,
        )

    @classmethod
    def livers(cls) -> tuple["Categories", ...]:
        """Categories related to liver."""
        return (
            cls.CIRRHOSIS, cls.STEATOSIS, cls.HCC, cls.HEALTHY_LIVER, cls.BENIGN_LIVER_INJURY,
        )

    @classmethod
    def cirrhcc_livers(cls) -> tuple["Categories", ...]:
        """Liver categories with cirrhosis and HCC."""
        return (cls.HCC, cls.CIRRHOSIS)

    @classmethod
    def diseased_livers(cls) -> tuple["Categories", ...]:
        """Livers that are diseased."""
        return (cls.HCC, cls.CIRRHOSIS, cls.STEATOSIS, cls.BENIGN_LIVER_INJURY)

    @classmethod
    def cirrhotic_state(cls) -> tuple["Categories", ...]:
        """Categories for cirrhotic state classification."""
        return (cls.HEALTHY_LIVER, cls.CIRRHOSIS, cls.HCC)


class DatasetMode(Enum):
    ORGAN_CLASSIFICATION = 5
    HEALTHY_LIVERS_OR_NOT = 2
    CIRRHOTIC_STATE = 3

    @classmethod
    def from_num_classes(cls, n: int) -> "DatasetMode":
        """Reverse lookup: given a class count, return the corresponding mode."""
        for mode in cls:
            if mode.value == n:
                return mode
        raise ValueError(f"No DatasetMode with {n} classes.")

    @classmethod
    def all(cls) -> list["DatasetMode"]:
        """All defined modes."""
        return list(cls)

    @property
    def num_classes(self) -> int:
        """Number of classes for this mode (just the enum’s value)."""
        return self.value

    def categories(self) -> tuple[Categories, ...]:
        return {
            DatasetMode.ORGAN_CLASSIFICATION: Categories.all(),
            DatasetMode.HEALTHY_LIVERS_OR_NOT: Categories.livers(),
            DatasetMode.CIRRHOTIC_STATE: Categories.cirrhotic_state(),
        }[self]

    def targets_mapping(self) -> dict[Categories, int]:
        return {
            DatasetMode.ORGAN_CLASSIFICATION: {
                **{cat: 0 for cat in Categories.livers()},
                Categories.SPLEEN: 1,
                Categories.PANCREAS: 2,
                Categories.KIDNEY: 3,
                Categories.GALLBLADDER: 4,
            },
            DatasetMode.HEALTHY_LIVERS_OR_NOT: {
                Categories.HEALTHY_LIVER: 0,
                **{cat: 1 for cat in Categories.diseased_livers()},
            },
            DatasetMode.CIRRHOTIC_STATE: {
                Categories.HEALTHY_LIVER: 0,
                Categories.CIRRHOSIS: 1,
                Categories.HCC: 2,
            },
        }[self]

    def directory(self) -> Dirs.Modes:
        return {
            DatasetMode.ORGAN_CLASSIFICATION: Dirs.Modes.ORGAN_CLASSIFICATION,
            DatasetMode.HEALTHY_LIVERS_OR_NOT: Dirs.Modes.HEALTHY_LIVERS_OR_NOT,
            DatasetMode.CIRRHOTIC_STATE: Dirs.Modes.CIRRHOTIC_STATE,
        }[self]



class ModelNames(Enum):
    VIT = "VisionTransformer"
    CONV = "ConvNeXt"
    EFFICIENT = "EfficientNet"
    DENSE = "DenseNet"
    RESNET = "ResNet"
    MYCNN = "CustomCNN"

    @classmethod
    def pretrained(cls) -> tuple["ModelNames", ...]:
        """Pretrained models available."""
        return (cls.EFFICIENT, cls.CONV, cls.DENSE, cls.RESNET, cls.VIT)

    @classmethod
    def all(cls) -> tuple["ModelNames", ...]:
        """All supported models."""
        return (cls.MYCNN,) + cls.pretrained()
    
    @classmethod
    def from_value(cls, value: str) -> "ModelNames":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No ModelNames member with value '{value}'")


