from typing import Dict, Type, Callable, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, v2, Lambda
from torchvision.transforms.functional import crop
from code.source.config.categories import DatasetMode
from code.source.config.images import ImageTrim


def load_datasets(
    dataset_class: Type[Dataset],
    dataset_mode: DatasetMode,
    transform: Optional[Callable],
    train: bool = True,
    validation: bool = False,
    test: bool = True,
    test_size: float = 1 / 3,
    only_bmode: bool = True
) -> Dict[str, Dataset]:
    """
    Loads datasets for training, validation, and testing from a custom dataset class.

    Parameters
    ----------
    dataset_class : Type[Dataset]
        Custom dataset class to instantiate.
    dataset_mode : DatasetMode
        Mode to define how the dataset is interpreted (e.g., binary, multiclass).
    transform : Callable or None
        Transformation function to apply to each image.
    train : bool, optional
        Whether to load the training dataset, by default True.
    validation : bool, optional
        Whether to split a portion of training data as validation, by default False.
    test : bool, optional
        Whether to load the test dataset, by default True.
    test_size : float, optional
        Fraction of the dataset to use as the test split, by default 1/3.
    only_bmode : bool, optional
        Whether to include only B-mode (grayscale) images, by default True.

    Returns
    -------
    dict of str, Dataset
        A dictionary with keys "train", "validation", and/or "test" containing dataset objects.
    """
    datasets = {}
    VALIDATION_PERCENTAGE = 1/3

    if train:
        full_train_dataset = dataset_class(
            dataset_mode=dataset_mode, split="train", img_transform=transform, test_size=test_size,only_bmode=only_bmode)

        if validation:
            train_size = len(full_train_dataset)
            validation_size = round(train_size*VALIDATION_PERCENTAGE)

            lengths = [train_size - validation_size, validation_size]
            generator = torch.Generator().manual_seed(1)

            train_dataset, validation_dataset = random_split(dataset=full_train_dataset,
                                                             lengths=lengths,
                                                             generator=generator  # Ensures reproducibility
                                                             )
            datasets["validation"] = validation_dataset
            datasets["train"] = train_dataset
        else:
            datasets["train"] = full_train_dataset
    if test:
    
        datasets["test"] = dataset_class(dataset_mode=dataset_mode, split="test", img_transform=transform, test_size=test_size,only_bmode=only_bmode)

    return datasets


def get_loaders(datasets: Dict[str, Dataset], batch_size: int) -> Dict[str, DataLoader]:
    """
    Wraps datasets into DataLoaders for batching and shuffling.

    Parameters
    ----------
    datasets : dict of str, Dataset
        Dictionary of datasets keyed by split name (e.g., "train", "val", "test").
    batch_size : int
        Batch size to use for all DataLoaders.

    Returns
    -------
    dict of str, DataLoader
        Dictionary of DataLoaders with the same keys as the input datasets.
    """
    loaders = {}
    for key, dataset in datasets.items():
        shuffle = key == "train"  # Only shuffle training data
        loaders[key] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4
        )

    return loaders


def load_data(
    dataset_class: Type[Dataset],
    dataset_mode: DatasetMode,
    transform: Optional[Callable],
    batch_size: int,
    train: bool = True,
    validation: bool = False,
    test: bool = True,
    test_size: float = 1/3,
    only_bmode: bool = True
) -> tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
    """
    Loads datasets and their corresponding DataLoaders based on configuration.

    Parameters
    ----------
    dataset_class : Type[Dataset]
        Dataset class to instantiate.
    dataset_mode : DatasetMode
        Classification mode used to preprocess labels (e.g., binary or multiclass).
    transform : Callable or None
        Transformations to apply to the images.
    batch_size : int
        Size of each batch in the DataLoader.
    train : bool, optional
        Whether to load the training dataset. Default is True.
    validation : bool, optional
        Whether to create a validation split. Default is False.
    test : bool, optional
        Whether to load the testing dataset. Default is True.
    test_size : float, optional
        Proportion of data used for testing. Default is 1/3.
    only_bmode : bool, optional
        If True, only use B-mode images. Default is True.

    Returns
    -------
    tuple of dict
        A tuple containing:
        - datasets : dict of str -> Dataset
            Dictionary mapping split names ('train', 'val', 'test') to Dataset instances.
        - loaders : dict of str -> DataLoader
            Dictionary mapping split names to corresponding DataLoaders.
    """
    datasets = load_datasets(dataset_class, dataset_mode, transform,
                             train=train, validation=validation, test=test, test_size=test_size, only_bmode=only_bmode)
    loaders = get_loaders(datasets, batch_size)

    return datasets, loaders


def img_norm_for_dataset(
    dataset_mode: DatasetMode,
    dataset_class:"LiverImg",
    grayscale: bool,
    batch_size: int = 32,
    max_batches: int|None = None
) -> tuple[list[float], list[float]]:
    """
    Build a small DataLoader over the specified dataset and compute its
    normalization statistics.

    Parameters
    ----------
    dataset_mode : DatasetMode
        Which subset of the LiverImg dataset to use.
    grayscale : bool
        Whether to apply grayscale conversion in the transforms.
    batch_size : int
        Batch size for the temporary loader.
    max_batches : int | None
        If provided, limits the number of batches processed.

    Returns
    -------
    tuple[list[float], list[float]]
        Tuple with the per-channel mean and standard deviation.
    """
    print("Calculating Normalization values for the dataset...")
    #num_classes = DatasetMode.num_classes(dataset_mode)
    num_classes = dataset_mode.num_classes
    transformation = build_transformations(grayscale) 
    dataset = dataset_class(dataset_mode = dataset_mode, img_transform=transformation,split="train", test_size=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size)
    return img_norm_from_loader(loader, max_batches=max_batches)


def img_norm_from_loader(loader: DataLoader, max_batches: int | None = None) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean and standard deviation from images in a DataLoader.

    Parameters
    ----------
    loader : DataLoader
        Must yield image tensors of shape (B, C, H, W).
    max_batches : int or None, optional
        Maximum number of batches to use for calculation.

    Returns
    -------
    tuple[list[float], list[float]]
        Tuple of (mean, std) per channel.
    """
    mean = 0.0
    sq_mean = 0.0
    total = 0

    for i, (images, *_) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.float()
        b, c, h, w = images.shape
        images = images.view(b, c, -1)

        mean += images.mean(dim=[0, 2])
        sq_mean += (images ** 2).mean(dim=[0, 2])
        total += 1

    if total == 0:
        raise ValueError("No valid data found in loader")

    mean /= total
    std = (sq_mean / total - mean ** 2).sqrt()
    
    return (mean.tolist(), std.tolist())


def crop_func(img:"torch.Tensor|PIL.Image")->"torch.Tensor":
    """
    Crops an input image tensor using fixed dimensions defined in `ImageTrim`.

    This function performs a top-left crop using the constants:
    - `ImageTrim.TOP`: vertical offset
    - `ImageTrim.LEFT`: horizontal offset
    - `ImageTrim.HEIGHT`: height of the crop
    - `ImageTrim.WIDTH`: width of the crop

    Parameters
    ----------
    img : torch.Tensor or PIL.Image
        The input image to crop. Should be in HWC or CHW format compatible with torchvision.

    Returns
    -------
    torch.Tensor
        The cropped image.
    """
    trim = ImageTrim()
    return crop(img, trim.TOP, trim.LEFT, trim.HEIGHT, trim.WIDTH)

def build_transformations(
    grayscale: bool,
    img_mean: list | tuple | None = None,
    img_std: list | tuple | None = None,
    resize: int | tuple | None = None,
    horizontal_flip: float | None = None
) -> Compose:
    """
    Builds a torchvision transformation pipeline based on input configuration.

    Parameters
    ----------
    grayscale : bool
        Whether to convert the image to grayscale.
    img_mean : list or tuple, optional
        Mean values for normalization. Should be of length 1 for grayscale, 3 for RGB.
    img_std : list or tuple, optional
        Standard deviation values for normalization. Same rules as `img_mean`.
    resize : int or tuple, optional
        Target size to resize the image to. Integer for square resize, or tuple (H, W).
    horizontal_flip : float, optional
        Probability of applying horizontal flip. Example: 0.5 for a 50% chance.

    Returns
    -------
    transforms.Compose
        A composed torchvision transformation pipeline.
    """
    transformations = [v2.ToImage()]

    if grayscale:
        transformations.append(v2.Grayscale())

    transformations.extend([
        Lambda(crop_func),
        # Normalize expects float input
        v2.ToDtype(torch.float32, scale=True)
    ])

    if horizontal_flip:
        transformations.append(v2.RandomHorizontalFlip(p=horizontal_flip))

    if resize:
        transformations.append(v2.Resize(size=resize))

    if img_mean and img_std:
        mean = [img_mean[0]] if grayscale else img_mean
        std = [img_std[0]] if grayscale else img_std
        transformations.append(v2.Normalize(mean=mean, std=std))

    return Compose(transformations)


def get_input_dimensions(loader: "DataLoader") -> tuple[int,int,int]:
    """
    Utility function to get input dimensions (in_H, in_W, in_C) from the first batch of a DataLoader.
    
    Parameters
    ----------
    loader : DataLoader
        A PyTorch DataLoader.

    Returns
    -------
    tuple
        A tuple of (width, height, channels).
    """

    data, _ = next(iter(loader))
    data_size = data.size()
    return (data_size[2], data_size[3], data_size[1])
