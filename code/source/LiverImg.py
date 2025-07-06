from typing import Literal
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose


from code.source.utils.files import get_newest_file
from code.source.config.paths import Dirs
from code.source.config.categories import DatasetMode



COLUMNS_IN_USE = ["category", "directory", "filename", "bmode", "img_path"]

class LiverImg(Dataset):
    """
    Custom PyTorch Dataset for liver image classification tasks.

    This dataset parses an annotation CSV and loads images accordingly,
    supporting multiple classification modes and automatic train/test splitting.

    Parameters
    ----------
    dataset_mode : DatasetMode
        Mode of classification (DatasetMode enum).
    img_transform : Compose
        Transformations to apply to the images.
    annotations_file : str, optional
        Path to the annotation CSV. If None, loads the latestfor the selected DatasetMode.
    target_transform : callable, optional
        Transformation to apply to target labels.
    split : Literal["train", "test", "full"], optional
        Dataset split to load: 'train' for training set, 'test' for test set,
        or 'full' to load the entire dataset without splitting. Default is 'train'.
    test_size : float, optional
        Proportion of the dataset to reserve for testing.
    only_bmode : bool, optional
        Whether to filter only B-mode (grayscale) images or allow Doppler ones.
    """
    
    def __init__(self,
                 dataset_mode: DatasetMode,
                 img_transform: Compose = Compose([]),
                 annotations_file: str | None  = None,
                 target_transform: Compose = None,
                 split: Literal["train","test","full"] = "train",
                 test_size:float = 1/3,
                 only_bmode:bool = True,):
        self.dataset_mode = dataset_mode
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.only_bmode = only_bmode
        self.test_size = test_size
        self.split = split
        # Load annotations
        if not annotations_file:
            annotations_file = get_newest_file(Dirs.Annotations.FINAL)
        
      

        #load full datset
        if split == "full":
            annotations = pd.read_csv(annotations_file, usecols=COLUMNS_IN_USE)
        else:
            split_col = f"{self.dataset_mode.name}_split"

            try:
                annotations = pd.read_csv(annotations_file, usecols=COLUMNS_IN_USE+[split_col])
            except ValueError:
                annotations = split_annotations_save(annotations_file, self.dataset_mode ,test_size)
                
            annotations = annotations[annotations[split_col] == split]    
        
        # Map categories to numerical labels
        annotations = self.__annotation_mapping__(annotations)
       

        # Use only grayscale images and required columns
        if only_bmode:
            self.annotations = annotations.loc[annotations.bmode, [
                    "img_path", "category"]]
        else:
            self.annotations = annotations[["img_path", "category"]]


        self.classes = self.annotations.category.unique()
        self.class_dist = self.annotations.category.value_counts()
 
    def __annotation_mapping__(self, annotations):
        """
        Maps raw category names from annotations to numerical class labels
        based on selected classification mode.

        Returns
        -------
        pd.DataFrame
            Annotations with categories converted to numeric labels.
        """
    
        categories = self.dataset_mode.categories()
        valid_category_names = [cat.value for cat in categories]

        mapping = self.dataset_mode.targets_mapping()
        string_mapping = {cat.value: idx for cat, idx in mapping.items()}

        annotations = annotations[annotations['category'].isin(valid_category_names)]
        
        with pd.option_context("future.no_silent_downcasting", True):
            annotations = annotations.replace({'category': string_mapping})
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Returns the transformed image and its label for the given index.

        Returns
        -------
        tuple
            (image_tensor, label_tensor)
        """
        img_path_column = 0
        label_column = 1
      
        img_path = self.annotations.iloc[idx, img_path_column] 
        image = read_image(img_path)

        label = self.annotations.iloc[idx, label_column]
        
        if len(self.classes) <= 2:
            label = torch.tensor(float(label), dtype=torch.float32).unsqueeze(0)
        else:
            label = torch.tensor(label)  
        
        if self.img_transform:
            image = self.img_transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    

def split_annotations_save(annotations_file:str,dataset_mode:DatasetMode,test_size:float):
     
    annotations = pd.read_csv(annotations_file)
    train_df, test_df = train_test_split(
        annotations,
        stratify=annotations["category"],
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )
    
    train_df[f"{dataset_mode.name}_split"] = "train"
    test_df[f"{dataset_mode.name}_split"] = "test"
    annotations = pd.concat([train_df, test_df], ignore_index=True)
    annotations.to_csv(annotations_file, index=False)
    return annotations