import torch
from torch.nn import Module, Linear
import torch.optim as optim

from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet121, DenseNet121_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights
)
from pathlib import Path
import gc
from tqdm import tqdm


from code.source.MetricsManager import MetricsManager
from code.source.utils.data_loading import build_transformations
from code.source.utils.models_common import get_confidences, get_loss_function, get_model_name, get_preds, get_probs
from code.source.config.categories import  ModelNames
from code.source.config.images import ImageNormalization, ImageTrim

class PretrainedConfig:
    BATCH_SIZE = 20
    MAX_EPOCHS = 30#60
    MAX_PATIENCE = 10#20
    PATIENCE = max(int(round(MAX_EPOCHS/3)), MAX_PATIENCE)
    LEARNING_RATE = 0.0001
    IMG_SIZE = ImageTrim().resize(4.5)
    MODEL_FILE = 'best_model.pth'
    NORMALIZATION = ImageNormalization.IMAGENET_COLOR


class PretrainedModel(Module):
    """
    Wrapper class for fine-tuning pretrained CNN architectures for liver image classification.

    Inherits from nn.Module to allow integration with PyTorch workflows.
    Encapsulates model loading, training, evaluation, saving, and transformation handling.

    Parameters
    ----------
    model_name : ModelNames
        Enum value indicating which pretrained architecture to use.
    root_path : str|Path
        Path where the model its gonna be saved
    num_classes: int
        Number of output classes for the model 
    download_weights : bool
        wether to download weights for the model from PytorchHub. Not needed a model is going to be loaded right away 
    """

    def __init__(self, model_name: ModelNames, root_path:str|Path , num_classes:int, download_weights:bool):
        super().__init__()
        self.model_name = model_name
        self.root_path = root_path
        self.num_classes  = num_classes
        self.grayscale = False
        self.img_mean, self.img_std = PretrainedConfig.NORMALIZATION
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(download_weights)
        self.model_path = self.get_model_path()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass through the internal pretrained model.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        return self.model(x)

    def _build_model(self,download_weights:bool) -> Module:
        """
        Initializes the selected pretrained model and modifies the classification head.

        Returns
        -------
        nn.Module
            The modified pretrained model.

        Raises
        ------
        ValueError
            If the model_name is not supported.
        """
        self.output_size = 1 if self.num_classes == 2 else self.num_classes

        if self.model_name == ModelNames.EFFICIENT:
            self.input_size=PretrainedConfig.IMG_SIZE

            if download_weights:
                weights = EfficientNet_B0_Weights.DEFAULT
            else:
                weights=None
            model = efficientnet_b0(weights=weights)
            model.classifier[1] = Linear(model.classifier[1].in_features, self.output_size )
            self.gradcam_compatible = True


        elif self.model_name == ModelNames.CONV:
            self.input_size=PretrainedConfig.IMG_SIZE

            if download_weights:
                weights = ConvNeXt_Tiny_Weights.DEFAULT
            else:
                weights=None
            model = convnext_tiny(weights=weights)
            model.classifier[2] = Linear(model.classifier[2].in_features, self.output_size )
            self.gradcam_compatible = True


        elif self.model_name == ModelNames.DENSE:
            self.input_size=PretrainedConfig.IMG_SIZE

            if download_weights:
                weights = DenseNet121_Weights.DEFAULT
            else:
                weights=None
            model = densenet121(weights=weights)
            model.classifier = Linear(model.classifier.in_features, self.output_size )
            self.gradcam_compatible = True


        elif self.model_name == ModelNames.RESNET:
            self.input_size=PretrainedConfig.IMG_SIZE

            if download_weights:
                weights = ResNet18_Weights.DEFAULT
            else:
                weights=None
            model = resnet18(weights=weights)
            model.fc = Linear(model.fc.in_features, self.output_size )
            self.gradcam_compatible = True


        elif self.model_name == ModelNames.VIT:
            self.input_size=(224, 224)
            if download_weights:
                weights = ViT_B_16_Weights.DEFAULT
            else:
                weights=None
            model = vit_b_16(weights=weights)
            model.heads.head = Linear(model.heads.head.in_features, self.output_size )
            self.gradcam_compatible = False

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model.to(self.device)

    def get_transforms(self) -> Compose:
        """
        Constructs a torchvision transformation pipeline for training pretrained models.

        The pipeline performs the following steps:
        1. Converts input from a PIL image.
        2. Crops the image using `crop_func`.
        3. Resizes it to a fixed resolution:
        - (224, 224) for Vision Transformer (ViT)-based models.
        - `PretrainedConfig.IMG_SIZE` for all other models.
        4. Converts the image to float tensor in [0, 1] range.
        5. Normalizes it using ImageNet mean and standard deviation.

        Parameters
        ----------
        vision_trans : bool
            If True, uses ViT-compatible resize (224x224). Otherwise, uses `PretrainedConfig.IMG_SIZE`.

        Returns
        -------
        transforms.Compose
            A composed torchvision transformation for image preprocessing.
        """

    
    @property
    def get_last_conv_layer(self)->Module:
        """Returns the last convolutional layer of a supported model.

        Returns
        -------
        nn.Module
            The last convolutional layer in the model.

        Raises
        ------
        ValueError
            If the model type is not supported.
        """

        if self.model_name == ModelNames.EFFICIENT:
            return self.model.features[-1]  # Last Conv layer in EfficientNet_B0
        elif self.model_name == ModelNames.RESNET:
            return self.model.layer4[-1]  # Last Conv layer in ResNet18
        elif self.model_name == ModelNames.DENSE:
            return self.model.features[-1]  # Last feature layer in DenseNet121
        elif self.model_name == ModelNames.CONV:
            return self.model.features[-1][-1]  # Last ConvNeXt block
        else:
            raise ValueError("Grad-CAM not supported for this model type.")

    def train_loop(self,train_loader,epoch,optimizer,criterion,scaler)->float:
        self.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{PretrainedConfig.MAX_EPOCHS} [Train]"):
            images = images.to(self.device)
            targets = targets.to(self.device)
                
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = self(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            del images, targets, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        return running_loss

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader
    ) -> None:
        """
        Trains a PyTorch model with validation, saving the best-performing model based on validation accuracy.

        Uses mixed precision (torch.amp) for faster training on GPU and applies early stopping.

        Parameters
        ----------
        model : Module
            The PyTorch model to train.
        train_loader : DataLoader
            Dataloader for the training set.
        valid_loader : DataLoader
            Dataloader for the validation set.
        datasetmode : DatasetMode
            Enum value indicating the classification mode used.
        transform : Compose | Callable
            The image transformation pipeline used (to be saved with the model).
        """
        
        criterion = get_loss_function(use_weights=False, 
                                      num_classes=self.num_classes, 
                                      dataset=None, 
                                      device=self.device) 
        optimizer = optim.Adam(self.model.parameters(), lr=PretrainedConfig.LEARNING_RATE)
        scaler = torch.amp.GradScaler('cuda')

        best_acc = 0
        best_loss =  float('inf')
        patience_counter = 0
    
        for epoch in range(PretrainedConfig.MAX_EPOCHS):

            running_loss=self.train_loop(train_loader, epoch, optimizer, criterion, scaler)

            avg_loss = running_loss / len(train_loader)
            val_loss, val_acc = self.validate(valid_loader, criterion)
            print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_acc:.2f}%")


            #if val_acc > best_acc:
            if val_loss < best_loss:
                patience_counter = 0
                best_loss = val_loss
                self.save()
            else:
                patience_counter += 1
                print(f"No improvement {patience_counter}/{PretrainedConfig.PATIENCE}.")
            if patience_counter > PretrainedConfig.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    def validate(self, loader: DataLoader, criterion: _Loss) -> tuple[float, float]:
        """
        Evaluates the model on a given dataset loader and returns the average loss and accuracy.

        Parameters
        ----------
        model : Module
            PyTorch model to evaluate.
        loader : DataLoader
            Dataloader for the dataset to evaluate on.
        criterion : _Loss
            Loss function used to calculate the evaluation loss.
        num_classes : int
            Number of target classes in the classification task.

        Returns
        -------
        Tuple[float, float]
            A tuple containing:
            - Average loss across the dataset.
            - Accuracy as a percentage.
        """
        self.eval()
        running_loss = 0.0
        all_targets, all_scores = [], []

        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Validation", unit="batch"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                scores = self(images)

                loss = criterion(scores, targets)
                
                running_loss += loss.item()

                all_targets.append(targets)
                all_scores.append(scores)

        all_targets = torch.cat(all_targets)
        all_scores = torch.cat(all_scores)

        preds = get_preds(all_scores, self.num_classes)


        accuracy = (preds == all_targets).sum().item() / all_targets.size(0)

        avg_loss = running_loss / len(loader)
        return avg_loss, accuracy * 100

    def save(self, model_path: str | Path | None = None) -> None:
        """
        Saves the model state and preprocessing configuration to disk.

        This includes:
        - Model parameters (`state_dict`)
        - Whether images are grayscale
        - Image normalization statistics (mean and std)
        - Input image size

        Parameters
        ----------
        model_path : str or Path, optional
            Destination path for the saved checkpoint.
            If None, uses the default `self.final_path`.
        """
        if not model_path:
            model_path = self.final_path

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_params": self.model.state_dict(),
            "grayscale": self.grayscale,
            "img_mean": self.img_mean,
            "img_std": self.img_std,
            "input_size": self.input_size,
        }

        torch.save(checkpoint, model_path)
        self.final_path = model_path  # update stored path
        print(f"Model saved at: {self.final_path}")
    
    def load(self, model_path: str) -> tuple['Module', Compose]:
        """
        Loads a pretrained model checkpoint and its associated transformation.

        Parameters
        ----------
        model : Module
            PyTorch model into which the pretrained weights will be loaded.
        model_path : str
            Path to the saved checkpoint file (.pth) containing model weights and transform.

        Returns
        -------
        Tuple[Module, Compose]
            - The model with loaded weights.
            - The transformation used during training.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        transform = build_transformations(
            grayscale = checkpoint["grayscale"],
            img_mean = checkpoint["img_mean"],
            img_std = checkpoint["img_std"],
            resize = checkpoint["input_size"],
            horizontal_flip = None
        )
        self.model.load_state_dict(checkpoint["model_params"])
        self.final_path=model_path
        self.eval()
        return transform

    def evaluate(self, loader: DataLoader, save: bool) -> None:
        """
        Evaluates a pretrained model on a given dataset and prints key performance metrics.

        This function:
        - Loads pretrained weights and transforms.
        - Computes loss, accuracy, F1 score, and AUROC.
        - Optionally saves the model with performance-based naming.

        Parameters
        ----------
        model : Module
            PyTorch model to evaluate.

        loader : DataLoader
            DataLoader for evaluation dataset.

        datasetmode : DatasetMode
            Specifies the classification type (binary, multiclass).

        save : bool
            Whether to save the model with performance in the filename.
        """
        
        criterion = get_loss_function(use_weights=False, num_classes=self.num_classes, dataset=None, device=self.device)

        transformation = self.load(self.final_path)
        
        all_targets, all_scores = [], []
        running_loss = 0.0

        self.eval()
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Final Evaluation", unit="batch"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                scores = self(images)

                loss = criterion(scores, targets)
                running_loss += loss.item()
                
                all_targets.append(targets)
                all_scores.append(scores)

        all_targets = torch.cat(all_targets)
        all_scores = torch.cat(all_scores)
        avg_loss = running_loss / len(loader)

       
        metrics_to_use = ['accuracy_macro','accuracy_weighted','accuracy_per_class','f1','auroc','precision','recall','specificity']
        metrics_manager = MetricsManager(metrics_to_use, self.num_classes, self.device)

        probs = get_probs(all_scores, self.num_classes)
        preds = get_preds(all_scores, self.num_classes)

        metrics_manager.update(preds, all_targets, probs)

        
        results = metrics_manager.compute()
        metrics_manager.print_summary()
        metrics_manager.reset()       

        if save:
            avg_acc = results["accuracy_weighted"] if "accuracy_weighted" in results else results["accuracy_macro"]
            fname = get_model_name(accuracy=avg_acc, loss=avg_loss)
            model_path = self.get_model_path(file_name=fname)
            self.save(model_path)
        return results

    def get_model_path(self,file_name: str = None) -> Path:
        """
        Build a directory path for saving/loading the model.

        Parameters
        ----------
        file_name : str, optional
            Optional custom filename. Defaults to PretrainedConfig.MODEL_FILE.

        Returns
        -------
        Path
            Full path to the model file.
        """
        model_name = self.model.__class__.__name__
        model_path = self.root_path/"pretrained"/model_name
        model_path.mkdir(parents=True, exist_ok=True)
        if file_name: 
            self.final_path = model_path/file_name
        else:
            self.final_path = model_path/PretrainedConfig.MODEL_FILE

        return self.final_path

    def predict(self, img_tensor) -> tuple[int,float]:
        self.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            scores = self.model(img_tensor)
            preds = get_preds(scores, self.num_classes)
            probs = get_probs(scores, self.num_classes)
            confidences = get_confidences(probs, preds, self.num_classes)
        
            return preds.item(), confidences.item()