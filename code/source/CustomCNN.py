import os
from pathlib import Path
from pprint import pprint



import pandas as pd
import torch
from torch.nn import Module, Sequential, AdaptiveAvgPool2d, Conv2d, MaxPool2d, Dropout, ReLU, Linear, BatchNorm2d
from code.source.config.images import ImageTrim

from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR, CosineAnnealingLR

from datetime import datetime

from code.source.utils.plotting import plot_evolution
from code.source.MetricsManager import MetricsManager
from code.source.utils.data_loading import (build_transformations, get_input_dimensions, load_data)
from code.source.utils.models_common import (get_confidences, get_loss_function, get_model_name, get_preds, get_probs, register_in_logs, step_scheduler)
from code.source.config.categories import DatasetMode, ModelNames
from code.source.config.paths import CSV_PATH


date_now = datetime.now()

class CustomCNN(Module):
    """
    Custom Convolutional Neural Network for liver image classification.

    This architecture includes:
    - Convolutional blocks with optional Batch Normalization
    - Fully connected hidden layers with optional Dropout
    - An output layer that adapts based on binary/multiclass classification
    - Utilities for GradCAM attributions
    - Model saving/loading capabilities

    Parameters
    ----------
    input_dimensions : tuple of int
        Shape of the input image as (height, width, channels).
    num_classes : int
        Number of output classes. If 2, output layer will have one neuron.
    directory : str or Path
        Path where model checkpoints and related files will be saved.
    cv_channels : list of int, optional
        Output channels for each convolution layer.
    cv_kernels : list of int, optional
        Kernel size(s) for convolutional layers. If one value is provided, it's shared across layers.
    cv_strides : list of int, optional
        Stride(s) for convolutional layers. If one value is provided, it's shared across layers.
    cv_paddings : list of int, optional
        Padding(s) for convolutional layers. If one value is provided, it's shared across layers.
    pool_kernels : list of int, optional
        Kernel sizes for pooling layers.
    pool_strides : list of int, optional
        Strides for pooling layers.
    pool_paddings : list of int, optional
        Paddings for pooling layers.
    batch_norm : bool, optional
        Whether to apply Batch Normalization after convolution layers.
    gap_outputs : tuple of int, optional
        Output size (H, W) for Global Average Pooling (GAP). If not provided, spatial features are flattened.
    hidden_layers : list of int, optional
        Sizes of the fully connected hidden layers.
    fcl_dropouts : list of float, optional
        Dropout rates for each fully connected layer. If one value is provided, it is reused.

    Attributes
    ----------
    model_directory : Path
        Path where model checkpoints will be saved.
    last_save_path : Path or None
        Path to the most recent model checkpoint, if any.
    conv_layers : nn.Sequential
        Convolutional feature extractor.
    gap : nn.Module or None
        Adaptive average pooling layer, if enabled.
    fc_layers : nn.Sequential
        Fully connected classifier layers.
    """
    
    def __init__(
            self,
            input_dimensions: tuple[int, int, int],
            num_classes: int,
            directory: str|Path,
            cv_channels: list[int] = [4, 8, 16, 32, 64],
            cv_kernels: list[int] = [3],
            cv_strides: list[int] = [1],
            cv_paddings: list[int] = [0],
            pool_kernels: list[int] = [2],
            pool_strides: list[int] = [2],
            pool_paddings: list[int] = [0],
            batch_norm: bool = False,
            gap_outputs=(),
            hidden_layers: list[int] = [64, 32],
            fcl_dropouts: list[float] = [0.5]
    ):
        super(CustomCNN, self).__init__()

        # INPUTS
        self.input_dimensions = input_dimensions
        in_H, in_W, in_C = input_dimensions

        # OUTPUTS
        self.num_classes = num_classes
        self.out_features = num_classes - 1 if num_classes == 2 else num_classes


        # LAYERS
        self.batch_norm = batch_norm

        self.cv_channels = cv_channels

        self.num_cv = len(cv_channels)
        self.all_channels = [in_C] + cv_channels

        self.cv_kernels = cv_kernels if len(
            cv_kernels) > 1 else cv_kernels * self.num_cv
        self.cv_strides = cv_strides if len(
            cv_strides) > 1 else cv_strides * self.num_cv
        self.cv_paddings = cv_paddings if len(
            cv_paddings) > 1 else cv_paddings * self.num_cv

        self.pool_kernels = pool_kernels if len(
            pool_kernels) > 1 else pool_kernels * self.num_cv
        self.pool_strides = pool_strides if len(
            pool_strides) > 1 else pool_strides * self.num_cv
        self.pool_paddings = pool_paddings if len(
            pool_paddings) > 1 else pool_paddings * self.num_cv
        
        self.gradcam_compatible=True
        self.gap_outputs = gap_outputs

        # PATHING STUFF
        self.model_directory = Path(directory)/f"cv_{str(cv_channels)[1:-1].replace(', ', '_')}_fcl_{str(hidden_layers)[1:-1].replace(', ', '_')}/"
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self.last_save_path = None 
        

        # ----------------------
        # CONVOLUTION LAYERS
        #       ---
        # CONV -> (BATCH) -> RELU -> POOL
        # ----------------------
        conv_layers = []
        for cv_index in range(self.num_cv):
            conv_layers.append(self._make_conv(cv_index))
        self.conv_layers = Sequential(*conv_layers)

        # ----
        # GAP
        # -----
        if self.gap_outputs:
            
            self.gap = AdaptiveAvgPool2d(gap_outputs)
            features_fcl = cv_channels[-1]*gap_outputs[0]*gap_outputs[1]
        else:
            # Flat layer inputs computing
            features_fcl = self._get_conv_output()

        # --------------------------
        # Fully Connected Layers
        # --------------------------

        self.hidden_layers = hidden_layers
        self.num_hidden_fcl = len(hidden_layers)

        if fcl_dropouts:

            if len(fcl_dropouts) == 1:
                # If just one element this is the dropout for all layers
                fcl_dropouts = fcl_dropouts * (self.num_hidden_fcl-1)

            # to know in what layers do we use dropout or not we map the values to True or False
            self.use_dropout = list(
                map(lambda x: True if x else False, fcl_dropouts))

        else:  # EMPTY LIST
            # if the list was empty we are not using dropout
            self.use_dropout = [False] * hidden_layers
        self.fcl_dropouts = fcl_dropouts

        self.fcl = [features_fcl] + hidden_layers + [self.out_features]
        self.num_fcl = len(self.fcl)

        # Ensure use_dropout has same length as number of layer *transitions*
        while len(self.use_dropout) < self.num_fcl - 1:
            self.use_dropout.append(False)

        fc_layers = []
        for fcl_index in range(1, self.num_fcl):
            fc_layers.append(self._make_fcl(fcl_index))
        self.fc_layers = Sequential(*fc_layers)

    def _make_conv(self, cv_index: int) -> Sequential:
        """
        Builds a single convolutional block composed of Conv2d, optional BatchNorm2d, ReLU, and MaxPool2d.

        Parameters
        ----------
        cv_index : int
            Index of the convolutional layer in the sequence. Used to extract
            the corresponding configuration (channels, kernel, stride, padding, etc.).

        Returns
        -------
        Sequential
            A PyTorch Sequential block containing the convolutional block.
        """
        cv_block = []
        # CONV -> (BATCH) -> RELU -> POOL
        cv_block.append(
            Conv2d(
                in_channels=self.all_channels[cv_index],
                out_channels=self.all_channels[cv_index+1],
                kernel_size=self.cv_kernels[cv_index],
                stride=self.cv_strides[cv_index],
                padding=self.cv_paddings[cv_index]
            )
        )

        if self.batch_norm:
            cv_block.append(BatchNorm2d(self.all_channels[cv_index+1]))

        cv_block.append(ReLU(inplace=True))

        cv_block.append(
            MaxPool2d(
                kernel_size=self.pool_kernels[cv_index],
                stride=self.pool_strides[cv_index],
                padding=self.pool_paddings[cv_index]
            )
        )
        return Sequential(*cv_block)

    def _make_fcl(self, fcl_index: int) -> Sequential:
        """
        Builds a fully connected (linear) block for the network.

        Each block consists of a Linear layer followed optionally by Dropout,
        depending on the configuration provided during initialization.

        Parameters
        ----------
        fcl_index : int
            Index of the fully connected layer in the list of layer sizes.
            This defines the input and output dimensions of the Linear layer.

        Returns
        -------
        Sequential
            A PyTorch Sequential block containing the Linear layer and
            optionally a Dropout layer.
        """

        fcl_block = []
        fcl_block.append(
            Linear(in_features=self.fcl[fcl_index-1],
                      out_features=self.fcl[fcl_index])
        )

        if self.use_dropout[fcl_index-1]:
            fcl_block.append(Dropout(self.fcl_dropouts[fcl_index-1]))
        return Sequential(*fcl_block)

    def _get_conv_output(self) -> int:
        """
        Computes the flattened size of the output from the convolutional layers.

        This method is useful for dynamically determining the input size
        of the first fully connected layer after the convolutional part
        of the network, especially when the input size or layer configuration
        may change.

        Returns
        -------
        int
            Number of features (flattened units) output by the last convolutional block.
        """
        shape = self.input_dimensions
        x = torch.zeros(1, shape[2], shape[0], shape[1])
        with torch.no_grad():
            x = self.conv_layers(x)
        return int(torch.prod(torch.tensor(x.shape[1:])))

    @property
    def device(self) -> str:
        """
        Returns the name of the device where the model's parameters are located.

        Returns
        -------
        str
            Name of the device (e.g., 'cpu' or 'cuda:0').
        """
        return str(next(self.parameters()).device)

    @property
    def get_last_conv_layer(self)->Module:
        """    Returns the last convolutional layer of a supported model.
        Returns
        -------
        nn.Module
            The last convolutional layer in the model.
        """
        return self.conv_layers[-1][0]

    @classmethod
    def load_model(cls, path_to_model: str | Path) -> tuple['CustomCNN', int, float, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, dict]:
        """Loads a `CustomCNN` model along with its optimizer and scheduler from a checkpoint.

        Parameters
        ----------
        path_to_model : str or Path
            Path to the saved model checkpoint file.

        Returns
        -------
        tuple
            Tuple ontaining:
                - model (CustomCNN): Loaded model instance
                - epoch (int): Last completed epoch
                - loss (float): Loss at that epoch
                - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
                - optimizer (torch.optim.Optimizer): Optimizer
                - params (dict): Dictionary of training hyperparameters
        """
        
        print(f"Loading from checkpoint:\n\t{path_to_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path_to_model, map_location=torch.device(device),weights_only=False)

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']     
       
        model = cls(
            input_dimensions=checkpoint["input_dimensions"],
            num_classes=checkpoint["num_classes"],
            cv_channels=checkpoint["cv_channels"],
            cv_kernels=checkpoint["cv_kernels"],
            cv_strides=checkpoint["cv_strides"],
            cv_paddings=checkpoint["cv_paddings"],
            pool_kernels=checkpoint["pool_kernels"],
            pool_strides=checkpoint["pool_strides"],
            pool_paddings=checkpoint["pool_paddings"],
            batch_norm=checkpoint["batch_norm"],
            hidden_layers=checkpoint["hidden_layers"],
            fcl_dropouts=checkpoint['fcl_dropouts'],
            directory=Path(path_to_model).parent.parent
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.send_to_device()

        # Optimizer
        optimizer_name = checkpoint["optimizer_name"]
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters())
        optimizer.load_state_dict(
            state_dict=checkpoint['optimizer_state_dict'])

        # Scheduler
        scheduler_name = checkpoint["scheduler_name"]
        if scheduler_name == "StepLR":
            step_size = checkpoint["scheduler_step_size"]
            step_gamma = checkpoint["scheduler_step_gamma"]
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
                optimizer=optimizer, step_size=step_size, gamma=step_gamma)
        elif scheduler_name == "CosineAnnealingLR":
            t_max = checkpoint["scheduler_T_max"]
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
                optimizer=optimizer, t_max=t_max)
        else:
            scheduler = getattr(torch.optim.lr_scheduler,
                                scheduler_name)(optimizer=optimizer)
        scheduler.load_state_dict(
            state_dict=checkpoint['scheduler_state_dict'])

        keys = ["grayscale", "img_mean", "img_std", "resize", "batch_size", "test_size", "only_bmode","input_dimensions","num_classes","max_epochs","patience"]
        params = {k: checkpoint[k] for k in keys}
        return model, epoch, loss, scheduler, optimizer, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN.

        Applies convolutional layers (including optional BatchNorm and Pooling),
        flattens the result, and then passes it through fully connected layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W), where B is batch size.

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, num_classes) if multiclass,
            or (B, 1) if binary classification.
        """

        # CONV -> BATCH -> RELU -> POOL
        x = self.conv_layers(x)

        if self.gap_outputs:
            x = self.gap(x)
        else:
        # Flatten and pass through fully connected layers
            x = torch.flatten(x, 1)

        # Full Connected Layers
        output = self.fc_layers(x)

        return output

    def send_to_device(self, force_cpu: bool = False) -> str:
        """
        Moves the model to the appropriate computation device.

        Sends the model to GPU if available, unless `force_cpu` is set to True.

        Parameters
        ----------
        force_cpu : bool, optional
            If True, forces the model to be sent to CPU even if CUDA is available.

        Returns
        -------
        str
            Name of the device the model was moved to.
        """

        device = "cpu"
        if force_cpu:
            self.to(device)
        elif torch.cuda.is_available():
            device = "cuda:0"
        self.to(device)
        return device

    def save_checkpoint(
        self,
        epoch: int,
        optimizer: "Optimizer",
        loss: float,
        accuracy: float,
        scheduler: torch.optim.lr_scheduler,
        hyperparameters: dict
    ) -> str:
        """
        Saves a checkpoint of the model including weights, optimizer, scheduler, and hyperparameters.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        optimizer : torch.optim.Optimizer
            Optimizer used during training.
        loss : float
            Loss value at the current epoch.
        accuracy : float
            Accuracy at the current epoch.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler used during training.
        hyperparameters : dict
            Dictionary containing training hyperparameters such as normalization stats,
            resizing, batch size, etc.

        Returns
        -------
        str
            Path where the checkpoint was saved.
        """

        epoch = epoch+1
        save_info = {
            "grayscale":hyperparameters["grayscale"], 
            "img_mean":hyperparameters["img_mean"], 
            "img_std":hyperparameters["img_std"], 
            "resize":hyperparameters["resize"], 
            "batch_size":hyperparameters["batch_size"], 
            "test_size":hyperparameters["test_size"], 
            "only_bmode":hyperparameters["only_bmode"],
            "patience":hyperparameters["patience"],
            "max_epochs":hyperparameters["max_epochs"],
            "epoch": epoch,
            "loss": loss,
            "input_dimensions": self.input_dimensions,
            "num_classes": self.num_classes,
            'batch_norm': self.batch_norm,
            "cv_channels": self.cv_channels,
            "cv_kernels": self.cv_kernels,
            "cv_strides": self.cv_strides,
            "cv_paddings": self.cv_paddings,
            "pool_kernels": self.pool_kernels,
            "pool_strides": self.pool_strides,
            "pool_paddings": self.pool_paddings,
            "hidden_layers": self.hidden_layers,
            'fcl_dropouts': self.fcl_dropouts,
            "model_state_dict": self.state_dict(),
            "optimizer_name": type(optimizer).__name__,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_name": type(scheduler).__name__,
            "scheduler_state_dict": scheduler.state_dict(),
        }


        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            save_info["scheduler_step_size"] = scheduler.step_size
            save_info["scheduler_step_gamma"] = scheduler.gamma
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            save_info["scheduler_T_max"] = scheduler.T_max

        

        save_path = self.model_directory / get_model_name(accuracy, loss)

        if self.last_save_path:
            os.remove(self.last_save_path)
            print(f"Deleted old model_state: {self.last_save_path}")
            
        
        self.last_save_path = save_path
        

        print(f"Saving model in {save_path}.")
        torch.save(save_info, save_path)
        return save_path

    def load_checkpoint(self, path_to_model: str) -> Module:
        """
        Loads model weights from a saved checkpoint and sets the model to evaluation mode.

        Parameters
        ----------
        path_to_model : str
            Path to the saved model checkpoint file.

        Returns
        -------
        Module
            The model instance with loaded weights.
        """

        print(f"Loading from checkpoint:\n\t{path_to_model}")
        checkpoint = torch.load(path_to_model, weights_only=False, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.eval()
        return self

    def report_csv(
        self,
        epoch: int,
        hyperparameters: dict,
        optimizer: torch.optim,
        scheduler: _LRScheduler,
        metrics: dict
    ) -> None:
        """
        Logs training metrics and model configuration to a CSV file.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        hyperparameters : dict
            Dictionary containing training hyperparameters (e.g., batch size, resize).
        optimizer : torch.optim.Optimizer
            Optimizer used during training.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler used during training.
        metrics : dict
            Dictionary with metric names and values (e.g., accuracy, loss).
        """
        hp_keys = list(hyperparameters.keys())
        hp_values = list(hyperparameters.values())

        metrics_keys = list(metrics.keys())
        metrics_values = list(metrics.values())

        COLUMNS = ["datetime", "epoch", "optimizer", "scheduler",
                   "input_dim", "num_classes", "model_name"] + metrics_keys + hp_keys
        
        new_row = [date_now, epoch,  str(optimizer), str(scheduler),
                   self.input_dimensions, self.num_classes, ModelNames.MYCNN] + metrics_values + hp_values

        new_df_row = pd.DataFrame(data=[new_row], columns=COLUMNS)

        if Path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.DataFrame(columns=COLUMNS)

        # column num_classes can duplicate if model is loaded and the saved again
        # but value is the same and correct
        if new_df_row.columns.duplicated().any():
            new_df_row = new_df_row.loc[:, ~new_df_row.columns.duplicated()]

        df = pd.concat([df, new_df_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        
    def fit(
            self, 
            loaders:dict[str, "DataLoader"], 
            criterion:"_Loss", 
            optimizer:torch.optim, 
            scheduler:torch.optim, 
            hyperparameters:dict, 
            validation:bool=False, 
            trial=None)->tuple[list[float], list[float], list[float]]:
        """
        Trains the model using the training set, with optional validation and early stopping.

        Parameters
        ----------
        loaders : dict[str, DataLoader]
            Dictionary containing 'train' and optionally 'validation' DataLoader(s).
        criterion : torch.nn._Loss
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.
        hyperparameters : dict
            Training hyperparameters, must include keys like 'max_epochs', 'patience', etc.
        validation : bool, optional
            Whether to evaluate on a validation set, by default False.
        trial : optuna.trial.Trial, optional
            Optuna trial for pruning support, by default None.

        Returns
        -------
        tuple[list[float], list[float], list[float]]
            Lists of training losses, validation losses, and validation accuracies.
        """
     

        max_epochs = hyperparameters["max_epochs"]
        start_epoch = 0
        patience = hyperparameters["patience"]
        counter = 0

        min_delta = 0.001

        best_loss = float('inf')
        best_acc = 0
        current_validation_loss = 0
        current_training_loss = 0
        training_losses = []
        validation_losses = []
        validation_accuracies = []

        save_path = ""

        
        try:
            from torch.utils.tensorboard import SummaryWriter
            from torchvision.utils import make_grid

            writer = SummaryWriter(log_dir=f'logs/{str(self.model_directory).replace("/", "_")}')
        except:
            pass

        # loop over the dataset multiple times
        for epoch in range(start_epoch, max_epochs):

            if writer:
                images, _ = next(iter(loaders["train"]))
                grid = make_grid(images)
                writer.add_image('images', grid, epoch)

            print(f"\n------------------- Epoch {epoch+1} -------------------\n")

            # TRAINING STEP
            current_training_loss = self.train_loop(
                loaders["train"], criterion, optimizer)
            training_losses.append(current_training_loss)

            if writer:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("learning_rate", current_lr, epoch)

            print(f"[Epoch: {epoch + 1}/{max_epochs}] Training Loss: {current_training_loss:.4f}")

            # Training with Validation
            if validation:

                # VALIDATION STEP
                metrics = self.evaluation_loop(loaders["validation"], criterion)
                pprint(metrics)
                current_validation_accuracy = metrics["accuracy_weighted"]
                validation_accuracies.append(current_validation_accuracy)
                current_validation_loss = metrics["avg_loss"]

                self.report_csv(epoch, hyperparameters,
                                optimizer, scheduler, metrics)

                # Scheduler update
                step_scheduler(scheduler, metric=current_validation_loss)

                validation_losses.append(current_validation_loss)

                if (best_acc - current_validation_accuracy) < min_delta:
                    best_acc = current_validation_accuracy
                # if (best_loss - current_validation_loss) > min_delta:
                #     best_loss = current_validation_loss
                    counter = 0
                    save_path = self.save_checkpoint(epoch, optimizer, current_validation_loss, current_validation_accuracy, scheduler, hyperparameters)

 
                else:
                    counter += 1
                    print(f"No improvement {counter}/{patience}.")

                # OPTUNA HYPER PARAM TRAINING CHECK
                _trial_block(trial, current_validation_accuracy, epoch)

                register_in_logs(writer, self, epoch, current_training_loss,current_validation_loss, current_validation_accuracy)

                if counter >= patience:

                    print(f"Early stopping at epoch {epoch+1}")

                    break

            # Training without Validation
            elif (best_loss - current_training_loss) > min_delta:
            
                best_loss = current_training_loss
                acc_value=0
                # save model
                save_path = self.save_checkpoint(epoch, optimizer, current_training_loss,acc_value ,scheduler, hyperparameters)
                _trial_block(trial, acc_value, epoch)

        print("\n---- End of Training ----\n\n")
        if writer:
          writer.close()
        self = self.load_checkpoint(save_path)
        return training_losses, validation_losses, validation_accuracies

    def train_loop(self, loader:"DataLoader", criterion:"_Loss", optimizer:torch.optim)->float:
        """
        Runs one epoch of training.

        Parameters
        ----------
        loader : DataLoader
            Dataloader for training data.
        criterion : _Loss
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.train()
        total_loss = 0.0
        total_samples = 0
        for _, (data, targets) in enumerate(tqdm(loader, desc="Training")):
            # Send data to device
            data = data.to(self.device)
            targets = targets.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Compute prediction and loss
            scores = self(data)
            # scores are the score/probabiity for each class
            loss = criterion(scores, targets)

            # Save loss
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size

            # Backpropagation
            loss.backward()
            optimizer.step()
            total_samples += batch_size
        avg_loss = total_loss / total_samples

        return avg_loss

    def evaluation_loop(self, loader:"DataLoader", criterion:"_Loss")->dict:
        """
        Evaluation loop for validation or test data.

        Parameters
        ----------
        loader : DataLoader
            DataLoader with evaluation data.
        criterion : _Loss
            Loss function.

        Returns
        -------
        dict
            Dictionary containing average loss and various evaluation metrics.
        """
        # Determine task type for metric computation
        num_classes = self.num_classes
        device = self.device

        # Block: Metric initialization
        
        metrics_to_use = ["accuracy_macro", "accuracy_weighted", "accuracy_per_class", "f1", "auroc"]
        metrics_manager = MetricsManager(metrics_to_use, num_classes, device)

        self.eval()  # Set to evaluation mode
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for _, (data, targets) in enumerate(tqdm(loader, desc="Evaluating")):
                
                # Send data to device
                data = data.to(device)
                targets = targets.to(device)

                # Get raw logits from the model
                logits = self(data)

                # Compute loss for the batch
                loss = criterion(logits, targets)

                # Weight loss by batch size and accumulate
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                probs = get_probs(logits, self.num_classes)
                preds = get_preds(logits, self.num_classes)

                metrics_manager.update(preds, targets, probs)

        # Compute all metrics
        results = metrics_manager.compute()
        avg_loss = total_loss / total_samples

        self.train()  

        print(f"Loss: {avg_loss:.4f}")
        metrics_manager.print_summary()
        metrics_manager.reset()      

        results['avg_loss'] = avg_loss
        # metrics = {#"avg_loss": avg_loss,#"accuracy": acc_weighted,## "precision": prec,# "recall": rec, "f1_score": f1,# "specificity": specificity, "cm_metric": cm_metric,# "auroc": auroc}
        return results

    def predict(self, img_tensor) -> tuple[int,float]:
        self.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            scores = self(img_tensor)
            preds = get_preds(scores, self.num_classes)
            probs = get_probs(scores, self.num_classes)
            confidences = get_confidences(probs, preds, self.num_classes)
        
            return preds.item(), confidences.item()



def _trial_block(trial: 'Trial', metric: float, epoch: int) -> None:
    """
    Reports a metric to an Optuna trial and checks for pruning.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object.
    metric : float
        The performance metric (e.g., validation accuracy) to report.
    epoch : int
        The current epoch number.

    Raises
    ------
    optuna.exceptions.TrialPruned
        Raised if the trial should be pruned based on intermediate results.
    """
    if trial:
        
        trial.report(metric, epoch)
        print(
            f"[Optuna] Trial Report: val_acc={metric:.4f} at epoch={epoch+1}")
        if trial.should_prune():
            from optuna.exceptions import TrialPruned
            print("Prunning this Trial...")
            raise TrialPruned


def main_loop(
    dataset_class: "LiverImg",
    dataset_mode: DatasetMode,
    hyperparameters: dict,
    validation: bool = False,
    test_as_validation: bool = False,
    force_cpu: bool = False,
    root_data_dir: str | Path = Path.cwd() / "data/"
) -> tuple[CustomCNN, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Main training loop for a CNN model on liver ultrasound data.

    Parameters
    ----------
    dataset_class : LiverImg
        Class to load liver ultrasound images.
    dataset_mode : DatasetMode
        Mode of the dataset (e.g., full, balanced, etc.).
    hyperparameters : dict
        Dictionary containing model and training hyperparameters.
    validation : bool, optional
        Whether to use a validation set, by default False.
    test_as_validation : bool, optional
        Whether to use the test set as validation set, by default False.
    force_cpu : bool, optional
        Forces model to run on CPU, by default False.
    root_data_dir : str | Path, optional
        Path to the root directory containing the dataset.

    Returns
    -------
    tuple[CustomCNN, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]
        Trained model, optimizer, and learning rate scheduler.
    """
    # ==============================
    # Dataset Paths and Loading
    # ==============================
    root_data_dir = Path(root_data_dir)
    data_dir = root_data_dir/f"model_state/{dataset_mode.name.lower()}/custom"

    # ==============================
    # Image Transformation Pipeline
    # ==============================
    transform = build_transformations(
        hyperparameters["grayscale"], hyperparameters["img_mean"],
        hyperparameters["img_std"],  hyperparameters["resize"],
        hyperparameters["horizontal_flip"])
    
    print("\n --- Data Transformations ---")
    pprint(transform)

    # ==============================
    # Load Dataset & Dataloaders
    # ==============================
    print(f"\nUsing Test as Validation inside Train: {test_as_validation}")



    dataset,loaders = load_data(
        dataset_class=dataset_class,
        dataset_mode=dataset_mode,
        transform=transform,
        batch_size=hyperparameters["batch_size"],
        # Only pass `validation` if not using test as validation
        validation=not test_as_validation and validation,
        test_size=hyperparameters["test_size"],
        only_bmode=hyperparameters["only_bmode"]
        ) 

    if test_as_validation:
        dataset["validation"] = dataset["test"]
        loaders["validation"] = loaders["test"]

    print("Training subset Sizes:", [f"{dat[0]}-{len(dat[1])}" for dat in dataset.items()])

    input_dimensions = get_input_dimensions(loaders["train"])
    num_classes = len(dataset["test"].classes)

    print(f"Input_dimensions:{input_dimensions}")
    print(f"Classes: {num_classes} - {dataset["test"].classes}\n")

    # =================================
    # Train Model
    # =================================

    model = CustomCNN(input_dimensions=input_dimensions,
                  num_classes=num_classes,
                  cv_channels=hyperparameters["cv_layers"],
                  cv_kernels=hyperparameters["cv_kernels"],
                  cv_strides=hyperparameters["cv_strides"],
                  cv_paddings=hyperparameters["cv_paddings"],
                  pool_kernels=hyperparameters["pool_kernels"],
                  pool_strides=hyperparameters["pool_strides"],
                  pool_paddings=hyperparameters["pool_paddings"],
                  hidden_layers=hyperparameters["fc_layers"],
                  batch_norm=hyperparameters["batch_norm"],
                  fcl_dropouts=hyperparameters["fcl_dropouts"],
                 directory=data_dir)

    
    device = model.send_to_device(force_cpu=force_cpu)
    print(f"Model is training on {device}")
    # Loss Function Selection.
    # ==============================
    criterion = get_loss_function(hyperparameters["class_weights"], num_classes, dataset["train"], device)

    # Print Model Info
    dummy_input = torch.randn(
        hyperparameters["batch_size"], input_dimensions[2], input_dimensions[0], input_dimensions[1]).to(model.device)
    try:
        from torchsummary import summary
        print("\nModel Architecture")
        summary(model=model, input_data=dummy_input, device=model.device)
    except ImportError:
        pass

    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    # ReduceLROnPlateau scheduler with validation loss monitoring
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=5, cooldown=2, threshold=0.01)
    # scheduler = StepLR(optimizer=optimizer, mode='min', factor=0.5, patience=5, cooldown=2, threshold=0.01)


    # ========================
    #    FIT
    # ================
    training_losses, validation_losses, validation_accuracies = model.fit(
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparameters=hyperparameters,
        validation=validation)

    # Plot losses if enouch epochs (more than 1)
    print("-- Plots --")
    if len(training_losses) > 1:
        print("\tTraining:",end="")
        plot_evolution(training_losses, plotting="loss", dataset="train", datamode=str(
            dataset_mode.name), color=(1, 0.1, 0.1))
    if len(validation_losses) > 1:
        print("\n\tValidation:",end="")
        plot_evolution(validation_losses, plotting="loss", dataset="validation", datamode=str(
            dataset_mode.name), color=(0.1, 1, 0.1), clean=False)
    if len(validation_accuracies) > 1:
        print("\tValidation:",end="")
        plot_evolution(validation_accuracies, plotting="accuracy", dataset="validation", datamode=str(
            dataset_mode.name), color=(0.1, 0.1, 1))

    # ==============================
    # Model Evaluation
    # ==============================

    # --- Train ---
    print("\n ---- Testing accuracy on Train ---- ")
    print("Train Size:", len(dataset["train"]))
    metrics = model.evaluation_loop(loaders["train"], criterion)
    # cm_metric = metrics["cm_metric"]
    # plot_conf_matrix(cm_metric, f"Train - {dataset_mode.name}")

    # --- Test ---
    print("\n ---- Testing accuracy on Test ---- ")
    print("Test Size:", len(dataset["test"]))
    metrics = model.evaluation_loop(loaders["test"], criterion)
    # cm_metric = metrics["cm_metric"]
    # plot_conf_matrix(cm_metric, f"Test - {dataset_mode.name}")
    return model, scheduler, optimizer




def get_hyperparameters(grayscale: bool, img_mean: [float], img_std: [float], max_epochs: int = 5, **overrides) -> dict:
    """
    Generate a dictionary of training hyperparameters, with logic to adapt 
    image normalization based on B-mode (grayscale) or Doppler (RGB) input. Additional 
    parameters can be overridden dynamically via keyword arguments.

    Parameters
    ----------
    max_epochs : int, optional
        Maximum number of training epochs (default is 200).
    only_bmode : bool, optional
        If True, assumes grayscale B-mode images and sets corresponding
        image normalization values. If False, assumes RGB images (e.g., Doppler) 
        and uses RGB normalization stats (default is True).
    **overrides : dict, optional
        Additional key-value pairs to override or add to the final hyperparameters dictionary.

        learning_rate : float
            Learning rate for the optimizer (default is 5.3e-5).
        batch_size : int, optional
            Number of samples per batch (default is 16).
        custom_norm : list of float or None
            Custom image normalization means. If None, it is inferred from `only_bmode`.
        custom_std : list of float or None
            Custom image normalization standard deviations. If None, it is inferred from `only_bmode`.

    Returns
    -------
    dict
        Dictionary of hyperparameters used for model training. Includes:

        learning_rate : float
            Learning rate for the optimizer during training.

        batch_size : int
            Number of samples per training batch.

        max_epochs : int
            Total number of training epochs.

        patience : int
            Number of epochs with no improvement after which training is stopped early (early stopping).

        batch_norm : bool
            Whether to apply Batch Normalization after convolutional layers.

        cv_layers : list of int
            List defining the number of output channels for each convolutional layer.

        cv_kernels : list of int
            Kernel sizes to be used in convolutional layers.

        cv_strides : list of int
            Stride values for convolutional layers.

        cv_paddings : list of int
            Padding values for convolutional layers.

        pool_kernels : list of int
            Kernel sizes for pooling layers (e.g., MaxPool2d).

        pool_strides : list of int
            Stride values for pooling layers.

        pool_paddings : list of int
            Padding values for pooling layers.

        gap_outputs : tuple or list, optional
            Output shape from Global Average Pooling, if used. If empty, GAP may not be used.

        fc_layers : list of int
            Sizes of fully connected (dense) layers after the convolutional backbone.

        fcl_dropouts : list of float
            Dropout probabilities applied between fully connected layers.

        bmode : bool
            If True, assumes B-mode images; affects input shape and normalization.

        img_mean : list of float
            Mean values for image normalization. Length 1 for grayscale, 3 for RGB.

        img_std : list of float
            Standard deviation values for image normalization. Length 1 for grayscale, 3 for RGB.

        horizontal_flip : float
            Probability of applying a horizontal flip as a data augmentation step.

        resize : tuple of int
            Final image size (height, width) used during preprocessing.

        test_size : float
            Proportion of dataset to use for the test set during splitting.

        only_bmode : bool
            If True, filters the dataset to include only B-mode images.

        class_weights : bool
            If True, applies class weights during loss computation to handle class imbalance.
    
    Dictionary containing all relevant training hyperparameters, including:
        - CNN and FC layer architecture
        - Normalization values
        - Image resize dimensions
        - Training strategy settings (e.g., patience, class_weights)
    """
    expected_len = 1 if grayscale else 3
    if len(img_mean) != expected_len or len(img_std) != expected_len:
        raise ValueError(
            f"For {grayscale=}, img_mean/img_std must have length {expected_len}. "
            f"Got {len(img_mean)=}, {len(img_std)=}."
        )

    base_hyperparams = {
        "learning_rate": 5.3e-05,
        "batch_size": 16,
        "max_epochs": max_epochs,
        "patience": round(0.33 * max_epochs),
        "batch_norm": True,
        "cv_layers": [8, 16, 32, 64, 128],
        "cv_kernels": [3],
        "cv_strides": [1],
        "cv_paddings": [0],
        "pool_kernels": [2],
        "pool_strides": [2],
        "pool_paddings": [0],
        "gap_outputs": (),
        "fc_layers": [128, 64, 32, 16, 8],
        "fcl_dropouts": [0.5],
        "grayscale": grayscale,
        "img_mean": img_mean,
        "img_std": img_std,
        "horizontal_flip": 0.5,
        "resize": ImageTrim().resize(2),
        "test_size": 0.2,
        "only_bmode": True,
        "class_weights": True,
    }

    base_hyperparams.update(overrides)
    return base_hyperparams