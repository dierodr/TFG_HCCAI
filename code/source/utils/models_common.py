import os
import sys


import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

sys.path.insert(0, os.path.abspath('..'))


def get_class_weights(samples_per_class: list, device: str) -> "torch.Tensor":
    """Calculates inverse-frequency weights for each class.

    Parameters
    ----------
    samples_per_class : list
        Number of samples per class.
    device : str
        Device on which the tensor will be stored.

    Returns
    -------
    torch.Tensor
        Tensor of weights for each class.
    """

    class_counts = torch.tensor(samples_per_class, dtype=torch.float32)
    weights = 1.0 / class_counts
    weights = weights.to(device)
    
    weights = weights * (len(class_counts) / weights.sum())
  
    return weights


def step_scheduler(
    scheduler: _LRScheduler | ReduceLROnPlateau,
    metric: float | None = None
) -> None:
    """
    Auto-step scheduler based on its class: handles both ReduceLROnPlateau and other schedulers.

    Parameters
    ----------
    scheduler : _LRScheduler or ReduceLROnPlateau
        A PyTorch learning-rate scheduler.
    metric : float, optional
        Required for ReduceLROnPlateau (validation metric to monitor).

    Raises
    ------
    ValueError
        If ReduceLROnPlateau is used without providing `metric`.
    TypeError
        If an unsupported scheduler type is passed.
    """
    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau requires `metric` to step.")
        scheduler.step(metric)
    elif isinstance(scheduler, _LRScheduler):
        scheduler.step()
    else:
        raise TypeError(f"Unsupported scheduler type: {type(scheduler).__name__}")


def register_in_logs(
    writer: "SummaryWriter",
    model: "Module",
    epoch: int,
    current_training_loss: float,
    current_validation_loss: float,
    current_validation_accuracy: float
) -> None:
    """
    Logs training metrics and gradients to TensorBoard.

    Parameters
    ----------
    writer : SummaryWriter
        TensorBoard writer instance.
    model : Module
        PyTorch model being trained.
    epoch : int
        Current epoch number.
    current_training_loss : float
        Training loss for the epoch.
    current_validation_loss : float
        Validation loss for the epoch.
    current_validation_accuracy : float
        Validation accuracy for the epoch.
    """
    if writer:
        writer.add_scalar('Loss/Train', current_training_loss, epoch)
        writer.add_scalar('Loss/Validation', current_validation_loss, epoch)
        writer.add_scalars('Loss', {'Train': current_training_loss, 'Validation': current_validation_loss}, epoch)
        writer.add_scalar('Accuracy/Validation',current_validation_accuracy, epoch)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"grad/{name}", param.grad, epoch)


def get_loss_function(use_weights:bool, num_classes:int, dataset:"Dataset", device:str)->"_Loss":
    """Creates and return a loss function 

    Parameters
    ----------
    use_weights : bool
        Whether to use class weights to calculate loss for each class.
    num_classes : int
        Number of classes of the data
    dataset : Dataset
        Dataset used to infer class distribution for weighting.
    device : str
        Device where the calculations will be made.

    Returns
    -------
    _Loss
        loss function to use
        
    """

    if use_weights:
        class_distribution = dataset.class_dist.values
        if num_classes <= 2:
            neg, pos = class_distribution[0], class_distribution[1]
            pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
            criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            class_weight = get_class_weights(class_distribution, device)
            criterion = CrossEntropyLoss(weight=class_weight)
    else:
        criterion = (BCEWithLogitsLoss() if num_classes <= 2 else CrossEntropyLoss())
    return criterion


def get_model_name(accuracy: float, loss: float) -> str:
    """
    Generate a filename for a saved model using its accuracy and loss.

    Parameters
    ----------
    accuracy : float
        Accuracy as a decimal (e.g., 0.68 for 68%).
    loss : float
        Loss value (e.g., 0.123).

    Returns
    -------
    str
        Filename string in the format:
        "m_aXX.XX_lY.YYY.pth" where:
        - XX.XX = accuracy * 100 rounded to two decimal places
        - Y.YYY = loss rounded to three decimal places
    """
    
    return f"m_a{accuracy*100:.2f}_l{loss:.3f}.pth"


def get_probs(scores: "torch.Tensor", num_classes: int) -> "torch.Tensor":
    if num_classes > 2:
        probs = F.softmax(scores, dim=1)#multi
    else:
        probs = torch.sigmoid(scores)#binary
    return probs


def get_preds(scores: "torch.Tensor", num_classes: int) -> "torch.Tensor":
    if num_classes > 2:
        preds = torch.argmax(scores, dim=1)#multi
    else:
        preds = (scores >= 0).float()#binary
    return preds


def get_confidences(probs: "torch.Tensor", preds: "torch.Tensor", num_classes: int) -> "torch.Tensor":
    if num_classes>2:
        confidences = probs[torch.arange(len(probs)), preds]
    else:
        confidences = probs.clone()
        confidences[preds == 0] = 1.0 - confidences[preds == 0]
    
    return confidences


