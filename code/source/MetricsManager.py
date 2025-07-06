import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC, Specificity, ROC


class MetricsManager:
    def __init__(self, metrics_to_use, num_classes, device, default_average_type: str = "macro"):
        """
        Initialize requested metrics.

        Parameters
        ----------
        metrics_to_use : list of str
            List of metric names to use.
        num_classes : int
            Number of classes.
        device : torch.device
            Device to place metrics on.
        default_acerage_type: str
            Average method to use by default
        """
        self.device = device
        self.task_type = "multiclass" if num_classes > 2 else "binary"
        self.num_classes = num_classes

        self.metrics = {}
        
        for name in metrics_to_use:
            if name == 'accuracy_macro':
                self.metrics[name] = Accuracy(task=self.task_type, num_classes=num_classes, average='macro').to(device)
            elif name == 'accuracy_weighted':
                self.metrics[name] = Accuracy(task=self.task_type, num_classes=num_classes, average='weighted').to(device)
            elif name == 'accuracy_per_class':
                self.metrics[name] = Accuracy(task=self.task_type, num_classes=num_classes, average=None).to(device)
            elif name == 'f1':
                self.metrics[name] = F1Score(task=self.task_type, average=default_average_type, num_classes=num_classes).to(device)
            elif name == 'auroc':
                self.metrics[name] = AUROC(task=self.task_type, num_classes=num_classes).to(device)
            elif name == 'precision':
                self.metrics[name] = Precision(task=self.task_type, average=default_average_type, num_classes=num_classes).to(device)
            elif name == 'recall':
                self.metrics[name] = Recall(task=self.task_type, average=default_average_type,num_classes=num_classes).to(device)
            elif name == 'confusion_matrix':
                self.metrics[name] = ConfusionMatrix(num_classes=num_classes).to(device)
            elif name == 'specificity':
                self.metrics[name] = Specificity(task=self.task_type, average=default_average_type, num_classes=num_classes).to(device)
            elif name == 'roc':
                self.metrics[name] = ROC(task=self.task_type, num_classes=num_classes).to(device)
            else:
                raise ValueError(f"Unsupported metric requested: {name}")

    def update(self, preds, targets, probs=None):
        """
        Update all metrics with predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted classes.
        targets : torch.Tensor
            True targets.
        probs : torch.Tensor, optional
            Predicted probabilities (required for some metrics like AUROC).
        """
        for name, metric in self.metrics.items():
            # Metrics that need probabilities for update
            if name in {'auroc', 'roc'}:
                if probs is None:
                    raise ValueError(
                        f"Metric {name} requires predicted probabilities.")
                metric.update(probs, targets)
            else:
                metric.update(preds, targets)

    def compute(self):
        """
        Compute all metrics.

        Returns
        -------
        dict
            Dictionary of metric name to computed scalar or tensor result.
        """
        results = {}
        for name, metric in self.metrics.items():
            val = metric.compute()
            # convert zero-dim tensor to python scalar
            if torch.is_tensor(val) and val.dim() == 0:
                val = val.item()
            results[name] = val
        return results

    def reset(self):
        """Reset all metrics to initial state."""
        for metric in self.metrics.values():
            metric.reset()

    def print_summary(self):
        """Print metrics summary."""
        results = self.compute()
        for name, value in results.items():
            if isinstance(value, float):
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: {value}")
