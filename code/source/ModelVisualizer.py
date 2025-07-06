from datetime import datetime
from io import BytesIO
from pathlib import Path
import sys
from typing import Optional
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from torch import cat
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from captum.attr import LayerGradCam, LayerAttribution


class ModelVisualizer:
    """
    Class for computing and visualizing attributions on a given model.
    """

    def __init__(
        self,
        model: "nn.Module",
        conv_layer: "nn.Module",
        normalization: Optional[tuple[list[float], list[float]]] = None,
        device: Optional["torch.device"] = None,
    ):
        """
        Initializes the visualizer.

        Parameters
        ----------
        model : nn.Module
            The trained model.
        conv_layer : nn.Module
            Convolutional layer that's going to be used for Grad-CAM.
        normalization : tuple of (mean, std), optional
            Mean and std used for unnormalization.
        device : torch.device, optional
            Device to use (CPU or CUDA).
        """
        self.model = model.eval()
        self.conv_layer = conv_layer
        self.normalization = normalization
        self.device = device or next(model.parameters()).device
        self.is_interactive = hasattr(sys, "ps1")
        self.timestamp = datetime.now()

    def unnormalize_tensor(self,tensor: "torch.Tensor") -> "torch.Tensor":
        """
        Unnormalizes a tensor using the given mean and std.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape (C, H, W)
        mean : list of float
            Mean used for normalization.
        std : list of float
            Std used for normalization.

        Returns
        -------
        torch.Tensor
            Unnormalized tensor
        """
        mean, std = self.normalization
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        
        return tensor
    
    def tensor_to_pil(self, tensor: "Tensor") -> Image.Image:
        """
        Converts a tensor to a PIL image, with optional unnormalization.

        Parameters
        ----------
        tensor : Tensor
            Tensor of shape (C, H, W) or (1, C, H, W).

        Returns
        -------
        Image.Image
            PIL image.
        """
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.detach().cpu().float()

        if self.normalization:
            tensor = self.unnormalize_tensor(tensor)

        tensor = tensor.clamp(0, 1)
        return to_pil_image(tensor)

    def compute_gradcam(
        self,
        input_tensor: "Tensor",
        target: int,
        relu_attributions: bool = False,
    ) -> "Tensor":
        """
        Computes Grad-CAM attribution.

        Parameters
        ----------
        input_tensor : Tensor
            Input image of shape (1, C, H, W).
        target : int
            Target class index.
        relu_attributions : bool
            Whether to apply ReLU to attributions.

        Returns
        -------
        Tensor
            Upsampled Grad-CAM heatmap of shape (1, 1, H, W).
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        self.input_tensor = input_tensor

        def forward_fn(x):
            logits = self.model(x)
            return cat([logits, -logits], dim=1)

        gradcam = LayerGradCam(forward_fn, self.conv_layer)
        attributions = gradcam.attribute(
            input_tensor,
            target=int(target),
            relu_attributions=relu_attributions,
        )

        H, W = input_tensor.shape[2], input_tensor.shape[3]

        self.attributions = LayerAttribution.interpolate(
            attributions,
            interpolate_dims=(H, W),
            interpolate_mode="bilinear",
        )   

        return self.attributions 

    def overlay_heatmap(
        self,
        original_image: Image.Image = None,
        attributions: "torch.Tensor" = None,
        heatmap_cmap: str="jet",
        alpha: float = 0.6,
        interface_color: str = "white",
        scale: int = 16,
        save = False
    ) -> Image.Image:
        """
        Overlays a heatmap of attributions on top of the original image and returns the result.

        Parameters
        ----------
        original_image : PIL.Image.Image
            The original image to overlay the attributions on.
        attributions : torch.Tensor
            Attribution map of shape (1, 1, H, W), typically from Grad-CAM or saliency methods.
        heatmap_cmap : str
            Matplotlib colormap name for the attribution heatmap.
        alpha : float, optional
            Opacity level of the heatmap overlay, by default 0.6.
        interface_color : str, optional
            Color used for interface elements like axis and colorbar ticks, by default "white".
        scale: 
            numer used to control the scale of the produced image.

        Returns
        -------
        PIL.Image.Image
            Blended image with heatmap overlay.
        """
        if original_image is None:
            original_image = self.tensor_to_pil(self.input_tensor)

        if attributions is None:
            attributions = self.attributions 

        attributions_np = attributions.detach().cpu().numpy()[0][0]
        cmap = 'gray' if len(original_image.getbands()) == 1 else None # RGB

        w, h = original_image.size
        fig_shape = np.array([h, w]) / max(w, h) * scale
        fig, ax = plt.subplots(figsize=(fig_shape))

    
        ax.imshow(original_image, cmap=cmap)
        im = ax.imshow(attributions_np, cmap=heatmap_cmap, alpha=alpha)
        ax.axis('off')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_tick_params(color=interface_color)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=interface_color)
        cbar.outline.set_edgecolor(interface_color)
        cbar.set_label("Asignación Automática", color=interface_color, fontsize=12, labelpad=10)

        # If images cant be show saved them
        plt.tight_layout()
        if self.is_interactive:
            plt.show()
        elif save:
            file = Path.cwd() / \
                f"data/plots/model_expl/{self.timestamp}_model_attr_overlay.png"
            file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Non-interactive mode: saving to {file}")
            plt.savefig(file)

        # Save to memory buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)

