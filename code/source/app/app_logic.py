from pathlib import Path

import re
import torch
from torch import Tensor



from PIL import Image

from code.source.PretrainedModel import PretrainedModel
from code.source.config.categories import DatasetMode, ModelNames
from code.source.utils.files import find_best_model
from code.source.CustomCNN import CustomCNN, build_transformations
from code.source.ModelVisualizer import ModelVisualizer
from typing import Optional
from fpdf import FPDF
import tempfile
import torchvision.transforms.v2 as transforms  

torch.classes.__path__ = []
def get_targets_text(dataset_mode,target):
    targets_dicts = {
        DatasetMode.ORGAN_CLASSIFICATION: {0: "Hígado",     1: "Bazo",     2: "Pancreas",     3: "Riñón",     4: "Vesícula"},
        DatasetMode.HEALTHY_LIVERS_OR_NOT: {0: "Hígado Sano", 1: "Hígado Enfermo"},
        DatasetMode.CIRRHOTIC_STATE: {0: "Hígado Sano", 1: "Hígado Cirrotico", 2: "Hepatocarcinoma"}}

    return targets_dicts[dataset_mode][target]

def get_avalible_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_pdf(
    prediction_text: str,
    selected_model: str,
    accuracy: float,
    selected_mode: str,
    image: Image.Image | None
) -> bytes:
    """
    Generates a classification report in PDF format.

    Parameters
    ----------
    prediction_text : str
        The predicted class.
    selected_model : str
        Name of the model used.
    accuracy : float
        Accuracy percentage.
    selected_mode : str
        Dataset mode used.
    image : PIL.Image.Image or None
        Grad-CAM overlay image.

    Returns
    -------
    bytes
        PDF as bytes (latin-1 encoded).
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=24)
    pdf.cell(200, 10, txt="Informe de Clasificación", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Modelo utilizado:")
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"\t\t\t\t{selected_model}", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Precisión del modelo:")
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"\t\t\t\t{accuracy:.2f}%", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Modo de funcionamiento:")
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"\t\t\t\t{selected_mode}", ln=True)     

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Predicción:")
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"\t\t\t\t{prediction_text}", ln=True)


    # Save image_bytes to a temporary in-memory file
    if image:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Imagen:")
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Ecografia con Grad-CAM superpuesto.", ln=True, align='C')
        pdf.cell(200, 10, txt="Ecografia con Grad-CAM superpuesto.", align='C')

        # Save PIL image to disk (needed by fpdf)
        # Create a temporary image file for fpdf to use
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img_file:
            image.save(tmp_img_file, format="png")
            tmp_img_path = tmp_img_file.name  # Get path for fpdf
            # Add image (saved earlier)
            pdf.image(tmp_img_path, x=11, y=107, w=195)

    pdf_output = pdf.output(dest='S').encode('latin-1')
    return pdf_output


def preprocess_image(
    image: Image.Image,
    device: str,
    transform: "transforms.Compose"
) -> Tensor:
    """
    Preprocesses a PIL image for model input.

    Applies the specified transformation, moves it to the correct device,
    and adds a batch dimension.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to preprocess.
    device : str
        The device to move the image to (e.g., "cuda" or "cpu").
    transform : torchvision.transforms.Compose or None, optional
        The image transformation pipeline. If None, no transformation is applied.

    Returns
    -------
    torch.Tensor
        A 4D tensor with shape (1, C, H, W), ready for model input.
    """
    img_tensor = transform(image)
    img_tensor = img_tensor.to(device)
    return img_tensor.unsqueeze(0)

def get_mode_dir(working_mode):
    dataset_mode = DatasetMode.all()[working_mode["index"]]
    mode_dir = dataset_mode.directory()
    return dataset_mode, mode_dir

def get_acc_from_path(model_path):
    model_path = Path(model_path)
    # Regex to find pattern like a0.688
    match = re.search(r"a(\d+\.\d+)", model_path.name)
    #match = re.search(r"a(\d+(?:\.\d+)?)", model_path.name)
    if match:
        acc = float(match.group(1))
        if acc <= 1:
            acc=acc*100
        return acc
    else:
        raise ValueError(f"Accuracy not found in model path: {model_path}")


def prepare_pretrained(model_path,selected_model ,dataset_mode):
    num_classes=dataset_mode.num_classes
    root_path=dataset_mode.directory()
    model = PretrainedModel(selected_model, root_path, num_classes, download_weights=False)
    transforms = model.load(model_path)
    
    return model, transforms


def prepare_custom(model_path):
    
    model, epoch, loss, scheduler, optimizer, params = CustomCNN.load_model(model_path)
    model.send_to_device()
    model.eval()

    transform = build_transformations(grayscale = params["grayscale"], 
                                      img_mean =  params["img_mean"], 
                                      img_std  =  params["img_std"], 
                                      resize =    params["resize"],
                                      horizontal_flip = 0)

    return model, transform


def get_normalization(transform)-> Optional[tuple[list[float], list[float]]]:
    for t in transform.transforms:
        if isinstance(t, transforms.Normalize):
            return (t.mean, t.std)
    return None,None


def get_attributions(transform,img_tensor,model,predicted,relu_attributions):
    normalization = get_normalization(transform)
    model_vis = ModelVisualizer(model,model.get_last_conv_layer, normalization)
    model_vis.compute_gradcam(img_tensor,predicted,relu_attributions)


    return model_vis


def find_selected_model(mode_dir, dataset_mode, selected_model):

    if selected_model ==  ModelNames.MYCNN:#CUSTOM CNN    

        models_dir = mode_dir/"custom"
        best_model_path = find_best_model(models_dir)
        model, transform = prepare_custom(best_model_path)
        model_acc = get_acc_from_path(best_model_path)


    elif selected_model in  ModelNames.pretrained():#PRETRAINED CNNs
       
        models_dir = mode_dir/"pretrained"
        model_path = models_dir/selected_model.value
        best_model_path = find_best_model(model_path)
        model, transform = prepare_pretrained(best_model_path, selected_model, dataset_mode)
        model_acc = get_acc_from_path(best_model_path)

    else:#BEST MODEL
        
        model_path=find_best_model(mode_dir)
        
        if 'pretrained' in model_path.parts:
            selected_model = ModelNames.from_value(model_path.parts[-2])
            model, transform = prepare_pretrained(model_path, selected_model, dataset_mode)  
            model_acc = get_acc_from_path(model_path)

        elif 'custom' in model_path.parts:
            selected_model= ModelNames.MYCNN
            model_path = mode_dir/"custom"
            best_model_path = find_best_model(model_path)
            model, transform = prepare_custom(best_model_path)
            model_acc = get_acc_from_path(best_model_path)

        else:
            raise FileNotFoundError(f"No valid .pth files found for {model_path}")

    model_name = selected_model.value
    
    return transform, model_name, model_acc, model
