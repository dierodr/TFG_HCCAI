import streamlit as st
page_icon_file = "./code/source/app/images/logotipo_uva.svg"
st.set_page_config(
    page_title="HCC-IA",
    layout="wide",
    page_icon=page_icon_file,
    menu_items={"About":"Esta aplicación es parte de un Trabajo de Fin de Grado."})

import torch
from PIL import Image

from app_logic import (get_mode_dir, get_avalible_device, create_pdf, 
                        preprocess_image, find_selected_model, 
                        get_targets_text, get_attributions)
from app_view import (show_main_block, show_model_info,
                      centered_info_style, show_gradcam_info,
                      show_image_uploader, show_centered_image,
                      show_selection_columns, show_visualization_settings, show_predict_text)
from datetime import datetime



torch.classes.__path__ = []

device = get_avalible_device()
transform = None

show_main_block()

selected_model, working_mode = show_selection_columns()
dataset_mode, mode_dir = get_mode_dir(working_mode)


try:  
    transform, model_name, model_acc, model = find_selected_model(mode_dir, dataset_mode, selected_model)
    gradcam_avaliable = model.gradcam_compatible
    model_available = True
except FileNotFoundError as e:
    model_available = False
    gradcam_avaliable = False

if not model_available:
    model_problem_text = "Modelo no disponible para el modo de clasificación seleccionado.<br>Por favor, seleccione otro modelo y/o modo."
    centered_info_style(model_problem_text, "WARNING")

else:
    model.to(device)
    show_model_info(model_name, working_mode["index"], model_acc)
    show_gradcam_info(gradcam_avaliable)
 

    st.divider()

    uploaded_file = show_image_uploader()

    #if uploaded_file and model_col and wm_col:
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        show_centered_image(img, caption="",space=[1,6,1])

        classified = False
        with st.spinner("Clasificando...", show_time=True):

            img_tensor = preprocess_image(img, device, transform).to(device)
            predicted, confidence =  model.predict(img_tensor) 
            text_result=f"{get_targets_text(dataset_mode, predicted)} con probabilidad {confidence*100:.2f}%."
            show_predict_text(dataset_mode, text_result, predicted)
        
            classified = True

         
        if classified and gradcam_avaliable:

            st.divider()
            vis_settings = show_visualization_settings()
        
            img_proccesed = False
            with st.spinner("Procesando visualización ...",show_time=True):
                model_vis = get_attributions(transform, img_tensor, model, predicted, vis_settings["relu_attributions"])
                img_heatmap = model_vis.overlay_heatmap(heatmap_cmap=vis_settings["colormap"].lower(), alpha=vis_settings["alpha"],interface_color="white")
                img_proccesed=True            
            if img_proccesed:
                    show_centered_image( img=img_heatmap,caption="Imagen superpuesta",space=[1,6,1])


        with st.spinner("Generando PDF ...", show_time=True):
            
            if gradcam_avaliable:
                img_heatmap_pdf = model_vis.overlay_heatmap(heatmap_cmap=vis_settings["colormap"].lower(), alpha=vis_settings["alpha"],interface_color="black")
            else:
                st.write("")
                
            
            st.download_button(
                label="📄 Descargar informe en PDF",
                data=create_pdf(text_result, model_name, model_acc,  working_mode["text"], img_heatmap_pdf),
                file_name=f"informe_ecografia_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pdf",
                mime="application/pdf"
            )

