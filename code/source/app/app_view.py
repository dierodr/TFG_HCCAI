"""This module encapsulates all the functions used to display info in the main.py
"""

import streamlit as st
from code.source.config.categories import DatasetMode, ModelNames

 
def show_predict_text(dataset_mode,text_result, label_id):
    text = f"Predicción: {text_result}"
    
    if dataset_mode == DatasetMode.HEALTHY_LIVERS_OR_NOT:#SANO O NO
        if label_id == 0:
            centered_info_style(text, "GOOD")
        elif label_id == 1:
            centered_info_style(text, "BAD")

    elif dataset_mode == DatasetMode.CIRRHOTIC_STATE:#DOLENCIA
        if label_id ==0:
            centered_info_style(text, "GOOD")

        elif label_id ==1:
            centered_info_style(text, "WARNING")

        elif label_id ==2:
            centered_info_style(text, "BAD")
    elif dataset_mode == DatasetMode.ORGAN_CLASSIFICATION:#TIPO ORGANO
        centered_info_style(text, "INFO")
 

def show_selection_columns():
    wm_col, model_col = st.columns(2)
    with wm_col:
        working_mode = show_mode_selection(default_mode=2)
    with model_col:
        selected_model = show_model_selection(default_model=6)

    return selected_model, working_mode

def centered_info_style(text,something):
    if something == "BAD":            
        background_color = "#3e2328"
        font_color =  "#ffdede"
    elif something == "GOOD":
        background_color = "#173828"
        font_color = "#dffde9"
    elif something == "WARNING":
        background_color = "#3e3a16"
        font_color = "#ffffc2"
    elif something == "INFO":
        background_color = "#172c43"
        font_color = "#c7ebff"
    st.markdown(f"""
            <div style='
                justify-content: center;
                margin-top: 1rem;
                text-align: center;
                width: 100%;
            '>
                <div style='
                    background-color: {background_color};
                    color: {font_color};
                    font-size: 1.75rem;
                    padding: 0.75rem 0px 1rem;
                    font-weight: bold;
                    border-radius: 9px;
                    font-family: "Source Sans Pro", sans-serif;
                    font-weight: 600;
                    line-height: 1.2;
                    margin: 0px;
                '>
                    {text}
                </div>
            </div>
            """, unsafe_allow_html=True)
       

def show_centered_image(img,caption,space=[1,2,1]):
    # Create three columns (empty - center - empty)
    left_col, center_col, right_col = st.columns(space)  # Ratio can be tuned

    # Show image in the center column
    with center_col:
        st.image(img, caption=caption,
                use_container_width=False)           


def show_gradcam_info(gradcam_avaliable):
    if gradcam_avaliable:
        #gradcam_text = "Este modelo :green-badge[SÍ] ofrece visualización de la toma de decisiones"
        st.markdown("""
    <div style='
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.875rem;;
        color: #ddd;
        text-align: center;
    '>
        Este modelo <span style="
        background-color: #1b4b31;
        color: #3dd56d;
        padding: 0.1rem 0.3rem;
        font-size: 0.875rem;
        border-radius: 4px;
    ">SÍ</span> ofrece visualización de la toma de decisiones.
</div>
""", unsafe_allow_html=True)
    else:
        #gradcam_text = "Este modelo :red-badge[NO] ofrece visualización de la toma de decisiones"
        st.markdown("""
        <div style='
            border: 1px solid #333;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            margin-top: 1rem;
            font-size: 0.875rem;;
            color: #ddd;
            text-align: center;
        '>
            Este modelo <span style="
            background-color: #562c31;
            color: #ff4b4b;
            padding: 0.1rem 0.3rem;
            font-size: 0.875rem;
            border-radius: 4px;
        ">NO</span> ofrece visualización de la toma de decisiones.
    </div>
    """, unsafe_allow_html=True)


def show_model_info(classifier_selection, working_mode_i,model_mode_acc):
    main_options = ["Órganos", "Sano o Enfermo", " Estado Cirrotico"]
    selected_main_op = main_options[working_mode_i]
    text_accuracy = f"Tasa de acierto de <b>{classifier_selection}</b> para clasificar <b>{selected_main_op}</b>: {model_mode_acc}%."
    centered_info_style(text_accuracy,"INFO")


def show_mode_selection(default_mode):

    st.subheader("Modo de funcionamiento")
    working_mode = {"Órganos: Clasifica entre Hígado, Bazo, Pancreas, Riñón o Vesícula": 0,
                    "Sano o Enfermo: Determina si un Higado está sano o enfermo": 1, 
                    "Estado Cirrotico: Determina el estado de un higado (Sano, Cirrotico o Hepatocarcinoma)": 2}
    
    wm_selected_text = st.selectbox(
        "Elija como quiere clasificar:",
        options=list(working_mode.keys()),
        index=list(working_mode.values())[default_mode],
        help=""
    )
    
    return  {"text": wm_selected_text, "index": working_mode[wm_selected_text]}


def show_model_selection(default_model):
    st.subheader("Selección de modelo")

    options = list(ModelNames.all()) + ["Mejor modelo"]

    selection = st.selectbox(
        "Elija qué modelo quiere usar:",
        options=[opt.value if isinstance(opt, ModelNames) else opt for opt in options],
        index=default_model
    )

    if selection == "Mejor modelo":
        return selection
    return ModelNames.from_value(selection)


def show_image_uploader():
    uploaded_file = st.file_uploader(
        "#### Arrastre o seleccione un archivo (JPG, JPEG, PNG, BMP, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if not uploaded_file:
        st.markdown("<h3 style='text-align: center;'>📤Cargue una ecografía</h3>", unsafe_allow_html=True)

    return uploaded_file


def show_visualization_settings():
    colormap_col, checkbox_col = st.columns(2)
    with colormap_col:
        colormap = st.selectbox(
            "Seleccione un mapa de color:", ("JET", "Seismic", "Viridis"), index=0, placeholder="JET")
    with checkbox_col:
        st.write("")
        st.write("")
    
        relu_attributions = st.checkbox("Solo contribuciones positivas.")

    default_alpha = 0.5

    alpha = st.slider("Nivel de transparencia:",
                    min_value=0.0, max_value=1.0, step=0.05, value=default_alpha)
    viz_settings={
        "colormap":colormap, 
        "relu_attributions":relu_attributions, 
        "alpha":alpha
    }
    return viz_settings


def show_main_block():
    st.html("""
    <style>
        .stMainBlockContainer {
            max-width: 90rem;
            margin-left: auto;
            margin-right: auto;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """
    )

    #st.title("Clasificador de Ecografías")
    st.markdown("<h1 style='text-align: center;'>Clasificador de Ecografías Abdominales</h1>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: left; margin-left: 30%;'>
    Para clasificar:
    </p>
    <div style='text-align: center;'>
    <ul style='display: inline-block; text-align: left;'>
        <li>Seleccione un modo de funcionamiento según el tipo de clasificación que desee realizar.</li>
        <li>Elija el modelo de IA que desea usar para realizar la clasificación.</li>
        <li>Suba una ecografía de la región abdominal.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)