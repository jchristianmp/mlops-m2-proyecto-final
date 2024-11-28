import streamlit as st
from PIL import Image 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import time 
from classes.classes import classPrediction


# Objeto de clase predicción
prediccionObj = classPrediction()

# Título
st.markdown("<h1 style='text-align: center; color: white;'>Predicción de Precios de Carros</h1>", 
            unsafe_allow_html=True)
st.divider()
st.html(
    '''
    <style>
    hr {
        border-color: #ff7c55;
    }
    </style>
    '''
)
st.markdown("<h2 style='text-align: center; color: white;'>Christian Matos</h2>", 
            unsafe_allow_html=True)
#image = Image.open("/ws/code/src/images/carsprice.jpeg")
image = Image.open('src/images/carsprice.jpeg') 
st.image(image, use_container_width=True)

# Sidebar
# Custom CSS para el sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 20px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)
#st.sidebar.image("/ws/code/src/images/datapath.png")
st.sidebar.image('src/images/datapath.png') 
st.sidebar.markdown("<h1 style='text-align: center; color: lightgrey;'>MLOps 9na Edición</h1>", 
            unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: center; color: lightgrey;'>Proyecto Final <br/> Módulo 2</h2>", 
            unsafe_allow_html=True)
#st.sidebar.write("MLOps 9va Edición")

# Botón para subir archivo
st.text("")
st.write("Suba el archivo CSV correspondiente para realizar la predicción")
css_carga_archivo='''
<style>
[data-testid="stFileUploaderDropzone"] div div::before { content:"Suba o arraste el archivo aqui"}
[data-testid="stFileUploaderDropzone"] div div span{display:none;}
[data-testid="stFileUploaderDropzone"] div div::after {font-size: .8em; content:"Límite: 200MB por archivo"}
[data-testid="stFileUploaderDropzone"] div div small{display:none;}
</style>
'''
st.markdown(css_carga_archivo, unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=['csv'])

if uploaded_file is not None:
    # Lectura de archvio csv y conversión a dataframe
    df_de_los_datos_subidos = pd.read_csv(uploaded_file)

    # Muestra contenido de csv cargado
    st.write('Contenido del archivo cargado:')
    st.dataframe(df_de_los_datos_subidos)

    if st.button("click aqui para enviar el CSV al Pipeline"):
        if uploaded_file is None:
            st.write("No se cargó correctamente el archivo, subalo de nuevo")
        else:
            with st.spinner('Pipeline y Modelo procesando...'):

                prediccion, prediccion_sin_escalar, datos_procesados = prediccionObj.prediccion_o_inferencia(df_de_los_datos_subidos)
                time.sleep(5)
                st.success('Listo!')

                # Concatenación de predicciones y creación de CSV para descarga
                df_resultado, csv = prediccionObj.contac_prediccion(datos_procesados,
                                                            prediccion,
                                                            prediccion_sin_escalar)

                # Mostramos los resultados de la predicción
                st.write('Resultados de la predicción:')
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown("### Histograma")
                    fig, ax = plt.subplots()
                    pd.Series(np.exp(prediccion)).hist(bins=50, ax=ax)
                    ax.set_title('Precios predichos')
                    ax.set_xlabel('Precio')
                    ax.set_ylabel('Frecuencia')
                    st.pyplot(fig)

                with col2:
                    st.markdown("### Pred.")
                    st.write(prediccion)
                with col3:
                    st.markdown("### Pred. sin escalar")
                    st.write(prediccion_sin_escalar)

                # Mostrar el Dataframe contatenado
                st.write('Datos originales con predicciones:')
                st.dataframe(df_resultado)

                # Botón para descargar el CSV
                st.download_button(
                    label="Descargar archivo CSV con predicciones",
                    data=csv,
                    file_name='predicciones_carros.csv',
                    mime='text/csv',
                )