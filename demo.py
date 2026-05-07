import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import streamlit as st

modelo = tf.keras.models.load_model("modelo_ajedrez.keras")
nombres = ['Alfil blanco', 'Alfil negro', 'Caballo blanco', 'Caballo negro', 'Dama blanca', 'Dama negra', 'Peon blanco', 'Peon negro', 'Rey blanco', 'Rey negro', 'Torre blanca', 'Torre negra']

st.title("Clasificador de piezas de ajedrez")

archivo = st.file_uploader("Sube una imagen")
st.caption("El modelo fue entrenado con piezas específicas. Para mejores resultados, usá fotos con fondo claro y buena iluminación.")
if archivo is not None: 
    img_original = Image.open(archivo)
    img = np.array(img_original)
    img = cv2.resize(img,(224,224))
    img = img * 1/255
    img = np.expand_dims(img,axis=(0))
    prediccion = modelo.predict(img)
    clase = np.argmax(prediccion)
    nombre = nombres[clase]
    confianza = prediccion[0][clase]
    st.image(img_original)
    st.write(f'**{nombre}**,- Confianza: {round(float(confianza)*100,1)}')
