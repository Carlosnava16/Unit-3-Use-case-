import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.set_page_config(page_title="ASL Alphabet Classifier", page_icon="🤟", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4B0082;'>ASL Alphabet Classifier</h1>
    <p style='text-align: center; font-size:18px; color: #555;'>
    Suba una imagen de un signo ASL (A, B, C, ...) y el modelo predirá la letra.
    </p>
""", unsafe_allow_html=True)

st.write("---")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg","png"])

model = load_model("best_asl_model.h5")
labels = {0:"A", 1:"B", 2:"C"}  # Ajusta según tus clases

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
    
    img = image.load_img(uploaded_file, target_size=(64,64))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred_prob = model.predict(img_array)
    pred_class = np.argmax(pred_prob)
    confidence = np.max(pred_prob) * 100
    
    st.markdown(f"""
        <h2 style='text-align:center; color:#228B22;'>
        Predicción: {labels[pred_class]} 
        </h2>
        <p style='text-align:center; color:#555;'>
        Confianza: {confidence:.2f}%
        </p>
    """, unsafe_allow_html=True)
    
    st.write("---")

st.markdown("""
    <p style='text-align:center; color:#888; font-size:12px;'>
    Creado por Carlos Navarrete, Angel Lopez y Angel Gabriel - Proyecto ASL Alphabet
    </p>
""", unsafe_allow_html=True)

#!pip install pyngrok
#from pyngrok import ngrok

#  Configura tu token (pon el tuyo aquí)
#ngrok.set_auth_token("31LGgmyA2tk2SaErn97c6Nridxi_7LSR9nZDT2Pk1icmMGXaK")

# Abre el túnel correctamente (nuevo formato)
#public_url = ngrok.connect("http://localhost:8501")
#print("Tu aplicación está disponible en:", public_url)

# Inicia streamlit
#!streamlit run app.py --server.port 8501 &>/dev/null&
