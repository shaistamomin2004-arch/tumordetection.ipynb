import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_URL = "https://drive.google.com/uc?id=1wDA7N46kMBAmu9m14ND-dsXawmO2gPrc"
MODEL_PATH = "unet_FINAL_GOOD.keras"

@st.cache_resource
def load_unet():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = load_unet()
st.success("✅ Model loaded successfully")
