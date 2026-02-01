import os
import gdown
import streamlit as st
import tensorflow as tf

MODEL_URL = "https://drive.google.com/file/d/1WcJb68tqwR2gMsdy2hARO2J5y0mUCd6J/view?usp=drive_linkE"
MODEL_PATH = "unet_tf219.keras"

@st.cache_resource
def load_unet():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("🧠 Brain Tumor Detection")

model = load_unet()
st.success("✅ Model loaded successfully")
