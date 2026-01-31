import os
import gdown
import streamlit as st
import tensorflow as tf

# 🔴 FORCE TensorFlow to use legacy Keras (THIS FIXES YOUR ERROR)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

MODEL_URL = "https://drive.google.com/uc?id=1wDA7N46kMBAmu9m14ND-dsXawmO2gPrc"
MODEL_PATH = "unet_FINAL_GOOD.keras"

@st.cache_resource
def load_unet():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("🧠 Brain Tumor Detection")

model = load_unet()
st.success("✅ Model loaded successfully")
