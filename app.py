import streamlit as st
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

MODEL_PATH = "unet_FINAL_GOOD.keras"
MODEL_URL = "https://drive.google.com/uc?id=1wDA7N46kMBAmu9m14ND-dsXawmO2gPrc"

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@st.cache_resource
def load_unet():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model… please wait (first run only)")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "dice_loss": dice_loss,
            "dice_coefficient": dice_coefficient
        }
    )

model = load_unet()
st.success("✅ Model loaded successfully!")
