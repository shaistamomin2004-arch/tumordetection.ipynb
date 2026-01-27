import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# 🔹 YOUR OLD CODE (slightly cleaned)
@st.cache_resource
def load_unet():
    return load_model("unet_FINAL_GOOD.keras", compile=False)

model = load_unet()
st.success("✅ Model loaded successfully!")
