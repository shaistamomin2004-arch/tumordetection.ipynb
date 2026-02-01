import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Model download details
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1WcJb68tqwR2gMsdy2hARO2J5y0mUCd6J"
MODEL_PATH = "unet_FINAL_GOOD.keras"

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_unet_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model


model = load_unet_model()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Brain Tumor Detection")
st.write(
    "Upload an MRI image. "
    "The image is processed **in memory only** and **not saved** anywhere."
)

# -------------------------------
# File uploader (PRIVACY SAFE)
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    # Read image (RAM only)
    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Preprocess
    img = image.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 128, 128, 1)

    # Predict
    with st.spinner("🧠 Analyzing MRI..."):
        prediction = model.predict(img)

    # Display result
    st.success("✅ Prediction completed")

    st.write("### Output mask (raw values)")
    st.write(prediction)

# -------------------------------
# Footer
# -------------------------------
st.caption(
    "⚠️ Educational project only. "
    "No images are stored. "
    "Public BraTS-style data usage."
)
