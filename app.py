import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

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

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


model = load_unet_model()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Brain Tumor Detection")
st.write(
    "Upload an MRI image. "
    "*Images are processed in memory only and never saved.*"
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
        width=400
    )

    # Preprocess
    img = image.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img_input = img.reshape(1, 128, 128, 1)

    # Predict
    with st.spinner("🧠 Analyzing MRI..."):
        prediction = model.predict(img_input)

    st.success("✅ Prediction completed")

    # -------------------------------
    # Post-process prediction
    # -------------------------------
    pred_mask = prediction[0, :, :, 0]        # (128,128)
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # -------------------------------
    # Display results
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Preprocessed MRI", clamp=True)

    with col2:
        st.image(binary_mask * 255, caption="Predicted Mask", clamp=True)

    with col3:
        # Overlay
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.imshow(binary_mask, cmap="Reds", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.caption(
    "⚠️ Educational use only • No images stored • "
    "Uses publicly available BraTS-style data"
)
