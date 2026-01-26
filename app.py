from tensorflow.keras.models import load_model

@st.cache_resource
def load_unet():
    return load_model("/content/drive/MyDrive/unet_FINAL_GOOD.keras", compile=False)

model = load_unet()
