from tensorflow.keras.models import load_model

@st.cache_resource
def load_unet():
    return load_model("unet_FINAl_GOOD.keras", compile=False)

model = load_unet()
