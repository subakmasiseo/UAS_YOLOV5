import pathlib

import streamlit as st
import torch
from PIL import Image

# =========================
# FIX Linux -> Windows Path
# =========================
pathlib.PosixPath = pathlib.WindowsPath

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="YOLOv5 Face Mask Detection", page_icon="😷", layout="centered")

st.title("😷 YOLOv5 Face Mask Detection")
st.markdown(
    """
Aplikasi ini menggunakan **YOLOv5** untuk mendeteksi penggunaan masker wajah.

**Kelas:**
- Masque  
- Notcorrect  
- PasMasque  
"""
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path="weights/best_fixed.pt", force_reload=False)


model = load_model()

# =========================
# CONFIDENCE SLIDER
# =========================
conf = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
model.conf = conf

st.divider()

# =========================
# IMAGE UPLOAD
# =========================
st.subheader("📤 Upload Gambar")
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Asli", use_container_width=True)

    results = model(image)
    results.render()

    st.image(results.ims[0], caption="Hasil Deteksi", use_container_width=True)

st.divider()

# =========================
# CAMERA INPUT
# =========================
st.subheader("📷 Kamera")
use_camera = st.checkbox("Gunakan kamera")

if use_camera:
    cam_image = st.camera_input("Ambil gambar dari kamera")
    if cam_image is not None:
        image = Image.open(cam_image).convert("RGB")
        st.image(image, caption="Gambar Kamera", use_container_width=True)

        results = model(image)
        results.render()

        st.image(results.ims[0], caption="Hasil Deteksi", use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("YOLOv5 • Streamlit • Face Mask Detection")
