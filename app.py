import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_extras.card import card
from streamlit_extras.let_it_rain import rain

# ===========================
# 1. Konfigurasi Halaman
# ===========================
st.set_page_config(
    page_title="Klasifikasi Gambar Kucing vs Anjing",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        text-align: center;
        color: #4a4a4a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: gray;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐶🐱 Klasifikasi Gambar Kucing vs Anjing")
st.markdown("### Aplikasi ini menggunakan model CNN dan Transfer Learning (MobileNetV2) untuk mengklasifikasi gambar hewan peliharaan 🐾")
st.markdown("---")


# ===========================
# 2. Sidebar
# ===========================


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=120)
    st.header("⚙️ Pilih Pengaturan")
    model_type = st.radio("Pilih Model yang Akan Digunakan:", ["CNN Biasa", "MobileNetV2 (Transfer Learning)"])
    st.info("""
    📌 **Petunjuk:**
    - Upload gambar kucing atau anjing.
    - Pilih model.
    - Lihat hasil prediksi dan tingkat keyakinan model.
    """)

    st.markdown("---")
    st.caption("🧠 Dibuat oleh Kelompok AI Week 8 - BINUS Online")

# ===========================
# 3. Load Model
# ===========================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

cnn_model = load_model('model_cats_dogs.h5')
mobilenet_model = load_model('mobilenet_cats_dogs.h5')

# ===========================
# 4. Upload Gambar
# ===========================
uploaded_file = st.file_uploader("📸 Upload Gambar Kucing atau Anjing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar diunggah", use_container_width=True)

    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Model sedang memproses gambar..."):
        if model_type == "CNN Biasa":
            pred = cnn_model.predict(img_array)
        else:
            pred = mobilenet_model.predict(img_array)

    label = "🐶 Anjing" if pred[0][0] > 0.5 else "🐱 Kucing"
    confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]

    # Animasi
    if label == "🐶 Anjing":
        rain(emoji="🐶", font_size=40, falling_speed=5, animation_length="infinite")
    else:
        rain(emoji="🐱", font_size=40, falling_speed=5, animation_length="infinite")

    st.markdown("### 🎯 **Hasil Prediksi**")
    col1, col2 = st.columns(2)

    with col1:
        card(
            title="🧩 Hasil Klasifikasi",
            text=f"<h3 style='text-align:center'>{label}</h3>",
            styles={
                "card": {"background-color": "#f1f3f4", "border-radius": "15px"},
                "title": {"font-size": "20px", "color": "#444"},
            },
        )

    with col2:
        card(
            title="📊 Tingkat Keyakinan Model",
            text=f"<h3 style='text-align:center'>{confidence*100:.2f}%</h3>",
            styles={
                "card": {"background-color": "#e9f7ef", "border-radius": "15px"},
                "title": {"font-size": "20px", "color": "#444"},
            },
        )

else:
    st.info("Silakan upload gambar terlebih dahulu.")

# ===========================
# 5. Footer
# ===========================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🎓 Proyek AI Week 8 — Kelompok Optimasi Model & Implementasi CNN</p>
    <p>BINUS ONLINE LEARNING © 2025</p>
</div>
""", unsafe_allow_html=True)
