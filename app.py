import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_extras.card import card
from streamlit_extras.let_it_rain import rain
import os
import gdown

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
# 3. Fungsi Download & Load Model
# ===========================
def download_from_gdrive(file_id, output_path):
    """Download file dari Google Drive jika belum ada di lokal"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

@st.cache_resource
def load_model_safe_gdrive(local_path, gdrive_file_id, model_type):
    """Load model Keras, fallback rebuild jika gagal"""
    path = download_from_gdrive(gdrive_file_id, local_path)
    
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.warning(f"⚠️ Gagal load model. Mencoba rebuild...")
        
        if model_type == "mobilenet":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(160, 160, 3),
                include_top=False,
                weights=None,
                pooling='avg'
            )
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:  # CNN Biasa
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        try:
            model.load_weights(path)
            st.success("✅ Model berhasil direbuild dan bobot dimuat.")
            return model
        except:
            st.error("❌ Model gagal dimuat, pastikan file model sesuai dengan jenis arsitektur.")
            return None

# ===========================
# 4. ID Google Drive Model
# ===========================
FILE_ID_CNN = "1bUfKbyoEmZG25eYXW_Uz1pUUdH35YfF1"
FILE_ID_MOBILENET = "11MT4ubPGdiTGOC8hr_xZbOeKriXwe0RM"

# ===========================
# 5. Load Model
# ===========================
cnn_model = load_model_safe_gdrive('model_cats_dogs.h5', FILE_ID_CNN, 'cnn')
mobilenet_model = load_model_safe_gdrive('mobilenet_cats_dogs.keras', FILE_ID_MOBILENET, 'mobilenet')

# ===========================
# 6. Upload & Prediksi
# ===========================
uploaded_file = st.file_uploader("📸 Upload Gambar Kucing atau Anjing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Gambar diunggah", use_container_width=True)

    if model_type == "CNN Biasa":
        img_resized = image.resize((150, 150))
        model = cnn_model
    else:
        img_resized = image.resize((160, 160))
        model = mobilenet_model

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Model sedang memproses gambar..."):
        pred = model.predict(img_array)

    label = "🐶 Anjing" if pred[0][0] > 0.5 else "🐱 Kucing"
    confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]

    # Animasi hasil
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
    st.info("📤 Silakan upload gambar terlebih dahulu untuk melihat hasil prediksi.")

# ===========================
# 7. Footer
# ===========================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🎓 Proyek AI Week 8 — Kelompok Optimasi Model & Implementasi CNN</p>
    <p>BINUS ONLINE LEARNING © 2025</p>
</div>
""", unsafe_allow_html=True)
