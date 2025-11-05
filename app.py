# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# --- Konfigurasi ---
st.set_page_config(page_title="Deteksi Penyakit Padi", layout="wide")
MODEL_PATH = "./model_out/rice_disease_model.keras"  # sesuaikan jika perlu
IMAGE_TARGET_SIZE = (150, 150)

# --- Label kelas (harus sama dengan model saat training) ---
CLASS_NAMES = [
    'BLB',
    'BPH',
    'Brown_Spot',
    'False_Smut',
    'Healthy_Plant',
    'Hispa',
    'Neck_Blast',
    'Sheath_Blight_Rot',
    'Stemborer'
]

# --- Rekomendasi pestisida/insektisida per penyakit ---
RECOMMENDATIONS = {
    'BLB': {
        'nama': 'Bacterial Leaf Blight (Hawar Daun Bakteri)',
        'rekomendasi': 'Gunakan bakterisida berbahan aktif Streptomycin atau Oxytetracycline (misal: Starbacin, Agrimycin). Semprot pagi atau sore hari.'
    },
    'BPH': {
        'nama': 'Brown Planthopper (Wereng Coklat)',
        'rekomendasi': 'Gunakan insektisida berbahan aktif Imidacloprid atau Fipronil (misal: Regent, Confidor). Jangan gunakan satu bahan aktif terus-menerus.'
    },
    'Brown_Spot': {
        'nama': 'Brown Spot (Bercak Coklat)',
        'rekomendasi': 'Gunakan fungisida berbahan aktif Mancozeb atau Propiconazole (misal: Dithane, Tilt). Semprot saat gejala awal muncul.'
    },
    'False_Smut': {
        'nama': 'False Smut (Bunt Palsu)',
        'rekomendasi': 'Gunakan fungisida berbahan aktif Carbendazim atau Tricyclazole. Lakukan penyemprotan saat fase malai mulai muncul.'
    },
    'Healthy_Plant': {
        'nama': 'Tanaman Sehat 🌿',
        'rekomendasi': 'Tidak perlu penyemprotan. Pertahankan kondisi tanaman, berikan nutrisi seimbang dan pengairan yang baik.'
    },
    'Hispa': {
        'nama': 'Hispa (Serangan Serangga Penggerek Daun)',
        'rekomendasi': 'Gunakan insektisida berbahan aktif Chlorpyrifos atau Cypermethrin (misal: Dursban, Cymbush).'
    },
    'Neck_Blast': {
        'nama': 'Neck Blast (Penyakit Busuk Leher Malai)',
        'rekomendasi': 'Gunakan fungisida berbahan aktif Tricyclazole (misal: Beam, Blastban). Lakukan saat awal pembentukan malai.'
    },
    'Sheath_Blight_Rot': {
        'nama': 'Sheath Blight (Busuk Pelepah)',
        'rekomendasi': 'Gunakan fungisida berbahan aktif Validamycin atau Propiconazole (misal: Validacin, Tilt).'
    },
    'Stemborer': {
        'nama': 'Stemborer (Penggerek Batang)',
        'rekomendasi': 'Gunakan insektisida berbahan aktif Fipronil, Cartap, atau Abamectin (misal: Regent, Virtako).'
    }
}

# --- Fungsi utilitas ---
@st.cache_resource
def load_tf_model(path):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model dari '{path}': {e}")
        return None

def preprocess_pil_image(pil_img, target_size):
    # Pastikan RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img_resized = pil_img.resize(target_size)
    x = keras_image.img_to_array(img_resized)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(model, pil_img):
    x = preprocess_pil_image(pil_img, IMAGE_TARGET_SIZE)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds)) * 100.0
    class_key = CLASS_NAMES[idx]
    return class_key, confidence, preds[0]

# --- UI ---
st.title("🔎 Deteksi Penyakit Daun Padi")
st.write("Upload foto daun padi (jpg/png). Bisa upload satu atau beberapa gambar sekaligus.")

with st.sidebar:
    st.header("Pengaturan")
    st.write("Model path:")
    st.code(MODEL_PATH)

# Muat model
model = load_tf_model(MODEL_PATH)
if model is None:
    st.stop()

uploaded_files = st.file_uploader("Pilih gambar (boleh banyak)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            img_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(img_bytes))

            # Tampilkan dan prediksi
            col = cols[i % 2]
            with col:
                st.image(pil_img, use_column_width='always', caption=uploaded_file.name)
                with st.spinner("Memprediksi..."):
                    class_key, confidence, raw_preds = predict_image(model, pil_img)

                info = RECOMMENDATIONS.get(class_key, {"nama": class_key, "rekomendasi": "Tidak ada rekomendasi tersedia."})

                st.markdown(f"**🟢 Hasil:** {info['nama']}")
                st.markdown(f"**📊 Confidence:** {confidence:.2f}%")
                st.markdown(f"**💡 Rekomendasi:** {info['rekomendasi']}")

                # Opsi tampilkan probabilitas semua kelas (collapsible)
                with st.expander("Lihat probabilitas semua kelas"):
                    for idx, name in enumerate(CLASS_NAMES):
                        prob = raw_preds[idx] * 100
                        st.write(f"- {name}: {prob:.2f}%")
        except Exception as e:
            st.error(f"Error memproses {uploaded_file.name}: {e}")

else:
    st.info("Belum ada gambar diupload — silakan upload gambar menggunakan tombol di atas.")

st.markdown("---")
st.caption("Catatan: Pastikan `CLASS_NAMES` cocok urutannya dengan output model Anda dan ukuran target gambar sama seperti waktu training.")
