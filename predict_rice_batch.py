import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Daftar label kelas (harus sama dengan saat training) ---
class_names = [
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
recommendations = {
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

# --- Load model hasil training ---
model_path = './model_out/rice_disease_model.keras'
model = load_model(model_path)

# --- Folder berisi foto-foto uji ---
test_folder = './test_images'

# --- Ambil semua file gambar di folder ---
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("❌ Tidak ada gambar di folder test_images. Masukkan dulu beberapa foto daun padi untuk diuji.")
    exit()

print(f"\n📂 Jumlah gambar terdeteksi: {len(image_files)}\n")

# --- Loop prediksi setiap gambar ---
for img_name in image_files:
    img_path = os.path.join(test_folder, img_name)

    # Preprocessing gambar
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalisasi

    # Prediksi
    predictions = model.predict(x, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100

    # --- Ambil rekomendasi pestisida ---
    info = recommendations[predicted_class]

    # --- Output hasil prediksi ---
    print("🖼️ Gambar:", img_name)
    print(f"   🌾 Penyakit Terdeteksi : {info['nama']}")
    print(f"   📊 Tingkat Kepercayaan : {confidence:.2f}%")
    print(f"   💡 Rekomendasi          : {info['rekomendasi']}\n")

print("✅ Semua gambar telah diprediksi dan diberikan rekomendasi pestisida.\n")
