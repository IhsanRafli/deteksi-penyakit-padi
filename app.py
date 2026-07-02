# app.py — Streamlit App Deteksi Penyakit Padi (YOLOv8 + Gemini AI)

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import base64

# --- Konfigurasi ---
st.set_page_config(
    page_title="🌾 PadiScan AI — Deteksi Penyakit Padi",
    page_icon="🌾",
    layout="wide"
)

MODEL_PATH = "./model_out/best.pt"
IMAGE_TARGET_SIZE = 224

# --- Label kelas ---
CLASS_NAMES = [
    'BLB', 'BPH', 'Brown_Spot', 'False_Smut',
    'Healthy_Plant', 'Hispa', 'Neck_Blast',
    'Sheath_Blight_Rot', 'Stemborer'
]

# --- Data penyakit ---
RECOMMENDATIONS = {
    'BLB': {
        'nama': 'Bacterial Leaf Blight',
        'nama_id': 'Hawar Daun Bakteri',
        'deskripsi': 'Disebabkan bakteri Xanthomonas oryzae. Gejala: bercak kuning kecoklatan dari tepi daun.',
        'rekomendasi': 'Gunakan bakterisida Streptomycin atau Oxytetracycline (Starbacin, Agrimycin). Semprot pagi/sore.',
        'icon': '🦠', 'severity': 'Tinggi', 'severity_en': 'High'
    },
    'BPH': {
        'nama': 'Brown Planthopper',
        'nama_id': 'Wereng Coklat',
        'deskripsi': 'Hama wereng menghisap cairan tanaman, menyebabkan hopperburn (daun kuning/kering mendadak).',
        'rekomendasi': 'Gunakan insektisida Imidacloprid atau Fipronil (Regent, Confidor). Rotasi bahan aktif.',
        'icon': '🪲', 'severity': 'Sangat Tinggi', 'severity_en': 'Critical'
    },
    'Brown_Spot': {
        'nama': 'Brown Spot',
        'nama_id': 'Bercak Coklat',
        'deskripsi': 'Penyakit jamur dengan bercak coklat bulat/oval pada daun padi.',
        'rekomendasi': 'Gunakan fungisida Mancozeb atau Propiconazole (Dithane, Tilt). Semprot saat gejala awal.',
        'icon': '🍂', 'severity': 'Sedang', 'severity_en': 'Medium'
    },
    'False_Smut': {
        'nama': 'False Smut',
        'nama_id': 'Gosong Palsu',
        'deskripsi': 'Jamur menyerang bulir padi, membentuk bola spora hijau-kuning pada gabah.',
        'rekomendasi': 'Gunakan fungisida Carbendazim atau Tricyclazole saat fase malai awal muncul.',
        'icon': '🟢', 'severity': 'Sedang', 'severity_en': 'Medium'
    },
    'Healthy_Plant': {
        'nama': 'Healthy Plant',
        'nama_id': 'Tanaman Sehat',
        'deskripsi': 'Tanaman padi dalam kondisi sehat, tidak terdeteksi gejala penyakit.',
        'rekomendasi': 'Pertahankan kondisi tanaman dengan nutrisi seimbang dan pengairan yang baik.',
        'icon': '🌿', 'severity': 'Sehat', 'severity_en': 'Healthy'
    },
    'Hispa': {
        'nama': 'Hispa',
        'nama_id': 'Penggerek Daun',
        'deskripsi': 'Serangga berduri menggerek jaringan daun, meninggalkan bekas putih transparan.',
        'rekomendasi': 'Gunakan insektisida Chlorpyrifos atau Cypermethrin (Dursban, Cymbush).',
        'icon': '🐛', 'severity': 'Sedang', 'severity_en': 'Medium'
    },
    'Neck_Blast': {
        'nama': 'Neck Blast',
        'nama_id': 'Busuk Leher Malai',
        'deskripsi': 'Blast menyerang leher malai, menyebabkan malai patah dan gabah hampa.',
        'rekomendasi': 'Gunakan fungisida Tricyclazole (Beam, Blastban) saat awal pembentukan malai.',
        'icon': '💀', 'severity': 'Sangat Tinggi', 'severity_en': 'Critical'
    },
    'Sheath_Blight_Rot': {
        'nama': 'Sheath Blight',
        'nama_id': 'Busuk Pelepah',
        'deskripsi': 'Jamur menyerang pelepah daun, membentuk bercak oval hijau keabu-abuan.',
        'rekomendasi': 'Gunakan fungisida Validamycin atau Propiconazole (Validacin, Tilt).',
        'icon': '🍄', 'severity': 'Tinggi', 'severity_en': 'High'
    },
    'Stemborer': {
        'nama': 'Stemborer',
        'nama_id': 'Penggerek Batang',
        'deskripsi': 'Larva ngengat menggerek batang dari dalam, menyebabkan sundep (mati pucuk) dan beluk.',
        'rekomendasi': 'Gunakan insektisida Fipronil, Cartap, atau Abamectin (Regent, Virtako).',
        'icon': '🐛', 'severity': 'Sangat Tinggi', 'severity_en': 'Critical'
    }
}

SEVERITY_COLOR = {
    'Sehat': '#00c853',
    'Sedang': '#ff9800',
    'Tinggi': '#f44336',
    'Sangat Tinggi': '#b71c1c',
    'Tidak diketahui': '#9e9e9e'
}

SEVERITY_BADGE_BG = {
    'Sehat': 'rgba(0,200,83,0.15)',
    'Sedang': 'rgba(255,152,0,0.15)',
    'Tinggi': 'rgba(244,67,54,0.15)',
    'Sangat Tinggi': 'rgba(183,28,28,0.2)',
    'Tidak diketahui': 'rgba(158,158,158,0.1)'
}


# ============================================================
# FUNGSI UTILITAS
# ============================================================

@st.cache_resource
def load_yolo_model(path):
    """Load model YOLOv8 classification."""
    try:
        from ultralytics import YOLO
        if not os.path.exists(path):
            return None, f"Model tidak ditemukan di '{path}'"
        if os.path.getsize(path) < 1_000_000:
            return None, "File model terlalu kecil / corrupt"
        return YOLO(path), None
    except Exception as e:
        return None, str(e)


def predict_image(model, pil_img):
    """Prediksi dengan YOLOv8. Returns (class_key, confidence, all_probs_dict)."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    results = model.predict(source=pil_img, imgsz=IMAGE_TARGET_SIZE, verbose=False)
    result = results[0]
    probs = result.probs
    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf) * 100.0
    names = result.names
    class_key = names.get(top1_idx, f"class_{top1_idx}")
    all_probs = {}
    for idx, pv in enumerate(probs.data.cpu().numpy()):
        all_probs[names.get(idx, f"class_{idx}")] = float(pv) * 100.0
    return class_key, top1_conf, all_probs


def image_to_base64(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_gemini_api_key():
    """Ambil API key dari berbagai sumber."""
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except Exception:
        pass
    return os.environ.get('GEMINI_API_KEY', '')


def get_ai_recommendation(class_key, confidence, all_probs, pil_img=None, luas_lahan=1.0, umur_padi="Vegetatif", cuaca="Cerah"):
    """
    Ambil rekomendasi dari Gemini — kembalikan full text (bukan generator).
    Aman dipanggil berkali-kali karena hasilnya di-cache di session_state oleh caller.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return None, "API Key Gemini belum diset. Masukkan di sidebar."

    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        genai.configure(api_key=api_key)

        info = RECOMMENDATIONS.get(class_key, {})
        disease_name = f"{info.get('nama', class_key)} ({info.get('nama_id', '')})"
        severity = info.get('severity', 'Tidak diketahui')

        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_text = "\n".join([f"  - {n}: {p:.1f}%" for n, p in sorted_probs])

        prompt = f"""Kamu adalah Pakar Agronomi dan Ahli Proteksi Tanaman Padi senior dari Indonesia.
Tugas utamamu adalah mendampingi petani dengan memberikan rekomendasi pengendalian hama dan penyakit padi yang 100% praktis, akurat, dan aplikatif berdasarkan hasil deteksi sistem Computer Vision (YOLOv8).

[DATA DETEKSI & KONDISI LAPANGAN]
- Penyakit/Hama Terdeteksi: {disease_name}
- Tingkat Keparahan (Severity): {severity}
- Luas Lahan Petani: {luas_lahan} Hektar
- Fase Pertumbuhan Padi: {umur_padi}
- Kondisi Cuaca Saat Ini: {cuaca}

[INSTRUKSI WAJIB]
1. Gunakan bahasa Indonesia yang merakyat, santai, namun tetap profesional (gaya bahasa penyuluh pertanian).
2. Jangan berikan teori yang bertele-tele. Petani sedang berada di sawah dan butuh solusi cepat.
3. Hitung dosis secara matematis dan akurat berdasarkan luas lahan {luas_lahan} Hektar. (Asumsi 1 Hektar butuh ~15 tangki semprot ukuran 16 Liter).
4. Kamu WAJIB menyusun jawabanmu EXACTLY sesuai dengan format Markdown di bawah ini, tanpa basa-basi di awal atau di akhir.

[FORMAT JAWABAN MARKDOWN]
## 🔍 Analisis Cepat Lapangan
[Berikan 2 kalimat tajam tentang status bahaya {disease_name} pada fase {umur_padi} di tengah kondisi cuaca {cuaca}]

## 🧪 Rekomendasi Obat (Insektisida/Fungisida)
- **Bahan Aktif Paling Ampuh:** [Sebutkan 1-2 bahan aktif yang tepat]
- **Contoh Merek di Kios Pertanian:** [Sebutkan 2-3 merek dagang yang paling gampang dicari di Indonesia]

## 🧮 Takaran Semprot (Real-time untuk {luas_lahan} Ha)
- **Dosis per Tangki (16 Liter):** [Sebutkan takaran praktis: misal ml/tangki atau sendok makan/tangki]
- **Estimasi Kebutuhan Air:** [Hitung total tangki yang dibutuhkan untuk menyemprot {luas_lahan} Ha]
- **Total Beli Obat:** [Hitung perkiraan total kemasan/botol yang harus dibeli petani hari ini]

## ⚠️ Aturan Main & Pantangan
- **Waktu Eksekusi:** [Kapan harus disemprot mengingat cuaca sedang {cuaca}]
- **Pantangan Fatal:** [Sebutkan 1 hal yang HARUS dihindari agar obat tidak mubazir]"""

        safety = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Kirim gambar jika bukan tanaman sehat (untuk analisis visual lebih akurat)
        if pil_img and class_key != 'Healthy_Plant':
            contents = [prompt, {"mime_type": "image/jpeg", "data": image_to_base64(pil_img)}]
        else:
            contents = prompt

        # ---- SOLUSI ANTI-404: Mencari model yang tersedia secara otomatis ----
        model_candidates = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-latest']
        response = None
        last_err = ""

        for model_name in model_candidates:
            try:
                gemini = genai.GenerativeModel(model_name)
                response = gemini.generate_content(contents, safety_settings=safety)
                if response:
                    break
            except Exception as e:
                last_err = str(e)
                # Jika errornya karena model tidak ditemukan, lanjut coba nama model berikutnya
                if "404" in last_err or "not found" in last_err.lower() or "not supported" in last_err.lower():
                    continue
                else:
                    raise e

        if not response:
            return None, f"Google tidak merespon model gratisan apa pun. Error terakhir: {last_err}"

        return response.text, None

    except ImportError:
        return None, "Library `google-generativeai` belum terinstall. Jalankan: `pip install google-generativeai`"
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            return None, "API Key tidak valid. Pastikan key sudah benar di sidebar."
        if "quota" in err.lower():
            return None, "Quota API Gemini habis. Coba lagi nanti atau upgrade plan."
        return None, f"Error Gemini: {err}"


# ============================================================
# RENDER RESULT CARD
# ============================================================

def render_result_card(class_key, confidence, all_probs, pil_img=None, label=""):
    """Render kartu hasil deteksi + tombol rekomendasi AI."""
    info = RECOMMENDATIONS.get(class_key, {
        'nama': class_key, 'nama_id': '', 'deskripsi': '-',
        'rekomendasi': '-', 'icon': '❓', 'severity': 'Tidak diketahui'
    })

    is_healthy = (class_key == 'Healthy_Plant')
    severity = info.get('severity', 'Tidak diketahui')
    sev_color = SEVERITY_COLOR.get(severity, '#9e9e9e')
    sev_bg = SEVERITY_BADGE_BG.get(severity, 'rgba(158,158,158,0.1)')

    if confidence >= 80:
        conf_color = '#00c853'
        conf_emoji = '✅'
    elif confidence >= 50:
        conf_color = '#ffd740'
        conf_emoji = '⚠️'
    else:
        conf_color = '#ff5252'
        conf_emoji = '❌'

    card_border = 'rgba(76,175,80,0.5)' if is_healthy else 'rgba(244,67,54,0.4)'
    card_bg = (
        'linear-gradient(135deg,#0a3d0a,#1b5e20,#2e7d32)'
        if is_healthy else
        'linear-gradient(135deg,#1a0a0a,#3e1212,#5e1b1b)'
    )

    # Confidence bar width
    bar_w = int(confidence)

    st.markdown(f"""
    <div class="rc-card" style="background:{card_bg}; border:1px solid {card_border};">
        <div class="rc-top">
            <span class="rc-icon">{info['icon']}</span>
            <div class="rc-title-group">
                <div class="rc-name">{info['nama']}</div>
                <div class="rc-name-id">{info['nama_id']}</div>
                <span class="rc-badge" style="background:{sev_bg}; border-color:{sev_color}; color:{sev_color};">
                    🎯 {severity}
                </span>
            </div>
        </div>
        <p class="rc-desc">{info['deskripsi']}</p>
        <div class="rc-conf-wrap">
            <div class="rc-conf-label">
                <span>{conf_emoji} Tingkat Kepercayaan AI</span>
                <span style="color:{conf_color}; font-weight:700; font-size:1.15em;">{confidence:.1f}%</span>
            </div>
            <div class="rc-bar-bg">
                <div class="rc-bar-fill" style="width:{bar_w}%; background:{conf_color};"></div>
            </div>
        </div>
        <div class="rc-rec">
            <span class="rc-rec-label">💊 Rekomendasi Dasar</span>
            <span class="rc-rec-text">{info['rekomendasi']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilitas ekspander
    with st.expander("📊 Detail probabilitas semua kelas"):
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_probs:
            emoji = "🟢" if prob > 50 else "🟡" if prob > 15 else "⚫"
            col_n, col_p = st.columns([3, 1])
            with col_n:
                st.progress(prob / 100)
            with col_p:
                st.write(f"{emoji} **{prob:.1f}%**")
            st.caption(f"  {name}")

    # ---- Tombol Rekomendasi AI ----
    ai_result_key = f"ai_result_{label}_{class_key}"
    ai_error_key  = f"ai_error_{label}_{class_key}"
    ai_done_key   = f"ai_done_{label}_{class_key}"

    st.markdown('<div class="ai-btn-wrap">', unsafe_allow_html=True)

    # Input data lapangan
    st.markdown("##### 📝 Data Lapangan untuk Rekomendasi Akurat")
    col1, col2, col3 = st.columns(3)
    with col1:
        luas_lahan = st.number_input("Luas Lahan (Ha)", min_value=0.1, value=1.0, step=0.1, key=f"luas_{label}_{class_key}")
    with col2:
        umur_padi = st.selectbox("Fase Padi", ["Vegetatif (Anakan)", "Generatif (Bunting/Malai)", "Pemasakan (Menguning)"], key=f"umur_{label}_{class_key}")
    with col3:
        cuaca = st.selectbox("Cuaca Saat Ini", ["Cerah / Panas", "Mendung", "Hujan Ringan", "Hujan Lebat"], key=f"cuaca_{label}_{class_key}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Jika sudah ada hasil sebelumnya, tampilkan langsung
    if ai_result_key in st.session_state:
        ai_text = st.session_state[ai_result_key]
        st.markdown("""
        <div class="ai-result-header">
            <span>🤖</span>
            <span>Rekomendasi Gemini AI</span>
            <span class="ai-live-badge">AI Generated</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="ai-result-box">', unsafe_allow_html=True)
        st.markdown(ai_text)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Tampilkan tombol manual jika belum ada hasil atau jika ingin retry setelah error quota
        if st.button("🤖 Dapatkan Rekomendasi Obat (Gemini AI)", key=f"btn_ai_{label}_{class_key}", use_container_width=True):
            api_key = get_gemini_api_key()
            if not api_key:
                st.error("⚠️ API Key Gemini belum diset. Masukkan di sidebar kiri.")
            else:
                with st.spinner("🧠 Gemini AI sedang meracik rekomendasi obat..."):
                    result, err = get_ai_recommendation(class_key, confidence, all_probs, pil_img, luas_lahan, umur_padi, cuaca)
                    if err:
                        st.error(f"{err}")
                    else:
                        st.session_state[ai_result_key] = result
                        st.rerun()


# ============================================================
# CSS — Responsif (Mobile / Tablet / Desktop)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Reset & Base ---- */
*, *::before, *::after { box-sizing: border-box; font-family: 'Inter', sans-serif; }

/* ---- Header ---- */
.main-header {
    text-align: center;
    padding: clamp(1rem, 3vw, 2rem) clamp(1rem, 4vw, 2rem);
    background: linear-gradient(135deg, rgba(46,125,50,0.12), rgba(38,166,154,0.06));
    border-radius: 20px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(76,175,80,0.25);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(76,175,80,0.08), transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-size: clamp(1.5rem, 4vw, 2.6rem);
    font-weight: 800;
    background: linear-gradient(135deg, #66bb6a, #26a69a, #42a5f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
    line-height: 1.2;
}
.main-header .subtitle {
    color: rgba(255,255,255,0.55);
    font-size: clamp(0.8rem, 2vw, 0.95rem);
    margin: 4px 0 0;
}
.badge-version {
    display: inline-block;
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: #69f0ae;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.03);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.07);
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 10px;
    padding: 0 16px;
    font-weight: 600;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.55);
    transition: all 0.25s;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(46,125,50,0.4);
}

/* ---- Result Card ---- */
.rc-card {
    border-radius: 18px;
    padding: clamp(1rem, 3vw, 1.5rem);
    margin: 0.75rem 0;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
    animation: fadeInUp 0.4s ease;
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.rc-top {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
}
.rc-icon { font-size: clamp(2rem, 5vw, 2.8rem); line-height: 1; }
.rc-title-group { flex: 1; min-width: 0; }
.rc-name {
    font-size: clamp(1rem, 2.5vw, 1.2rem);
    font-weight: 700;
    color: #fff;
    line-height: 1.3;
}
.rc-name-id {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 6px;
}
.rc-badge {
    display: inline-block;
    border: 1px solid;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    white-space: nowrap;
}
.rc-desc {
    color: rgba(255,255,255,0.78);
    font-size: clamp(0.82rem, 2vw, 0.92rem);
    line-height: 1.6;
    margin: 0 0 12px;
}
.rc-conf-wrap { margin: 10px 0; }
.rc-conf-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
    margin-bottom: 6px;
    flex-wrap: wrap;
    gap: 4px;
}
.rc-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.rc-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}
.rc-rec {
    background: rgba(255,255,255,0.06);
    border-left: 3px solid #69f0ae;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.rc-rec-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #69f0ae;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.rc-rec-text {
    color: rgba(255,255,255,0.85);
    font-size: 0.88rem;
    line-height: 1.55;
}

/* ---- AI Button ---- */
.ai-btn-wrap { margin-top: 14px; }
.stButton button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
}

/* ---- AI Result ---- */
.ai-result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: clamp(0.95rem, 2.5vw, 1.1rem);
    color: #fff;
    margin: 18px 0 8px;
    flex-wrap: wrap;
}
.ai-live-badge {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: #90caf9;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: 1px solid rgba(144,202,249,0.3);
}
.ai-result-box {
    background: linear-gradient(135deg, rgba(13,27,42,0.95), rgba(21,27,46,0.95));
    border: 1px solid rgba(100,181,246,0.25);
    border-radius: 14px;
    padding: clamp(1rem, 3vw, 1.5rem);
    line-height: 1.75;
    color: rgba(255,255,255,0.88);
    font-size: clamp(0.85rem, 2vw, 0.95rem);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}
.ai-result-box h2 {
    font-size: clamp(0.95rem, 2.5vw, 1.05rem) !important;
    color: #90caf9 !important;
    margin: 16px 0 6px !important;
    border-bottom: 1px solid rgba(100,181,246,0.15);
    padding-bottom: 4px;
}
.ai-result-box ul, .ai-result-box ol { padding-left: 1.4em; }
.ai-result-box li { margin-bottom: 4px; }
.ai-result-box strong { color: #fff; }

.ai-error-box {
    background: rgba(244,67,54,0.1);
    border: 1px solid rgba(244,67,54,0.35);
    border-radius: 10px;
    padding: 12px 16px;
    color: #ef9a9a;
    font-size: 0.88rem;
    margin-top: 8px;
}

/* ---- Camera Card ---- */
.camera-card {
    background: linear-gradient(135deg, #0d1b2a, #1a1a2e);
    border-radius: 18px;
    padding: clamp(1rem, 3vw, 1.5rem);
    border: 1px solid rgba(100,181,246,0.2);
    margin-bottom: 1rem;
}
.camera-card h3 {
    font-size: clamp(1rem, 2.5vw, 1.2rem);
    font-weight: 700;
    color: #fff;
    margin: 0 0 6px;
}
.camera-card p {
    color: rgba(255,255,255,0.55);
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
}

/* ---- Empty State ---- */
.empty-state {
    text-align: center;
    padding: clamp(2rem, 5vw, 3.5rem) clamp(1rem, 4vw, 2rem);
    background: rgba(255,255,255,0.02);
    border-radius: 16px;
    border: 1px dashed rgba(255,255,255,0.1);
}
.empty-state .es-icon { font-size: clamp(3rem, 8vw, 4.5rem); margin-bottom: 12px; }
.empty-state h4 { color: rgba(255,255,255,0.45); margin: 0 0 8px; font-size: clamp(1rem, 2.5vw, 1.1rem); }
.empty-state p { color: rgba(255,255,255,0.3); font-size: clamp(0.8rem, 2vw, 0.9rem); line-height: 1.5; }

/* ---- Tips box ---- */
.tip-box {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.82em;
    color: rgba(255,255,255,0.45);
    border: 1px solid rgba(255,255,255,0.06);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1b2a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.3);
    margin: 16px 0 8px;
}
.api-status-ok {
    background: rgba(0,200,83,0.12);
    border: 1px solid rgba(0,200,83,0.35);
    border-radius: 8px;
    padding: 6px 12px;
    color: #69f0ae;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 6px;
}
.api-status-missing {
    background: rgba(255,152,0,0.1);
    border: 1px solid rgba(255,152,0,0.35);
    border-radius: 8px;
    padding: 6px 12px;
    color: #ffd740;
    font-size: 0.82rem;
    margin-top: 6px;
}

/* ---- Responsive breakpoints ---- */
@media (max-width: 768px) {
    /* Di HP, kolom kamera & hasil jadi stack vertikal otomatis oleh Streamlit */
    .rc-top { gap: 8px; }
    .rc-icon { font-size: 2rem; }
    .ai-result-box { padding: 0.9rem; }
    .stTabs [data-baseweb="tab"] { padding: 0 10px; font-size: 0.82rem; height: 40px; }
}
@media (max-width: 480px) {
    .main-header { padding: 1rem; }
    .main-header h1 { font-size: 1.4rem; }
}

/* ---- Misc Streamlit overrides ---- */
.stExpander { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.07) !important; }
.stSpinner > div { color: #69f0ae !important; }
div[data-testid="stFileUploader"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">🌾 PadiScan AI v2.0</div>', unsafe_allow_html=True)
    st.markdown("Deteksi penyakit padi berbasis **YOLOv8** + **Google Gemini AI**")

    st.markdown("---")

    # --- Setup Gemini API Key ---
    st.markdown('<div class="sidebar-section-title">🤖 Gemini AI Setup</div>', unsafe_allow_html=True)
    st.markdown("Untuk fitur **Rekomendasi AI Mendalam**, masukkan API Key:")

    current_key = get_gemini_api_key()
    key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        value="",
        help="Dapatkan gratis di aistudio.google.com/app/apikey"
    )
    if key_input:
        os.environ['GEMINI_API_KEY'] = key_input.strip()
        st.markdown('<div class="api-status-ok">✅ API Key aktif di sesi ini</div>', unsafe_allow_html=True)
    elif current_key:
        st.markdown('<div class="api-status-ok">✅ API Key sudah tersedia</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status-missing">⚠️ API Key belum diset</div>', unsafe_allow_html=True)

    st.markdown("[🔑 Dapatkan API Key Gratis →](https://aistudio.google.com/app/apikey)", unsafe_allow_html=False)

    st.markdown("---")

    # --- Info Model ---
    st.markdown('<div class="sidebar-section-title">⚙️ Info Model</div>', unsafe_allow_html=True)
    st.markdown(f"- Model: `YOLOv8 Classification`\n- Path: `{MODEL_PATH}`\n- Input: `{IMAGE_TARGET_SIZE}×{IMAGE_TARGET_SIZE}px`\n- Kelas: `{len(CLASS_NAMES)}`")

    st.markdown("---")

    # --- Daftar Penyakit ---
    st.markdown('<div class="sidebar-section-title">📋 Daftar Penyakit</div>', unsafe_allow_html=True)
    sev_icons = {'Sehat': '🟢', 'Sedang': '🟡', 'Tinggi': '🟠', 'Sangat Tinggi': '🔴'}
    for cls in CLASS_NAMES:
        info = RECOMMENDATIONS.get(cls, {})
        si = sev_icons.get(info.get('severity', ''), '⚪')
        st.markdown(f"{si} {info.get('icon', '')} **{info.get('nama', cls)}**  \n&nbsp;&nbsp;&nbsp;&nbsp;*{info.get('nama_id', '')}*")

    st.markdown("---")
    st.caption("© 2024 PadiScan AI · YOLOv8 + Gemini")


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <div class="badge-version">🌾 PadiScan AI</div>
    <h1>Deteksi Penyakit Daun Padi</h1>
    <p class="subtitle">Upload foto atau gunakan kamera — AI akan mendeteksi penyakit & memberikan rekomendasi penanganan</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
yolo_model, model_err = load_yolo_model(MODEL_PATH)
if yolo_model is None:
    st.error(f"⚠️ Gagal memuat model: {model_err}")
    st.info("Pastikan model sudah di-train dan tersedia di `model_out/best.pt`")
    st.code("python train_rice_model.py", language="bash")
    st.stop()


# ============================================================
# PILIH MODE DETEKSI (PENGGANTI TABS)
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
pilihan_mode = st.radio(
    "Pilih Mode Input:",
    ["📁  Upload Gambar", "📸  Foto Langsung"],
    horizontal=True,
    label_visibility="collapsed"
)


# ================================================================
# TAB 1: UPLOAD GAMBAR
# ================================================================
if pilihan_mode == "📁  Upload Gambar":
    uploaded_files = st.file_uploader(
        "Pilih satu atau beberapa foto daun padi (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Bisa upload lebih dari satu gambar sekaligus"
    )

    if uploaded_files:
        st.markdown(f"**📊 Hasil Analisis — {len(uploaded_files)} gambar**")
        cols = st.columns(min(len(uploaded_files), 2))
        for i, uf in enumerate(uploaded_files):
            try:
                pil_img = Image.open(io.BytesIO(uf.read()))
                with cols[i % 2]:
                    st.image(pil_img, use_container_width=True, caption=f"🖼 {uf.name}")
                    with st.spinner("🔍 Menganalisis..."):
                        class_key, confidence, all_probs = predict_image(yolo_model, pil_img)
                    render_result_card(class_key, confidence, all_probs, pil_img=pil_img, label=f"up{i}")
            except Exception as e:
                st.error(f"❌ Error memproses `{uf.name}`: {e}")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="es-icon">📁</div>
            <h4>Belum ada gambar diupload</h4>
            <p>Drag & drop atau klik area di atas untuk memilih<br>foto daun padi dari perangkat kamu</p>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# TAB 2: KAMERA LANGSUNG
# ================================================================
elif pilihan_mode == "📸  Foto Langsung":
    st.markdown("""
    <div class="camera-card">
        <h3>📸 Ambil Foto Langsung</h3>
        <p>Foto akan dianalisis <strong style="color:#69f0ae;">otomatis</strong> begitu kamu mengambil gambar.
        Pastikan daun padi mengisi sebagian besar frame dan pencahayaan cukup.</p>
    </div>
    """, unsafe_allow_html=True)

    col_cam, col_res = st.columns([1, 1], gap="medium")

    with col_cam:
        st.markdown("##### 📷 Kamera")
        # PENTING: JANGAN tambahkan st.image() setelah st.camera_input()
        # di container/kolom yang sama — akan menyebabkan React DOM conflict.
        camera_image = st.camera_input(
            "Arahkan kamera ke daun padi, lalu klik tombol kamera",
            help="Foto langsung dianalisis setelah diambil"
        )
        st.markdown("""
        <div class="tip-box">
            💡 <b>Tips:</b> Jarak 20–30 cm &nbsp;·&nbsp; Cahaya cukup &nbsp;·&nbsp; Daun isi >70% frame &nbsp;·&nbsp; Tidak blur
        </div>
        """, unsafe_allow_html=True)

    with col_res:
        st.markdown("##### 📊 Hasil Deteksi")

        if camera_image is not None:
            pil_cam = Image.open(camera_image)
            # Preview di kolom hasil (bukan kolom kamera) — aman dari DOM conflict
            st.image(pil_cam, caption="📸 Foto yang Diambil", use_container_width=True)

            with st.spinner("🔍 Menganalisis gambar..."):
                try:
                    class_key, confidence, all_probs = predict_image(yolo_model, pil_cam)
                except Exception as e:
                    st.error(f"❌ Error analisis: {e}")
                    st.stop()

            render_result_card(class_key, confidence, all_probs, pil_img=pil_cam, label="cam")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="es-icon">📷</div>
                <h4>Belum ada foto</h4>
                <p>Ambil foto menggunakan kamera di sebelah kiri.<br>
                Analisis berjalan <strong style="color:#69f0ae;">otomatis</strong> setelah foto diambil.</p>
            </div>
            """, unsafe_allow_html=True)

    # Tips Expandable
    with st.expander("💡 Tips Pengambilan Gambar yang Baik"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**✅ Lakukan ini:**\n- Fokus pada daun\n- Cahaya terang & merata\n- Daun isi >70% frame\n- Gambar tajam")
        with c2:
            st.markdown("**❌ Hindari ini:**\n- Gambar gelap\n- Terlalu jauh\n- Gambar blur / goyang\n- Latar terlalu ramai")
        with c3:
            st.markdown("**📐 Posisi Ideal:**\n- Jarak 20–30 cm\n- Sudut tegak lurus\n- Satu daun per foto\n- Saat cuaca cerah")
