"""
main.py
-------
Script untuk download model YOLOv8 dari Google Drive (jika belum ada).
Dijalankan otomatis oleh Streamlit Cloud saat deploy.
"""

import os
import gdown

# Folder tempat model disimpan
os.makedirs("model_out", exist_ok=True)

# URL Google Drive (ganti file_id dengan ID file kamu di Google Drive)
# PENTING: Pastikan ID tidak mengandung parameter tambahan seperti ?hl
file_id = "1a9bScLk9fvVgFxVRhLZesXlq2XZdjsg2"
download_url = f"https://drive.google.com/uc?id={file_id}"
output_path = "model_out/best.pt"

# Ukuran minimum model yang valid (dalam bytes)
# Model YOLOv8 classification biasanya > 5 MB
MIN_MODEL_SIZE = 1_000_000  # 1 MB


def download_model():
    """Download model dari Google Drive."""
    needs_download = False

    if not os.path.exists(output_path):
        print("📥 Model belum ada, mengunduh dari Google Drive...")
        needs_download = True
    else:
        # Cek apakah file corrupt (terlalu kecil)
        file_size = os.path.getsize(output_path)
        if file_size < MIN_MODEL_SIZE:
            print(f"⚠️ Model terlalu kecil ({file_size} bytes), kemungkinan corrupt.")
            print("📥 Mengunduh ulang dari Google Drive...")
            os.remove(output_path)
            needs_download = True
        else:
            print(f"✅ Model sudah ada ({file_size:,} bytes): {output_path}")

    if needs_download:
        try:
            gdown.download(download_url, output_path, quiet=False)
            # Verifikasi hasil download
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size < MIN_MODEL_SIZE:
                    print(f"⚠️ Download selesai tapi file terlalu kecil ({file_size} bytes).")
                    print("   Kemungkinan file ID Google Drive salah atau file tidak public.")
                    print("   Pastikan sharing setting di Google Drive = 'Anyone with the link'")
                else:
                    print(f"✅ Model berhasil diunduh ({file_size:,} bytes): {output_path}")
            else:
                print("❌ Download gagal! File tidak ditemukan.")
        except Exception as e:
            print(f"❌ Error saat download: {e}")
            print("   Pastikan koneksi internet aktif dan file ID Google Drive benar.")


if __name__ == "__main__":
    download_model()
