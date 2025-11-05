import os
import gdown

# Folder tempat model disimpan
os.makedirs("model_out", exist_ok=True)

# URL Google Drive (ubah dengan ID kamu sendiri)
file_id = "1a9bScLk9fvVgFxVRhLZesXlq2XZdjsg2?hl"
download_url = f"https://drive.google.com/uc?id={file_id}"
output_path = "model_out/rice_disease_model.keras"

# Cek apakah model sudah ada, kalau belum download
if not os.path.exists(output_path):
    print("📥 Mengunduh model dari Google Drive...")
    gdown.download(download_url, output_path, quiet=False)
else:
    print("✅ Model sudah ada di folder lokal.")
