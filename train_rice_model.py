import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === 1. Pastikan path dataset ===
base_dir = "./dataset/Rahman2020"

train_dir = os.path.join(base_dir, "")
val_dir = os.path.join(base_dir, "")

# === 2. Data augmentation dan normalisasi ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 80% train, 20% validasi
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)
print(f"📊 Jumlah kelas terdeteksi: {num_classes}")
print("Daftar kelas:", train_generator.class_indices)

# === 3. Bangun model CNN ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    # Output sesuai jumlah kelas
    layers.Dense(num_classes, activation='softmax')
])

# === 4. Kompilasi model ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 5. Training ===
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# === 6. Simpan model ===
output_dir = "./model_out"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "rice_disease_model.keras")
model.save(model_path)

print(f"\n✅ Model berhasil disimpan di {model_path}")
print("Selesai training dan penyimpanan model 🌾✨")
