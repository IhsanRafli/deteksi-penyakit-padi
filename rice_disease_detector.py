
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Core lightweight deps we expect to exist in many envs
import numpy as np
from PIL import Image

# Try to import optional heavy deps (TensorFlow, OpenCV). If missing,
# keep flags and provide fallback behavior.
TF_AVAILABLE = True
CV2_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing import image as keras_image
except Exception as e:
    TF_AVAILABLE = False
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    keras_image = None  # type: ignore

try:
    import cv2
except Exception:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore


# ---------------------- Simple recommendations ----------------------
RECOMMENDATIONS: Dict[str, str] = {
    "healthy": "Tanaman tampak sehat. Lanjutkan pemeliharaan rutin dan pemantauan.",
    "rice_blast": "Gejala Blast. Terapkan IPM: pangkas bagian terinfeksi, kurangi aplikasi N berlebih. Jika perlu pakai fungisida sesuai label.",
    "bacterial_leaf_blight": "Gejala Bacterial Leaf Blight. Tingkatkan sanitasi, gunakan benih sehat. Konsultasikan dengan penyuluh; kontrol kimia terbatas.",
    "brown_spot": "Gejala Brown Spot. Atur drainase dan nutrisi. Pertimbangkan fungisida sesuai anjuran lokal.",
    "sheath_blight": "Gejala Sheath Blight. Rotasi tanaman, sanitasi, dan aplikasi fungisida bila perlu.",
}


# ---------------------- Helper utilities ----------------------

def informative_missing_tf_message() -> str:
    return (
        "TensorFlow tidak tersedia di lingkungan ini.\n"
        "Untuk menjalankan training atau inference asli, install TensorFlow: \n"
        "  pip install tensorflow\n\n"
        "Jika Anda berada di lingkungan terbatas (colab/lokal), pastikan Anda memilih runtime yang memiliki TensorFlow." 
    )


def load_image_pil(path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load image with PIL and return float32 array scaled to [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


# ---------------------- Fallback mock predictor ----------------------

def mock_predict_from_filename_or_color(image_paths: List[str], top_k: int = 3) -> List[Dict]:
    """
    Very small heuristic predictor used when TensorFlow is not available.

    Heuristics (in order):
    1. If filename contains a keyword (blast, blight, brown, sheath), return that class.
    2. Else compute mean green channel; very low green -> likely diseased -> return 'brown_spot' as default.
    3. Else return 'healthy'.

    This is ONLY for testing/integration when TF is missing and DOES NOT
    represent a real classifier.
    """
    results = []
    for p in image_paths:
        name = Path(p).stem.lower()
        preds: List[Tuple[str, float]] = []
        if "blast" in name:
            preds = [("rice_blast", 0.95), ("healthy", 0.03), ("brown_spot", 0.02)]
        elif "blight" in name:
            preds = [("bacterial_leaf_blight", 0.93), ("healthy", 0.04), ("brown_spot", 0.03)]
        elif "brown" in name:
            preds = [("brown_spot", 0.9), ("healthy", 0.06), ("rice_blast", 0.04)]
        elif "sheath" in name:
            preds = [("sheath_blight", 0.9), ("healthy", 0.07), ("brown_spot", 0.03)]
        else:
            # Try naive color heuristic if file exists
            try:
                arr = load_image_pil(p)
                mean_g = float(np.mean(arr[:, :, 1]))
                mean_r = float(np.mean(arr[:, :, 0]))
                mean_b = float(np.mean(arr[:, :, 2]))
                # If green is dominant -> likely healthy
                if mean_g > mean_r and mean_g > mean_b and mean_g > 0.35:
                    preds = [("healthy", 0.9), ("rice_blast", 0.05), ("brown_spot", 0.05)]
                else:
                    preds = [("brown_spot", 0.7), ("rice_blast", 0.2), ("healthy", 0.1)]
            except FileNotFoundError:
                preds = [("healthy", 0.6), ("brown_spot", 0.2), ("rice_blast", 0.2)]
            except Exception:
                preds = [("healthy", 0.6), ("brown_spot", 0.2), ("rice_blast", 0.2)]

        top_label = preds[0][0]
        recommendation = RECOMMENDATIONS.get(top_label, "Tidak ada rekomendasi spesifik. Konsultasikan ke penyuluh.")

        # For mock mode we do not create Grad-CAM images. Instead we set gradcam=None.
        results.append({
            "image": p,
            "predictions": preds,
            "gradcam": None,
            "recommendation": recommendation,
            "model_type": "mock"
        })
    return results


# ---------------------- TensorFlow-based implementations ----------------------

if TF_AVAILABLE:
    # Define real functions only when TF is available
    def build_datasets(data_dir: str, img_size: int = 224, batch_size: int = 32, val_split: float = 0.2, seed: int = 123):
        data_dir = Path(data_dir)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="training",
            seed=seed,
            image_size=(img_size, img_size),
            batch_size=batch_size
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="validation",
            seed=seed,
            image_size=(img_size, img_size),
            batch_size=batch_size
        )
        class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, class_names


    def build_model(num_classes: int, img_size: int = 224, base_trainable: bool = False):
        inputs = keras.Input(shape=(img_size, img_size, 3))
        x = layers.Rescaling(1.0 / 255)(inputs)
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        x = data_augmentation(x)

        base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
        base_model.trainable = base_trainable

        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        return model


    def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, pred_index: int = None) -> np.ndarray:
        grad_model = tf.keras.models.Model([
            model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()


    def save_and_display_gradcam(img_path: str, heatmap: np.ndarray, cam_path: str = "gradcam.jpg", alpha: float = 0.4):
        # Use OpenCV if available; otherwise use PIL to blend
        if CV2_AVAILABLE and cv2 is not None:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_uint = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
            superimposed = heatmap_color * alpha + img
            superimposed = np.uint8(superimposed)
            cv2.imwrite(cam_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
            return cam_path
        else:
            # PIL-based blending
            img = Image.open(img_path).convert("RGB")
            img_arr = np.asarray(img).astype(np.uint8)
            heatmap_resized = np.asarray(Image.fromarray(np.uint8(255 * heatmap)).resize((img_arr.shape[1], img_arr.shape[0])))
            # Map heatmap to RGB colormap using a simple jet-like mapping
            cmap = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                cmap[i] = [min(255, 4*(i-64)), min(255, 4*i), min(255, 4*(192-i))]
            heatmap_rgb = cmap[heatmap_resized]
            superimposed = (heatmap_rgb * alpha + img_arr).astype(np.uint8)
            Image.fromarray(superimposed).save(cam_path)
            return cam_path


    def train(data_dir: str, output_dir: str, img_size: int = 224, batch_size: int = 32, epochs: int = 10):
        os.makedirs(output_dir, exist_ok=True)
        train_ds, val_ds, class_names = build_datasets(data_dir, img_size=img_size, batch_size=batch_size)

        num_classes = len(class_names)
        model = build_model(num_classes, img_size=img_size, base_trainable=False)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        checkpoint_path = os.path.join(output_dir, "best_model.h5")
        callbacks = [
            keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

        # Fine-tune
        # Attempt to find base_model within the model.layers -- safe guard
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'name') and 'mobilenet' in layer.name.lower():
                base_model = layer
                break
        if base_model is None:
            # best-effort fallback: set whole model trainable
            model.trainable = True
        else:
            base_model.trainable = True
            for l in getattr(base_model, 'layers', [])[:100]:
                l.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        fine_tune_epochs = 5
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks)

        # Save model and classes
        saved_model_path = os.path.join(output_dir, "saved_model")
        model.save(saved_model_path)
        with open(os.path.join(output_dir, "class_names.json"), "w") as f:
            json.dump(class_names, f)

        print("Training selesai. Model disimpan di:", saved_model_path)
        return saved_model_path, class_names


    def load_model_and_classes(model_dir: str):
        model = tf.keras.models.load_model(os.path.join(model_dir, "saved_model"))
        with open(os.path.join(model_dir, "class_names.json"), "r") as f:
            class_names = json.load(f)
        return model, class_names


    def preprocess_image_for_model(img_path: str, img_size: int = 224) -> np.ndarray:
        img = keras_image.load_img(img_path, target_size=(img_size, img_size))
        arr = keras_image.img_to_array(img)
        arr = arr / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr


    def predict_images_with_tf(model_dir: str, image_paths: List[str], img_size: int = 224, top_k: int = 3) -> List[Dict]:
        model, class_names = load_model_and_classes(model_dir)
        results = []

        # find last conv layer name safely
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            # check nested model case
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            if hasattr(layer, 'layers'):
                for sub in reversed(getattr(layer, 'layers')):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub.name
                        break
                if last_conv_layer_name:
                    break

        for img_path in image_paths:
            arr = preprocess_image_for_model(img_path, img_size=img_size)
            preds = model.predict(arr)
            top_indices = preds[0].argsort()[-top_k:][::-1]
            top = [(class_names[i], float(preds[0][i])) for i in top_indices]

            # Grad-CAM for top-1 if possible
            cam_path = None
            try:
                if last_conv_layer_name:
                    heatmap = make_gradcam_heatmap(arr, model, last_conv_layer_name, pred_index=int(top_indices[0]))
                    cam_path = f"{Path(img_path).stem}_gradcam.jpg"
                    save_and_display_gradcam(img_path, heatmap, cam_path)
            except Exception as e:
                # Non-fatal: continue without gradcam
                cam_path = None

            top_label = top[0][0]
            recommendation = RECOMMENDATIONS.get(top_label, "Tidak ada rekomendasi spesifik. Konsultasikan ke penyuluh.")

            results.append({
                "image": img_path,
                "predictions": top,
                "gradcam": cam_path,
                "recommendation": recommendation,
                "model_type": "tensorflow"
            })
        return results


# ---------------------- Public API functions ----------------------

def predict_images(model_dir: str, image_paths: List[str], img_size: int = 224, top_k: int = 3) -> List[Dict]:
    """Public wrapper: uses TF if available and model exists, otherwise mock predictor."""
    if TF_AVAILABLE:
        # check for saved model
        saved_model_folder = os.path.join(model_dir, "saved_model")
        if os.path.isdir(saved_model_folder):
            try:
                return predict_images_with_tf(model_dir, image_paths, img_size=img_size, top_k=top_k)
            except Exception as e:
                print(f"Terjadi error pada prediksi TensorFlow: {e}")
                print("Falling back to mock predictor.")
                return mock_predict_from_filename_or_color(image_paths, top_k=top_k)
        else:
            print("Model TensorFlow tidak ditemukan di", saved_model_folder)
            print("Menjalankan mock predictor sebagai gantinya.")
            return mock_predict_from_filename_or_color(image_paths, top_k=top_k)
    else:
        print(informative_missing_tf_message())
        print("Menjalankan mock predictor (filename/color heuristics).")
        return mock_predict_from_filename_or_color(image_paths, top_k=top_k)


def train_wrapper(data_dir: str, output_dir: str, img_size: int = 224, batch_size: int = 32, epochs: int = 10):
    if not TF_AVAILABLE:
        raise RuntimeError(informative_missing_tf_message())
    return train(data_dir, output_dir, img_size=img_size, batch_size=batch_size, epochs=epochs)  # type: ignore


# ---------------------- Self-test (lightweight) ----------------------

def selftest() -> int:
    """Run a few lightweight tests that do not require TensorFlow.

    Returns exit code (0 success; non-zero failure).
    """
    print("Menjalankan selftest ringan...")
    tmp_dir = Path("._rdd_selftest")
    tmp_dir.mkdir(exist_ok=True)

    # Create tiny images to test mock predictor behavior
    colors = {
        "healthy_test.jpg": (30, 200, 30),  # green-ish
        "brown_test.jpg": (150, 100, 70),   # brown-ish
        "blast_test.jpg": (10, 10, 10)
    }
    created = []
    try:
        for name, rgb in colors.items():
            path = tmp_dir / name
            img = Image.new('RGB', (64, 64), color=rgb)
            img.save(path)
            created.append(str(path))

        # Run mock predictor explicitly (works even if TF is present)
        results = mock_predict_from_filename_or_color(created)
        print(json.dumps(results, indent=2, ensure_ascii=False))

        # Minimal assertions
        assert results[0]['predictions'][0][0] in ('healthy', 'brown_spot', 'rice_blast', 'bacterial_leaf_blight', 'sheath_blight')
        assert isinstance(results[0]['predictions'][0][1], float)

        print("Selftest selesai: OK")
        return 0
    except AssertionError as err:
        print("Selftest gagal: assertion error", err)
        return 2
    except Exception as e:
        print("Selftest gagal dengan error:", e)
        return 3
    finally:
        # Cleanup
        for f in tmp_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass


# ---------------------- CLI ----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Rice disease detector: train & predict (TF optional)")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--data_dir', type=str, help='Path to dataset (for training)')
    parser.add_argument('--output_dir', type=str, default='./model_out', help='Where to save model & artifacts')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, help='Directory where model saved (for predict). If omitted uses --output_dir')
    parser.add_argument('--image_paths', nargs='*', help='Path(s) to image(s) to predict')
    parser.add_argument('--selftest', action='store_true', help='Run a lightweight self-test (no TF required)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.selftest:
        code = selftest()
        sys.exit(code)

    if args.mode == 'train':
        if not args.data_dir:
            print('Error: untuk mode train, --data_dir harus disediakan')
            sys.exit(2)
        try:
            train_wrapper(args.data_dir, args.output_dir, img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs)
        except RuntimeError as e:
            print(e)
            sys.exit(3)

    elif args.mode == 'predict':
        model_dir = args.model_dir if args.model_dir else args.output_dir
        if not args.image_paths:
            print('Error: untuk mode predict, berikan minimal satu path image lewat --image_paths')
            sys.exit(2)
        results = predict_images(model_dir, args.image_paths, img_size=args.img_size)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("Tidak ada perintah. Gunakan --mode train|predict atau --selftest")


if __name__ == '__main__':
    main()
