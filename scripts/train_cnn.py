"""
Entraîne deux CNN (vitesse / autres) → ONNX NCHW pour OpenCV DNN (Java).

  py -3 scripts/train_cnn.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Installez TensorFlow : py -3 -m pip install tensorflow")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "dataset_filtered", "train")
MODELS = os.path.join(ROOT, "models")
IMG_SIZE = 64
MAX_PER_CLASS = 200
EPOCHS = 25
BATCH = 32


def list_classes(train_dir: str, speed_only: bool) -> list[str]:
    names = []
    for name in sorted(os.listdir(train_dir)):
        path = os.path.join(train_dir, name)
        if os.path.isdir(path) and speed_only == name.startswith("Speed_limit_"):
            names.append(name)
    return names


def load_dataset(train_dir: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label_id, label in enumerate(classes):
        folder = os.path.join(train_dir, label)
        count = 0
        for fname in sorted(os.listdir(folder)):
            if count >= MAX_PER_CLASS:
                break
            low = fname.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None or img.size == 0:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # NCHW
            xs.append(img)
            ys.append(label_id)
            count += 1
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def build_cnn(num_classes: int) -> keras.Model:
    inp = keras.Input(shape=(3, IMG_SIZE, IMG_SIZE))
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.06)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(data_format="channels_first")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(data_format="channels_first")(x)
    x = layers.Conv2D(96, 3, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.GlobalAveragePooling2D(data_format="channels_first")(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_onnx(model: keras.Model, path: str) -> None:
    import tf2onnx
    import onnx

    spec = (tf.TensorSpec((None, 3, IMG_SIZE, IMG_SIZE), tf.float32, name="input"),)
    proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(proto, path)
    print("  ONNX :", path)


def train_split(name: str, speed_only: bool) -> None:
    classes = list_classes(TRAIN, speed_only)
    print(f"\n=== CNN {name} ({len(classes)} classes) ===")
    x, y = load_dataset(TRAIN, classes)
    print(f"  Images : {len(y)}")

    model = build_cnn(len(classes))
    cb = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH, validation_split=0.15,
              verbose=1, callbacks=cb)

    os.makedirs(MODELS, exist_ok=True)
    prefix = "speed" if speed_only else "other"
    export_onnx(model, os.path.join(MODELS, f"{prefix}_cnn.onnx"))
    with open(os.path.join(MODELS, f"{prefix}_cnn_labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")


def main() -> None:
    print("TensorFlow", tf.__version__)
    keras.utils.set_random_seed(42)
    train_split("vitesse", True)
    train_split("autres", False)
    print("\nTermine. Relancez SignClassifierCnn en Java.")


if __name__ == "__main__":
    main()
