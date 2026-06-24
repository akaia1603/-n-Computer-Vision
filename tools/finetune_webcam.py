"""
tools/finetune_webcam.py
========================
Fine-tune baseline CNN với dữ liệu webcam đã thu thập.

Cách dùng:
    python tools/finetune_webcam.py                  # nhanh: chỉ train data webcam mới
    python tools/finetune_webcam.py --full           # chậm: train full data + webcam
    python tools/finetune_webcam.py --epochs 5 --lr 0.0001
"""

import os, sys, argparse
import tensorflow as tf
import numpy as np
import cv2
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import RAF_MEAN, RAF_STD, preprocess_raf_np

EMOTIONS = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness",
            5: "Sadness", 6: "Anger", 7: "Neutral"}


def load_webcam_data(data_dir):
    images, labels = [], []
    for cid in EMOTIONS:
        paths = glob(os.path.join(data_dir, str(cid), "webcam_*.jpg"))
        for p in paths:
            img = cv2.imread(p)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))
            img = preprocess_raf_np(img)
            images.append(img)
            labels.append(cid - 1)
    if not images:
        return None, None
    return np.array(images), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--full", action="store_true",
                        help="Train full RAF-DB dataset (chậm ~15-30p)")
    args = parser.parse_args()

    model_path = "outputs/models/baseline_cnn.keras"
    if not os.path.exists(model_path):
        print(f" Không tìm thấy {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f" Loaded: {model_path}")

    if args.full:
        from utils.data_loader import create_datasets
        train_ds, val_ds, test_ds, class_weights, info = create_datasets(
            data_dir="data/DATASET", img_size=(100, 100),
            batch_size=args.batch_size, augment_train=True,
        )
        val_data = val_ds
    else:
        X, y = load_webcam_data("data/DATASET/train")
        if X is None:
            print(" Không tìm thấy data webcam nào!")
            print(" Chạy tools/capture_webcam_data.py trước để thu thập.")
            return
        print(f" Data webcam: {len(X)} ảnh")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(256).batch(args.batch_size).prefetch(2)
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_data = val_data.batch(args.batch_size).prefetch(2)
        class_weights = None

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="outputs/models/baseline_cnn_finetuned.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    model.fit(train_ds, epochs=args.epochs, validation_data=val_data,
              class_weight=class_weights, callbacks=callbacks)

    model.save("outputs/models/baseline_cnn_finetuned.keras")
    print(f"\n Saved: outputs/models/baseline_cnn_finetuned.keras")
    print(" Dùng app -> chọn model Keras, nó sẽ tự load bản finetuned")


if __name__ == "__main__":
    main()
