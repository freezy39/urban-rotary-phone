# camp_classifier.py
# Purpose: train a small TensorFlow image classifier for camping furniture colors or styles
# How to run:
#   python camp_classifier.py --data_dir "camp_furniture" --epochs 10
# Optional quick test after training:
#   python camp_classifier.py --data_dir "camp_furniture" --epochs 10 --predict "camp_furniture/chair/your_image.jpg"
#
# Folder layout expected:
#   camp_furniture/
#     chair/
#       img1.jpg
#       ...
#     cot/
#       ...
#     table/
#       ...
#
# Notes:
# - Keep class names as folder names you like seeing in reports
# - Script prints class names and saves model to camp_classifier.keras and labels.txt

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_datasets(data_dir: str, img_size=(224, 224), batch_size=32, val_split=0.25, seed=123):
    # Load raw datasets
    train_raw = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_raw = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    # Capture class names now, before mapping
    class_names = train_raw.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Preprocess and lightweight augmentation
    AUTOTUNE = tf.data.AUTOTUNE

    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
        ],
        name="augment",
    )

    def prep_train(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = augment(x, training=True)
        return x, y

    def prep_val(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    train_ds = train_raw.map(prep_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_raw.map(prep_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names, num_classes


def build_model(input_shape, num_classes):
    # Simple CNN that trains fast on CPU
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(class_names, path="labels.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Wrote {path}")


def load_labels(path="labels.txt"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def predict_one(model_path, labels_path, image_path, img_size=(224, 224)):
    print(f"Predicting {image_path} ...")
    model = keras.models.load_model(model_path)
    labels = load_labels(labels_path) or []
    img = keras.utils.load_img(image_path, target_size=img_size)
    arr = keras.utils.img_to_array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else str(idx)
    print(f"Predicted: {label}  confidence={probs[idx]:.2f}")
    # Show top 3 for context
    top3 = np.argsort(-probs)[:3]
    for k in top3:
        name = labels[k] if k < len(labels) else str(k)
        print(f"  {name:>20}: {probs[k]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train a simple camping furniture classifier with TensorFlow")
    parser.add_argument("--data_dir", type=str, default="camp_furniture", help="Root folder with class subfolders")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224], help="Image size H W")
    parser.add_argument("--model_out", type=str, default="camp_classifier.keras", help="Output model path")
    parser.add_argument("--labels_out", type=str, default="labels.txt", help="Output labels file")
    parser.add_argument("--predict", type=str, default=None, help="Optional image path to predict after training")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Folder not found: {Path(data_dir).resolve()}")

    img_h, img_w = args.img_size
    img_size = (img_h, img_w)

    # Build datasets
    train_ds, val_ds, class_names, num_classes = build_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=0.27,  # small sets benefit from a bit more validation
        seed=123,
    )

    # Build and train model
    model = build_model(input_shape=img_size + (3,), num_classes=num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save artifacts
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")
    save_labels(class_names, args.labels_out)

    # Optional quick prediction
    if args.predict:
        predict_one(args.model_out, args.labels_out, args.predict, img_size=img_size)


if __name__ == "__main__":
    main()
