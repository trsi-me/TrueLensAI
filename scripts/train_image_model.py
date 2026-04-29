# -*- coding: utf-8 -*-
# Train EfficientNetB0 binary classifier (e.g. CIFAKE: train/REAL vs train/FAKE).
import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)
DATASET_ROOT = os.path.join(BASE_DIR, "datasate")
IMAGE_ARCHIVE_DIR = os.path.join(DATASET_ROOT, "archive")
IMAGE_TRAIN_DIR = os.path.join(IMAGE_ARCHIVE_DIR, "train")
IMAGE_TEST_DIR = os.path.join(IMAGE_ARCHIVE_DIR, "test")
OUT_MODEL = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.h5")


def _default_image_train_dir() -> str:
    """Prefer datasate/archive/train/{FAKE,REAL}; else datasate/archive/{classes}."""
    if os.path.isdir(IMAGE_TRAIN_DIR):
        return IMAGE_TRAIN_DIR
    return IMAGE_ARCHIVE_DIR


def build_model(img_size: int = 224) -> keras.Model:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Folder with class subdirs (REAL/FAKE); default: datasate/archive or datasate/archive/train",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    data_dir = args.data_dir or _default_image_train_dir()
    if not os.path.isdir(data_dir):
        raise SystemExit(
            "Data directory not found: "
            + data_dir
            + " — put images under datasate/archive (see README) or set --data_dir."
        )
    print("Using data_dir:", data_dir)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )
    print("Class indices:", train_gen.class_indices)
    print(
        "Output is sigmoid P(class=1); app ImageDetector maps to fake probability for CIFAKE."
    )
    model, base = build_model(args.img_size)
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[es],
    )
    val_acc = history.history.get("val_accuracy", [0])[-1]
    print("Last val_accuracy:", round(float(val_acc), 4))
    base.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(5, args.epochs),
        callbacks=[es],
    )
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    model.save(OUT_MODEL)
    print("Saved model:", OUT_MODEL)
    if val_acc < 0.90:
        print("Warning: val_accuracy below 90%. Consider more data or epochs.")
    _evaluate_on_test_folder(model, args.img_size, args.batch)


def _evaluate_on_test_folder(model, img_size: int, batch: int) -> None:
    test_dir = IMAGE_TEST_DIR
    if not os.path.isdir(test_dir):
        return
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch,
        class_mode="binary",
        shuffle=False,
    )
    loss, acc = model.evaluate(test_gen, verbose=0)
    print("Held-out test (datasate/archive/test): loss=%.4f accuracy=%.4f" % (loss, acc))


if __name__ == "__main__":
    main()
