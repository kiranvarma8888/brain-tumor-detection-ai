"""
train.py
--------
Training pipeline for the Brain Tumor Detection CNN.

Uses transfer learning on EfficientNetB0 (pretrained on ImageNet), fine-tuned
on a 4-class brain MRI dataset:
    Glioma Tumor | Meningioma Tumor | No Tumor | Pituitary Tumor

Expected dataset layout (Kaggle "Brain Tumor MRI Dataset" format):

    dataset/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/

Run:
    python train.py --epochs 30 --batch_size 32

Outputs:
    models/brain_tumor_model.h5      - trained model
    screenshots/accuracy_graph.png   - training/validation accuracy curve
    screenshots/loss_graph.png       - training/validation loss curve
    screenshots/confusion_matrix.png - confusion matrix on the test set
    classification_report.txt        - precision / recall / f1 per class
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for servers / CI
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def build_data_generators(batch_size):
    """
    Data augmentation for the training set (rotation, shifts, zoom, flips)
    to improve generalization; validation/test data is only rescaled.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        validation_split=0.15  # carve a validation split out of the training folder
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "Training"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        shuffle=True,
        seed=42
    )
    val_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "Training"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        shuffle=False,
        seed=42
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "Testing"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False
    )
    return train_gen, val_gen, test_gen


def build_model(num_classes=4, fine_tune_at=100):
    """
    Transfer learning model: EfficientNetB0 backbone (ImageNet weights) +
    a custom classification head. The base is initially frozen; the top
    `fine_tune_at` layers can later be unfrozen for fine-tuning.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # freeze for the first training phase

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def plot_history(history, out_dir):
    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_graph.png"), dpi=150)
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_graph.png"), dpi=150)
    plt.close()


def evaluate_and_report(model, test_gen, out_dir):
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)
    with open(os.path.join(BASE_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    return report


def main(args):
    print("Building data generators...")
    train_gen, val_gen, test_gen = build_data_generators(args.batch_size)

    print("Building model (EfficientNetB0 transfer learning)...")
    model, base_model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "brain_tumor_model.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
    ]

    print("Phase 1: training classification head (base frozen)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # --- Fine-tuning phase: unfreeze top layers of the backbone ---
    print("Phase 2: fine-tuning top layers of EfficientNetB0...")
    base_model.trainable = True
    for layer in base_model.layers[:-args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks
    )

    # Merge histories for plotting
    for k in history.history:
        history.history[k] += history_fine.history[k]

    plot_history(history, SCREENSHOT_DIR)

    print("Evaluating on test set...")
    evaluate_and_report(model, test_gen, SCREENSHOT_DIR)

    model.save(os.path.join(MODEL_DIR, "brain_tumor_model.h5"))
    print(f"Model saved to {os.path.join(MODEL_DIR, 'brain_tumor_model.h5')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Detection model")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs for head-training phase")
    parser.add_argument("--fine_tune_epochs", type=int, default=15, help="Epochs for fine-tuning phase")
    parser.add_argument("--fine_tune_at", type=int, default=40, help="Number of top backbone layers to unfreeze")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
