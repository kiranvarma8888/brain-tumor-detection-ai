"""
predict.py
----------
Standalone CLI script to run a single-image prediction without starting
the Flask server. Useful for quick testing and for the training pipeline's
sanity checks.

Usage:
    python predict.py --image path/to/mri_scan.jpg
"""

import argparse
import os
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_model.h5")


def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI image")
    parser.add_argument("--image", required=True, help="Path to an MRI image (png/jpg)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained .h5 model")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train it first with train.py."
        )

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    arr = load_and_preprocess(args.image)
    probs = model.predict(arr)[0]
    pred_idx = int(np.argmax(probs))

    print(f"\nPrediction: {CLASS_NAMES[pred_idx]}")
    print(f"Confidence: {probs[pred_idx] * 100:.2f}%\n")
    print("Class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:<20s}: {p * 100:6.2f}%")


if __name__ == "__main__":
    main()
