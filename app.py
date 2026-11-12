"""
app.py
------
Flask backend for the Brain Tumor Detection web application.

Responsibilities:
  - Serve the frontend (Home, About, Upload, Result, Contact pages)
  - Accept MRI image uploads
  - Run inference using the trained CNN (models/brain_tumor_model.h5)
  - Generate a Grad-CAM heatmap for explainability
  - Persist patient + prediction records to SQLite
  - Provide a simple JSON API for programmatic access
  - Provide a downloadable PDF report of a prediction
  - Basic logging + error handling suitable for deployment
"""

import os
import io
import uuid
import sqlite3
import logging
from datetime import datetime

import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, g
)
from werkzeug.utils import secure_filename
from PIL import Image

from utils.grad_cam import generate_gradcam
from utils.report import build_pdf_report

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_model.h5")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Lazy model loading (avoids loading TensorFlow at import time for routes
# that don't need it, and keeps app startup fast on free-tier cloud hosts)
# --------------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first, "
                f"or place a trained brain_tumor_model.h5 in the models/ folder."
            )
        logger.info("Loading model from %s", MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            gender TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            image_path TEXT NOT NULL,
            heatmap_path TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Load an image, resize, and normalize it for model input."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # In production, wire this up to an email service (e.g. Flask-Mail, SendGrid)
        name = request.form.get("name", "")
        logger.info("Contact form submitted by %s", name)
        flash("Thanks for reaching out — we'll get back to you soon.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/history")
def history():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    return render_template("history.html", records=rows) if os.path.exists(
        os.path.join(BASE_DIR, "templates", "history.html")
    ) else jsonify([dict(r) for r in rows])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "mri_image" not in request.files:
            flash("No file selected.", "danger")
            return redirect(url_for("home"))

        file = request.files["mri_image"]
        if file.filename == "" or not allowed_file(file.filename):
            flash("Please upload a valid PNG/JPG MRI image.", "danger")
            return redirect(url_for("home"))

        patient_name = request.form.get("patient_name", "Anonymous")
        age = request.form.get("age", type=int)
        gender = request.form.get("gender", "N/A")

        # Save upload with a unique filename to avoid collisions
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)
        logger.info("Saved upload: %s", save_path)

        # Preprocess + predict
        arr, pil_img = preprocess_image(save_path)
        model = get_model()
        probs = model.predict(arr)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        # Grad-CAM heatmap for explainability
        heatmap_name = f"gradcam_{unique_name}"
        heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_name)
        try:
            generate_gradcam(model, arr, save_path, heatmap_path)
        except Exception as e:
            logger.warning("Grad-CAM generation failed: %s", e)
            heatmap_name = None

        probabilities = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}

        # Persist to DB
        db = get_db()
        db.execute(
            """INSERT INTO predictions
               (patient_name, age, gender, prediction, confidence, probabilities,
                image_path, heatmap_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patient_name, age, gender, pred_class, confidence, str(probabilities),
                f"uploads/{unique_name}",
                f"uploads/{heatmap_name}" if heatmap_name else None,
                datetime.utcnow().isoformat()
            )
        )
        db.commit()
        record_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        is_tumor = pred_class != "No Tumor"

        return render_template(
            "result.html",
            patient_name=patient_name,
            age=age,
            gender=gender,
            prediction=pred_class,
            confidence=round(confidence, 2),
            probabilities=probabilities,
            image_path=f"uploads/{unique_name}",
            heatmap_path=f"uploads/{heatmap_name}" if heatmap_name else None,
            is_tumor=is_tumor,
            record_id=record_id
        )

    except FileNotFoundError as e:
        logger.error("Model missing: %s", e)
        flash(str(e), "danger")
        return redirect(url_for("home"))
    except Exception as e:
        logger.exception("Prediction failed")
        flash(f"Something went wrong while processing the image: {e}", "danger")
        return redirect(url_for("home"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint — returns prediction without rendering HTML."""
    if "mri_image" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["mri_image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    arr, _ = preprocess_image(save_path)
    model = get_model()
    probs = model.predict(arr)[0]
    pred_idx = int(np.argmax(probs))

    return jsonify({
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    })


@app.route("/report/<int:record_id>")
def download_report(record_id):
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        flash("Report not found.", "danger")
        return redirect(url_for("home"))

    pdf_buffer = build_pdf_report(dict(row), BASE_DIR)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"tumor_report_{record_id}.pdf",
        mimetype="application/pdf"
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error")
    return render_template("error.html", code=500, message="Internal server error"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
