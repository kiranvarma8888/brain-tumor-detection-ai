# 🧠 Brain Tumor Detection Using Deep Learning

A full-stack web application that classifies brain MRI scans into **Glioma
Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, or **No Tumor** using a
transfer-learning CNN (EfficientNetB0), with Grad-CAM explainability, a
Flask backend, and a SQLite-backed prediction history.

> ⚠️ Academic project. Not a certified medical device — do not use for real
> clinical decisions.

---

## ✨ Features

- CNN classification via EfficientNetB0 transfer learning (fine-tuned)
- Data augmentation, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Accuracy/loss curves, confusion matrix, precision/recall/F1 report
- Grad-CAM heatmap overlay for model explainability
- Responsive Bootstrap 5 UI with dark mode
- Patient record + prediction history stored in SQLite
- Downloadable PDF report per prediction
- REST API endpoint (`/api/predict`) for programmatic access
- Docker, Render, Railway, and Hugging Face Spaces deployment ready

---

## 🗂️ Project Structure

```
BrainTumorDetection/
├── app.py                  # Flask backend
├── train.py                # Model training pipeline
├── predict.py               # CLI single-image prediction
├── requirements.txt
├── Procfile                 # Render/Railway
├── runtime.txt
├── Dockerfile                # HuggingFace Spaces / GCP
├── README.md
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── uploads/              # uploaded MRI images + heatmaps
├── templates/
│   ├── base.html, index.html, result.html, about.html, contact.html, error.html
├── dataset/                  # place Kaggle dataset here (Training/ Testing/)
├── models/                   # trained brain_tumor_model.h5 goes here
├── utils/
│   ├── grad_cam.py
│   └── report.py
└── screenshots/              # generated accuracy/loss/confusion-matrix plots
```

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd BrainTumorDetection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Dataset

Uses the **Brain MRI Dataset** from Kaggle (e.g. "Brain Tumor MRI Dataset" by
Masoud Nickparvar). Download and arrange it as:

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## 🏋️ Training

```bash
python train.py --epochs 25 --fine_tune_epochs 15 --batch_size 32
```

This runs a two-phase transfer-learning schedule (frozen backbone → fine-tune
top layers), then saves:
- `models/brain_tumor_model.h5`
- `screenshots/accuracy_graph.png`, `loss_graph.png`, `confusion_matrix.png`
- `classification_report.txt`

---

## 🚀 Running the App

```bash
python app.py
```

Visit `http://localhost:5000`.

CLI single-image prediction:
```bash
python predict.py --image path/to/scan.jpg
```

---

## ☁️ Deployment

### Render
1. Push this repo to GitHub.
2. New → Web Service → connect repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add a persistent disk (or external storage) if you need uploads/DB to survive restarts.

### Railway
1. `railway init` in the project folder.
2. `railway up` to deploy — Railway auto-detects the `Procfile`.
3. Set `PORT` env var is handled automatically by Railway.

### Hugging Face Spaces (Docker SDK)
1. Create a new Space → SDK: **Docker**.
2. Push this repo — Spaces will build from the included `Dockerfile`.
3. Ensure `models/brain_tumor_model.h5` is committed via Git LFS (it exceeds the normal file-size soft limit).

### GitHub upload

```bash
git init
git add .
git commit -m "Initial commit: Brain Tumor Detection project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

---

## 📈 Results

See `screenshots/` after training for accuracy/loss curves and the confusion
matrix, and `classification_report.txt` for per-class precision/recall/F1.
Target: >95% test accuracy with the EfficientNetB0 fine-tuned configuration.

---

## 🔭 Future Scope

- Multi-modal input (CT + MRI)
- Tumor segmentation (U-Net) in addition to classification
- Login/auth + role-based admin dashboard
- Email delivery of PDF reports
- Mobile app client consuming the `/api/predict` endpoint

---

## 📄 License

Academic/research use. Not for clinical deployment without regulatory
approval and clinical validation.
