# 🌿 AgroCare: Plant Disease Detection System

AgroCare is a deep learning-based plant disease recognition system built using PyTorch and ResNet-18. The project features an interactive Streamlit UI for prediction, Grad-CAM visualizations, multilingual output, and speech synthesis.

---

## 🚀 Features

- 🔍 Detects 38 types of plant diseases from leaf images
- 📈 ResNet-18 based classification with best model saving & early stopping
- 🗣️ Text-to-speech output in English, Hindi, Tamil using gTTS
- 🌐 Translates prediction results into multiple languages
- 🔥 Grad-CAM visual explanation of predictions
- 📊 Metrics: Accuracy, Precision, Recall, F1-score

---

## 🗂️ Directory Structure

```
app.py                   # Streamlit UI (Home, About, Disease Recognition)
train.py                 # Model training with early stopping & checkpointing
test.py                  # Run inference on test_split folder
model_evaluation.py      # Evaluate model: accuracy, precision, recall, F1
model/                   # Model architecture definition (ResNet-18)
config/                  # Config file with paths, device, batch size
speak.py                 # Text-to-speech (gTTS) utility
languages.py             # Translate prediction to multiple languages
grad_cam_utils.py        # Grad-CAM heatmap visualizer
plant_disease_model_v2.pth  # Final saved PyTorch model
requirements.txt         # Python dependencies
home_page.jpeg           # UI header image for homepage
data/
├── train_split/         # Training images
├── val_split/           # Validation images
└── test_split/          # Testing images
```

---

## 🧪 How to Run

1. Install dependencies  
```bash
pip install -r requirements.txt
```

2. Train the model  
```bash
python train.py
```

3. Evaluate the model  
```bash
python model_evaluation.py
```

4. Run the web app  
```bash
streamlit run app.py
```

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`.

---

## 👨‍💻 Author

Developed by a team of 3 — Contributions included frontend development, model building, data preprocessing, and multilingual + TTS enhancements.

---

## 📸 Sample UI

![UI](home_page.jpeg)

---

Dataset link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
Download dataset and split into train_split, val_split and test_split.

## 📜 License

This project is for academic purposes.
