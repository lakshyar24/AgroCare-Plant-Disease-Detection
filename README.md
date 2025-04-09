# 🌿 AgroCare: AI-Powered Plant Disease Detection

![AgroCare Banner](home_page.jpeg)

AgroCare is a smart, multilingual **plant disease recognition system** built using **PyTorch** and **ResNet-18**, enhanced with **Grad-CAM** visualizations and **text-to-speech (TTS)** support. Designed for farmers, agriculturists, and researchers, this tool helps identify **38+ plant leaf diseases** with high accuracy using a simple **Streamlit interface**.

---

## 🚀 Features

- 🔍 **Deep Learning Powered** (ResNet-18 + Softmax)
- 🎯 **98% Accuracy** on test dataset
- 🧠 **Grad-CAM Visualization** to highlight affected areas
- 🗣️ **Multilingual Text-to-Speech** output (English, Hindi, Tamil)
- 🌐 **Real-time Image Prediction**
- 📊 **Model Evaluation** with precision, recall, F1-score
- 📦 Lightweight, fast, and intuitive UI with Streamlit

---

## 🖼️ Model Architecture

1. **Input Upload**: Leaf image via web interface  
2. **Preprocessing**: Resize, normalize image  
3. **Feature Extraction**: ResNet-18 pretrained on ImageNet  
4. **Classification**: 38 output classes  
5. **Softmax + Confidence Score**  
6. **Grad-CAM Heatmap**  
7. **TTS + Translation** for prediction output

---

## 📁 Project Structure

