import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import numpy as np
from languages import translate_text
from speak import speak_text

# ✅ Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ✅ Streamlit config
st.set_page_config(page_title="AgroCare: Plant Disease Detector 🌿", layout="centered")

# ========================
# Configuration
# ========================
MODEL_PATH = "plant_disease_model_v2.pth"
CLASS_NAMES = sorted(os.listdir("data/train_split"))

# ========================
# Load model + target layer for Grad-CAM
# ========================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    target_layer = model.layer4[-1]  # ✅ A valid Grad-CAM compatible layer
    return model, target_layer

model, target_layer = load_model()

# ========================
# Define transforms
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========================
# Streamlit UI
# ========================
# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Home page
if(app_mode=="Home"):
    st.header("AGROCARE: PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the AgroCare: Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (38000 images)
                2. test (5700 images)
                3. validation (13870 images)

                """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

        with st.spinner("Predicting..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_class = torch.max(probs, 0)

        # Format class name for display and speech
        raw_class = CLASS_NAMES[predicted_class]
        clean_class = raw_class.replace("_", " ").replace("  ", " ").strip()

        st.success(f"✅ **Prediction:** {clean_class}")
        st.write(f"🔬 **Confidence:** {confidence.item():.2%}")


        # 🌐 Language Selector
        lang_map = {
            "English": "en",
            "Hindi (हिंदी)": "hi",
            "Tamil (தமிழ்)": "ta"
        }
        selected_lang = st.selectbox("🌐 Choose language", list(lang_map.keys()))
        lang_code = lang_map[selected_lang]

        # 📝 Textual Conclusion
        conclusion_text = f"The plant is affected by {clean_class}."
        translated_conclusion = translate_text(conclusion_text, lang_code)
        st.markdown(f"📝 **Conclusion:** {translated_conclusion}")

        # 🔊 Text-to-Speech
        if st.button("🔊 Speak Diagnosis"):
            speak_text(translated_conclusion, lang=lang_code)

        # Grad-CAM Visualization
        if st.button("🔍 Show Grad-CAM Visualization"):
            with st.spinner("Generating Grad-CAM..."):
                cam = GradCAM(model=model, target_layers=[target_layer])
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=[ClassifierOutputTarget(predicted_class.item())])[0]

                image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
                cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

                st.image(cam_image, caption="🔎 Grad-CAM Activation Map", use_container_width=True)
