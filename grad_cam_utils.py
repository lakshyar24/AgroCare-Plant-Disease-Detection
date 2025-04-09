# grad_cam_utils.py

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

def get_grad_cam_image(model, image_pil, target_category, target_layer):
    # Prepare input
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_pil).unsqueeze(0)

    # Grad-CAM setup
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])
    grayscale_cam = grayscale_cam[0, :]

    # Convert PIL to numpy for overlay
    import numpy as np
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    return cam_image
