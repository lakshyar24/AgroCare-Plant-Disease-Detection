import os
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from config import *
from tqdm import tqdm

# Load model
model = get_model(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels
class_labels = sorted(os.listdir(TRAIN_DATA_PATH))

# Collect test image paths and true labels
test_dir = TEST_DATA_PATH
test_data = []

for cls in os.listdir(test_dir):
    cls_path = os.path.join(test_dir, cls)
    for fname in os.listdir(cls_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            test_data.append((os.path.join(cls_path, fname), cls))

print(f"\n🔍 Found {len(test_data)} test images. Running predictions...\n")

# Track metrics
correct = 0
total = 0
verbose = False  # Set to True to show per-image predictions

# Inference loop
for img_path, true_class in tqdm(test_data, desc="Testing (test_split)"):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        predicted_class = class_labels[pred_idx]

    if predicted_class == true_class:
        correct += 1
    total += 1

    if verbose:
        print(f"✅ {os.path.basename(img_path)}: {predicted_class} ({confidence:.2f})")

# Final accuracy
accuracy = correct / total * 100
print(f"\n🎯 Test Accuracy: {accuracy:.2f}% on {total} images.\n")
