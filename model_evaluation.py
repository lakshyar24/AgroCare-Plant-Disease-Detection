import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import get_model
from config import *
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Transformations
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Dataset and loader
# -------------------------
val_dataset = ImageFolder(root=VAL_DATA_PATH, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = val_dataset.classes

# -------------------------
# Load model
# -------------------------
model = get_model(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------
# Evaluation loop
# -------------------------
all_preds = []
all_labels = []
correct, total = 0, 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating (val_split)"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

# -------------------------
# Metrics
# -------------------------
accuracy = 100 * correct / total
print(f"\n✅ Validation Accuracy: {accuracy:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# -------------------------
# Confusion Matrix
# -------------------------
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_val.png")
plt.show()
