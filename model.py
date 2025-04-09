from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet18_Weights

def get_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
