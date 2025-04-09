# config.py

TRAIN_DATA_PATH = r"C:\Users\laksh\Downloads\Plant_Disease_Prediction\data\train_split"
VAL_DATA_PATH = r"C:\Users\laksh\Downloads\Plant_Disease_Prediction\data\val_split"
TEST_DATA_PATH = r"C:\Users\laksh\Downloads\Plant_Disease_Prediction\data\test_split"
MODEL_PATH = "plant_disease_model_v2.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 38
