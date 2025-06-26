from pathlib import Path
import torch

# Base paths 
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'Data'
MODEL_DIR = BASE_DIR / 'Models'

# Data Paths 
CLASSIFICATION_DATA  =  {
    "train" : DATA_DIR / 'Classification' / "Training_Data",
    "test" : DATA_DIR / 'Classification' / "Test_Data"
}

SEGMENTATION_DATA  =  {
    "train" : DATA_DIR / 'Segmentation' / "Training_Data",
    "test" : DATA_DIR / 'Segmentation' / "Test_Data"
}

# Model parameters 
CLASSIFICATION_CONFIG = {
    "model_name": "resnet50",
    "number_classes": 2, # Either malignant or non-malignant
    "input_size": (224, 224),
    "batch_size": 32,
    "epochs": 30
}

SEGMENTATION_CONFIG = {
    "model_name": "unet",
    "image_size": (256, 256),
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 60
}

# Training parameters 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
RANDOM_SEED = 42

# Model paths
MODEL_PATHS = {
    "Classification": MODEL_DIR / "classification_model.h5",
    "Segmentation": MODEL_DIR / "segmentation_model.h5"
}

# Create directories if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)