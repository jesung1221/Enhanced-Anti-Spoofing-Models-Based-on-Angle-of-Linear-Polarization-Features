import torch.nn as nn
from torchvision import models
import torch

# Function to initialize the model
def initialize_model():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # 2 classes: real, non-real
    return model

# Initialize the model for immediate use
model = initialize_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
