# Filename: dataset_preparation.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class AOLPDataset(Dataset):
    def __init__(self, root_dir, transform=None, subjects=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'face': 0, 'paper': 1, 'screen': 1}
        self.images = []
        self.labels = []
        self.subjects = subjects  # List of subjects to include

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue  # Skip if the class directory does not exist
            for img_name in os.listdir(cls_dir):
                # Extract the subject name from the image filename
                subject_name = img_name.split('_')[0]
                if self.subjects is None or subject_name in self.subjects:
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.classes[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path  # Return image path for consistency

def get_subjects(root_dir):
    subjects = set()
    for cls in ['face', 'paper', 'screen']:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue  # Skip if the class directory does not exist
        for img_name in os.listdir(cls_dir):
            subject_name = img_name.split('_')[0]
            subjects.add(subject_name)
    return sorted(subjects)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
