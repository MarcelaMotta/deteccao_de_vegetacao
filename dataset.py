import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class VegetationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('L')  # L mode for grayscale

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Ensure labels are in the right shape for binary classification
        label = (label > 0).float()  # Convert to binary values (0 or 1)

        return image, label
