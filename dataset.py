import random
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, path, transform, training=True, contrasive_image=False):
        super().__init__()
        self.training = training
        self.path = path
        self.transform = transform
        self.contrasive_image = contrasive_image
        self.image_paths = []
        self.labels = []  # Class labels

        # Load image paths and labels
        if self.training:
            for category in os.listdir(path):
                category_path = os.path.join(self.path, category)
                if not os.path.isdir(category_path):
                    continue  # Skip non-folder files
                
                image_files = sorted([os.path.join(category_path, x) for x in os.listdir(category_path) if x.endswith(".jpg")])
                self.image_paths.extend(image_files)
                self.labels.extend([int(category)] * len(image_files))  # Assign class label based on folder name
        else:
            self.image_paths = sorted([os.path.join(self.path, x) for x in os.listdir(path) if x.endswith(".jpg")])
            self.labels = None  # No labels for test set

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img1_path = self.image_paths[index]
        img1 = Image.open(img1_path).convert("RGB")
        img1 = self.transform(img1)

        if not self.training:
            return img1, img1_path  # Return only image and filename for test set

        label1 = self.labels[index]

        if not self.contrasive_image:
            return img1, label1

        # Select a second image: positive (same class) or negative (different class)
        if random.random() > 0.5:
            # Select a positive pair (same class)
            positive_indices = [i for i, lbl in enumerate(self.labels) if lbl == label1 and i != index]
            if positive_indices:
                index2 = random.choice(positive_indices)
            else:
                index2 = index  # Fallback (unlikely edge case)
            target = 1  # Similar
        else:
            # Select a negative pair (different class)
            negative_indices = [i for i, lbl in enumerate(self.labels) if lbl != label1]
            index2 = random.choice(negative_indices)
            target = -1  # Dissimilar

        img2_path = self.image_paths[index2]
        img2 = Image.open(img2_path).convert("RGB")
        img2 = self.transform(img2)
        label2 = self.labels[index2]

        return img1, img2, torch.tensor(target, dtype=torch.float32), label1, label2
