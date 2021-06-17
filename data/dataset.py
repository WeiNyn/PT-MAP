import torch
from PIL import Image
import json
import numpy as np
import torchvision
import torchvision.transforms as tfs
import os

identity = lambda x: x

class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, transform, target_transform=identity, len :int = 5000, crop_size = (256, 256)):
        self.dataset = torchvision.datasets.ImageFolder(data_file)
        self.crop_size=crop_size
        self.transform = transform
        self.target_transform = target_transform
        self.len = len
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        chosen_index = idx%len(self.dataset)
        image, label = self.dataset[chosen_index]
        image = tfs.RandomCrop(self.crop_size)(image)
        image = tfs.RandomAffine(degrees=15)(image)
        image = self.transform(image)
        label = self.target_transform(label)
        
        return image, label
    