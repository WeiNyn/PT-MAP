import os
import sys

from torchvision.transforms.functional import crop


sys.path.append(os.getcwd())

import collections

from PIL import Image

import torch
import torchvision
from torchvision import transforms

from data.datamgr import PatternDataManager, TransformLoader
from src.model.res_model import ResNet

class Model:
    def __init__(self, model: ResNet,
                 samples_root: str,
                 n_shot: int = 5,
                 net_size: int = 84,
                 crop_size = (256, 256),
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        self.device = device
        self.model = model.to(self.device)
        self.samples_root = samples_root
        self.n_shot = n_shot
        self.net_size = net_size
        self.crop_size = crop_size
        
        data_mng = PatternDataManager(self.net_size, batch_size=32)
        self.tfs = data_mng.trans_loader.get_compose_transform(aug=False)
        
        self.num_classes = len(os.listdir(self.samples_root))
        
        self.corpus = self.get_corpus()
        
        self.class_list = list(self.corpus.keys())
        
    def get_corpus(self):
        folders = os.listdir(self.samples_root)
        images_dict = collections.defaultdict(list)
        cropper = transforms.RandomCrop(self.crop_size)
        
        for folder in folders:
            images_dict[folder] = [Image.open(os.path.join(self.samples_root, folder, image)).convert('RGB')
                           for image in os.listdir(os.path.join(self.samples_root, folder))]
            
        
        for label, images in images_dict.items():
            new_images = []
            for i in range(self.n_shot):
                index = i%len(images)
                new_images.append(cropper(images[index]))
                
            images_dict[label] = new_images

        return images_dict 
    
    def make_data(self, crop_size, input_image):
        with torch.no_grad():
            data = torch.zeros(0, *input_image.shape, device=self.device)
            cropper = transforms.CenterCrop(crop_size)
            
            for label, images in self.corpus.items():
                tfs_images = [self.tfs(cropper(image))
                            for image in images]
                
                tfs_images = torch.stack(tfs_images).to(self.device).view(self.n_shot, *input_image.shape)
                
                data = torch.cat([data, tfs_images])
                
            data = torch.cat([data, torch.stack([input_image]).to(self.device)])
            
        torch.cuda.empty_cache()
        return data
    
    def preprocess_image(self, image: Image):
        w, h = image.size
        if min(w, h) < self.crop_size[0]:
            crop_size = (min(w, h), min(w, h))
            
        else:
            crop_size = self.crop_size
            
        cropper = transforms.CenterCrop(crop_size)
        image  = cropper(image)
        image = self.tfs(image)
        return image, crop_size

    def preprocess(self, f):
        f = torch.pow(f + 1e-6, 0.5)
        f = f - f.mean(1, keepdim=True)
        f = f/torch.norm(f, 2, 1)[:, None]
        return f

    def get_probas(self, data):
        d = data[:self.n_shot*self.num_classes]
        q = data[self.n_shot*self.num_classes:]
        
        d = self.preprocess(d)
        q = self.preprocess(q)
        
        mus = d.reshape(self.num_classes, self.n_shot, -1).mean(1)

        dist = (q.unsqueeze(1) - mus.unsqueeze(0)).norm(dim=2).pow(2)
        
        return dist

    def predict_image(self, image: Image):
        image, crop_size = self.preprocess_image(image)
        data = self.make_data(crop_size, image)
        
        with torch.no_grad():
            # data = data.to(self.device)
            data, _ = self.model(data)
            probas = self.get_probas(data)
            probas = probas.cpu()
            
        min_dist = torch.argmin(probas)
        label = list(self.corpus.keys())[min_dist]
        
        return label