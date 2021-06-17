import os
from os import listdir, path
import sys

sys.path.append(os.getcwd())

import collections

from PIL import Image

import torch
from torchvision import transforms

from data.datamgr import TransformLoader, PatternDataManager
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
        
        data_mng = PatternDataManager(image_size=self.net_size, batch_size=32)
        
        classes = os.listdir(self.samples_root)
        self.num_classes = len(classes)
        self.data_loader = data_mng.get_data_loader(self.samples_root, aug=False, len=self.num_classes*self.n_shot, crop_size=(100, 100))
        self.tfs = self.data_loader.dataset.transform
        self.classes = self.data_loader.dataset.dataset.classes
        
        output_dict, self.inputs = self.extract_features()
        data, _ = self.gen_corpus(output_dict)
        data = self.preprocess(data)
        self.mus = self.gen_mus(data).to(self.device)
        
    def extract_features(self):
        output_dict = collections.defaultdict(list)
        
        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs = inputs.to(self.device)
                
                outputs, _ = self.model(inputs)
                outputs = outputs.cpu()
                
                for out, label in zip(outputs, labels):
                    output_dict[label.item()].append(out)
                    
        return output_dict, inputs.cpu()
    
    def gen_corpus(self, features: dict):
        labels = sorted(features.keys())
        features_size = features[labels[0]][0].shape[-1]
        
        data = torch.zeros((0, self.n_shot, features_size))
        out_labels = []
        
        for label in labels:
            data = torch.cat([data, torch.stack(features[label]).view(1, self.n_shot, -1)], dim=0)
            
            out_labels += [label for _ in range(self.n_shot)]
            
        return data.permute(1, 0, 2).reshape(self.n_shot*self.num_classes, -1), out_labels
    
    @staticmethod
    def preprocess(data):
        data = torch.pow(data + 1e-6, 0.5)
        data = data - data.mean(1, keepdim=True)
        data = data/torch.norm(data, 2, 1)[:, None]
        
        return data
    
    def gen_mus(self, data):
        return data.reshape(self.n_shot, self.num_classes, -1).mean(0)
    
    def compute_optimal_transport(self, M, r, c, lam, epsilon=1e-6):
        with torch.no_grad():
            r = r.to(self.device)
            c = c.to(self.device)
            
            n, m = M.shape
            P = torch.exp(- lam * M)
            P /= P.view((-1)).sum(0).unsqueeze(0)
            
            u = torch.zeros(n).to(self.device)
            
            max_iters = 1000
            iters = 1
            
            while torch.max(torch.abs(u - P.sum(1))) > epsilon:
                u = P.sum(1)
                P *= (r / u).view((-1, 1))
                P *= (c / P.sum(0)).view((1, -1))
                
                if iters == max_iters:
                    break
                
                iters += 1
            
        torch.cuda.empty_cache()    
        return P
    
    def get_probas(self, inputs: torch.Tensor):
        dist = (inputs.unsqueeze(1) - self.mus.unsqueeze(0)).norm(dim=2).pow(2)
        
        p_xj = torch.zeros_like(dist)
        r = torch.ones(inputs.shape[0])
        c = torch.ones(self.num_classes)
        
        p_xj = self.compute_optimal_transport(dist, r, c, lam=10)
        
        return p_xj
    
    def predict_features(self, inputs):
        inputs = self.preprocess(inputs)
        
        dist = (inputs.unsqueeze(1) - self.mus.unsqueeze(0)).norm(dim=2).pow(2)
        predicts = dist.argmin(dim=1)
        # probas = self.get_probas(inputs)
        # predicts = probas.argmax(dim=1)
        labels = [self.classes[int(p)] for p in predicts]
        return labels
    
    def preprocess_image(self, image: Image, upper_thresh: int = 320):
        w, h = image.size
        if self.crop_size[0] < min(w, h) < upper_thresh:
            tfs = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Lambda(lambda image: torch.stack([image]))  
            ])
            
        elif min(w, h) > upper_thresh:
            tfs = transforms.Compose([
                transforms.FiveCrop(self.crop_size),
                transforms.Lambda(lambda crops: [crop for crop in crops])
            ])
            
        images = tfs(image)
        images = torch.stack([self.tfs(image) for image in images])
            
        return images, images.shape[0]
    
    def predict_image(self, image: Image):
        images, num_samples = self.preprocess_image(image)
        
        images = torch.cat([images, self.inputs[num_samples:]])
        
        with torch.no_grad():
            images = images.to(self.device)
            
            features, _ = self.model(images)
            features = features[:num_samples]
            labels = self.predict_features(features)
            
        if num_samples == 1:
            return labels[0]
            
        else:
            counter = collections.Counter(labels)
            return max(counter.keys(), key=lambda label: counter[label])
        
    def predict_images(self, images: list):
        images_list = torch.zeros(0, 3, 84, 84)
        num_samples = []
        for image in images:
            processed_image, num_sample = self.preprocess_image(image)
            images_list = torch.cat([images_list, processed_image])
            num_samples.append(num_sample)
            
        images_count = images_list.shape[0]
        
        if images_count%64 != 0:
            remain = images_count%64
            
            images_list = torch.cat([images_list, self.inputs[:remain]])
            
        index = 0
        labels = []
        with torch.no_grad():
            while index < images_count:
                
                current = images_list[index:index+64].to(self.device)
                
                features, _ = self.model(current)
                
                labels += self.predict_features(features)
                
                index +=64
                
        labels = labels[:images_count]
        
        output = []
        label_index = 0
        for num in num_samples:
            if num == 1:
                output.append(labels[index])
            
            else:
                counter = collections.Counter(labels[label_index: index+num])
                output.append(max(counter.keys(), key=lambda x: counter[x]))
            
            label_index += num
            
        return output