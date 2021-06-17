import torch
from PIL import Image
import numpy as np
import torchvision.transforms as tfs
from torchvision.transforms import transforms
import data.additional_transforms as add_transforms
from data.dataset import PatternDataset
from abc import abstractmethod

class TransformLoader:
    def __init__(self, 
                 image_size: int, 
                 normalize_param: dict = dict(mean=[.485, .456, .406], std=[.229, .224, .225]), 
                 jitter_param: dict = dict(Brightness=.4, Color=.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        
    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        
        method = getattr(tfs, transform_type)
        
        if transform_type=='RandomCrop':
            return method(self.image_size)
        
        elif transform_type=='CenterCrop':
            return method(self.image_size)
        
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        
        elif transform_type=='Normalize':
            return method(**self.normalize_param)
        
        else:
            return method()
        
    def get_compose_transform(self, aug=False):
        if aug:
            transform_list = ['RandomCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            
        else:
            transform_list = ['Scale', 'CenterCrop', 'ToTensor', 'Normalize']
            
        transform_functions = [self.parse_transform(x) for x in transform_list]
        transform = tfs.Compose(transform_functions)
        return transform
    
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass
    
class PatternDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(PatternDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        
    def get_data_loader(self, data_file, aug, len=5000, crop_size=(256, 256)):
        transform = self.trans_loader.get_compose_transform(aug)
        dataset = PatternDataset(data_file, transform, len=len, crop_size=crop_size)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        
        return data_loader