import torch
from PIL import ImageEnhance

transform_type_dict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter:
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
        
    def __call__(self, img):
        out = img
        rand_tensor = torch.rand(len(self.transforms))
        
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(rand_tensor[i]*2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
            
            
        return out