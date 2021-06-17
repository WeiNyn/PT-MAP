from torchvision.models import resnet50
import torch
from torch.nn import Sequential, Linear

class ResNet50(torch.nn.Module):
    def __init__(self, num_classes: int):
        model = resnet50(pretrained=True)
        model.fc = Linear(2048, num_classes)
        
        self.ember = Sequential(*list(model.children())[:-1])
        self.fc = Sequential(*list(model.children()[-1:]))
        
    def forward(self, x):
        x = self.ember(x)
        scores = self.fc(x.view(x.shape[0], -1))
        
        return x, scores
    
    
def resnet50(num_class: int):
    return ResNet50(num_classes=num_class)