import os
import sys
from typing import Pattern

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from data.datamgr import DataManager, PatternDataManager

from src.model.res_model import ResNet, resnet18

from os import path

def train(base_loader: DataManager, 
          base_test_loader: DataManager, 
          model: ResNet,
          params, tmp, 
          start_epoch: int = 0, stop_epoch: int = 1000,
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    rotate_classifier = nn.Sequential(nn.Linear(512, 4))
    
    rotate_classifier.to(device)
    
    if 'rotate' in tmp:
        print('loading rotate model')
        rotate_classifier.load_state_dict(tmp['rotate'])
        
    optimizer = torch.optim.Adam([
        dict(params=model.parameters()),
        dict(params=rotate_classifier.parameters())
    ])
    
    loss_function = nn.CrossEntropyLoss()
    
    print('start epoch', start_epoch, 'end_epoch', stop_epoch)
    
    current_loss = 1e9
    stack = 0
    
    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch:', epoch)
        
        model.train()
        rotate_classifier.train()
        
        avg_loss = 0
        avg_rotate_loss = 0
        
        for i, (x, y) in enumerate(base_loader):
            x_, y_, a_ = prepare_batch(x, y, device)
            
            f, scores = model.forward(x_)
            rotate_scores = rotate_classifier(f)
            
            optimizer.zero_grad()
            rotate_loss = loss_function(rotate_scores, a_)
            class_loss = loss_function(scores, y_)
            loss = rotate_loss + class_loss
            loss.backward()
            optimizer.step()
            
            avg_loss = avg_loss + class_loss.data.item()
            avg_rotate_loss = avg_rotate_loss + rotate_loss.data.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(base_loader)} | Loss {avg_loss/float(i + 1)}| Rotate Loss {avg_rotate_loss/float(i + 1)}")
                
        if not os.path.isdir(params.checkpoint_dir):
            os.mkdir(params.checkpoint_dir)
            
        if (epoch%params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, f"{epoch}.pth")
            torch.save(dict(epoch=epoch, state=model.state_dict(), rotate=rotate_classifier.state_dict()), outfile)
            
        model.eval()
        rotate_classifier.eval()
        
        with torch.no_grad():
            correct = rotate_correct = total = 0
            for i, (x, y) in enumerate(base_test_loader):

                x_, y_, a_ = prepare_batch(x, y, device)
                
                f, scores = model(x_)
                rotate_scores = rotate_classifier(f)
                p1 = torch.argmax(scores, 1)
                total += p1.size(0)
                correct += (p1 == y_).sum().item()
                p2 = torch.argmax(rotate_scores, 1)
                rotate_correct = (p2 == a_).sum().item()
                
            print(f"Epoch {epoch} | Accuracy {float(correct)*100/total} | Rotate Accuracy {float(rotate_correct)*100/total}")
        
        torch.cuda.empty_cache()
    
        if avg_loss > current_loss * (1 + params.es_scale):
            stack += 1
        
        else:
            stack = 0
        
        if avg_loss < current_loss:
            current_loss = avg_loss
        
        if stack >= params.patience:
            break
    
    return model                    

def prepare_batch(x, y, device):
    bs = x.size(0)
    x_ = []
    y_ = []
    a_ = []

    for j in range(bs):
        x90 = x[j].transpose(2, 1).flip(1)
        x180 = x90.transpose(2, 1).flip(1)
        x270 = x180.transpose(2, 1).flip(1)

        x_ += [x[j], x90, x180, x270]
        y_ += [y[j] for _ in range(4)]
        a_ += [torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3)]

    x_ = torch.stack(x_, 0).to(device)
    y_ = torch.stack(y_, 0).to(device)
    a_ = torch.stack(a_, 0).to(device)
    return x_, y_, a_


if __name__ == '__main__':
    from setting.setting import Setting
    
    dataset_folder = Setting.data_folder
    start_epoch = Setting.start_epoch
    stop_epoch = Setting.stop_epoch
    
    base_datamgr = PatternDataManager(image_size=Setting.image_size, batch_size=Setting.train_batch_size)
    base_loader = base_datamgr.get_data_loader(dataset_folder, aug=Setting.train_aug)
    base_test_datamgr = PatternDataManager(image_size=Setting.image_size, batch_size=Setting.test_batch_size)
    base_test_loader = base_test_datamgr.get_data_loader(dataset_folder, aug=Setting.train_aug)
    
    model = resnet18(num_classes=Setting.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    if Setting.resume is not None:
        print(f"Resume file: {Setting.resume}")
        tmp = torch.load(Setting.resume)
        start_epoch = tmp['epoch'] + 1
        print(f"Restored epoch is: {start_epoch - 1}")
        state = tmp['state']
        model.load_state_dict(state)
    else:
        tmp = {}
    
    model = train(base_loader, base_test_loader, model, Setting, tmp, start_epoch, stop_epoch, device)
    
    