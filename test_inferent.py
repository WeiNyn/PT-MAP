from torchvision import transforms
from src.inference.new_infr import Model

from src.model.res_model import resnet18

import torch

from PIL import Image
from time import time

model = resnet18(num_classes=120)

data_root = 'test_base/'
data_root = '120_samples_database_cut/'
checkpoint_path = 'checkpoints/180.pth'

checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['state'])

corpus = Model(model=model, samples_root=data_root, n_shot=3)

from data.datamgr import PatternDataManager

data_mng = PatternDataManager(image_size=84, batch_size=8)
data_loader = data_mng.get_data_loader(data_root, aug=False, len=100)

images = [image for image, _ in data_loader.dataset.dataset]
labels = [data_loader.dataset.dataset.classes[label] for _, label in data_loader.dataset.dataset]

sizes = [100, 150, 200, 250, 300, 350]

count = 0
true = 0
total_time = 0
for size in sizes:
    for image, label in zip(images, labels):
        for i in range(3):
            image = transforms.RandomAffine(degrees=10)(image)
            image = transforms.RandomCrop(size)(image)
            s = time()
            result = corpus.predict_image(image)
            total_time += (time() - s)
            count += 1
            if label == result:
                true += 1
            else:
                print(size, label, result)
                
print(count, true, float(true)/count, total_time/count)