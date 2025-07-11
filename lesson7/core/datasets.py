import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
from torchvision import transforms

FAKE_DATASET_SIZE = 3200

class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""
    
    def __init__(self, root_dir, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            preprocessing: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.target_size = target_size
        
        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Собираем все пути к изображениям
        self.pictures = []
        self.targets = []

        preprocessing = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    image = Image.open(img_path).convert('RGB')
                    image = preprocessing(image)
                    self.pictures.append(image)
                    self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.pictures)
    
    def __getitem__(self, idx):
        image = self.pictures[idx]
        label = self.targets[idx]
        
        return image, label
    
    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes

class RandomImageDataset(Dataset):
    def __init__(self, target_size=(3, 224, 224)):
        self.target_size = target_size
        pictures = [torch.randn(self.target_size) for _ in range(FAKE_DATASET_SIZE)]
        targets = [torch.randint(0, 1000, (1,)) for _ in range(FAKE_DATASET_SIZE)]
        self.pictures = pictures
        self.targets = targets

    def __len__(self):
        return FAKE_DATASET_SIZE
    
    def __getitem__(self, idx):
        image, label = self.pictures[idx], self.targets[idx]
        return image, label

if __name__ == '__main__':
    dataset = RandomImageDataset()
    print(dataset[0])
    dl = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    for batch in dl:
        print(batch[0].shape)