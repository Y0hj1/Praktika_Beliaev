from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        # Загружаем датасет, где каждая подпапка — отдельный класс
        self.inner = ImageFolder(root=directory)
        self.transform = transform

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        # Получаем путь и метку
        img_path, label = self.inner.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label