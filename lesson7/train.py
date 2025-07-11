import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from core.datasets import CustomImageDataset
from core.net import Resnet18
from trainer import execute_training

sizes = [224, 256, 384, 512]
computing_device = 'cuda' if torch.cuda.is_available() else 'cpu'

for size in sizes:
    preprocessing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomImageDataset(root_dir='../lesson5/data/train')
    test_dataset = CustomImageDataset(root_dir='../lesson5/data/test')
    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

    net = Resnet18(num_classes=len(train_dataset.get_class_names()))
    net = net.to(computing_device)
    output_weights_path = f'./weights/best_resnet18_{size}.pth'

    execute_training(net, loader_train, loader_test, epochs=10, lr=0.001, computing_device=computing_device, output_weights_path=output_weights_path)