import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from core.datasets import CustomImageDataset
from core.net import Resnet18


def process_epoch(net, batch_iterator, loss_fn, opt=None, computing_device='cpu', is_test=False):
    if is_test:
        net.eval()
    else:
        net.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(batch_iterator)):
        data, target = data.to(computing_device), target.to(computing_device)
        
        if not is_test and opt is not None:
            opt.zero_grad()
        
        output = net(data)
        loss = loss_fn(output, target)
        
        if not is_test and opt is not None:
            loss.backward()
            opt.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(batch_iterator), correct / total


def execute_training(net, loader_train, loader_test, epochs=10, lr=0.001, computing_device='cpu', output_weights_path='./weights/best_resnet18.pth'):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = process_epoch(net, loader_train, loss_fn, opt, computing_device, is_test=False)
        test_loss, test_acc = process_epoch(net, loader_test, loss_fn, None, computing_device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)
        
        # Сохраняем лучшие веса
        if output_weights_path is not None and test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
            torch.save(net.state_dict(), output_weights_path)
            print(f'Лучшие веса сохранены в {output_weights_path} (Test Acc: {best_acc:.4f})')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

if __name__ == '__main__':
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomImageDataset(root_dir='../lesson5_augmentations/data/train', preprocessing=preprocessing)
    test_dataset = CustomImageDataset(root_dir='../lesson5_augmentations/data/test', preprocessing=preprocessing)
    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)
    net = Resnet18()
    net = net.to('cuda')
    execute_training(net, loader_train, loader_test, epochs=10, lr=0.001, computing_device='cuda', output_weights_path='./weights/best_resnet18.pth')