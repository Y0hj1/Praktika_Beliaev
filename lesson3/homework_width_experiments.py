
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.trainer import train_model

class FCNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def experiment_with_widths():
    train_loader, test_loader = get_mnist_loaders()
    input_size = 28 * 28
    output_size = 10

    widths = [32, 64, 128, 256]
    results = []

    for width in widths:
        hidden_sizes = [width] * 3
        print(f"Training model with width: {width}")
        model = FCNet(input_size, output_size, hidden_sizes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        history = train_model(model, train_loader, test_loader, epochs=10)
        results.append({
            "width": width,
            "train_acc": history["train_accs"][-1],
            "test_acc": history["test_accs"][-1],
            "history": history
        })

    for result in results:
        name = f"width {result['width']}"
        plt.plot(result["history"]["train_accs"], label=f"{name} train", linestyle="--")
        plt.plot(result["history"]["test_accs"], label=f"{name} test")
    plt.title("Accuracy vs Width")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("width_accuracy.png")
    plt.close()

    # Грид-поиск по двум параметрам ширины
    width1 = [32, 64, 128]
    width2 = [64, 128, 256]
    heatmap = np.zeros((len(width1), len(width2)))

    for i, w1 in enumerate(width1):
        for j, w2 in enumerate(width2):
            print(f"Training with widths: {w1}, {w2}")
            model = FCNet(input_size, output_size, [w1, w2])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            history = train_model(model, train_loader, test_loader, epochs=5)
            heatmap[i, j] = history["test_accs"][-1]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, annot=True, xticklabels=width2, yticklabels=width1, fmt=".2f", cmap="viridis")
    plt.xlabel("Layer 2 Width")
    plt.ylabel("Layer 1 Width")
    plt.title("Grid Search Accuracy Heatmap")
    plt.savefig("width_heatmap.png")
    plt.close()

if __name__ == "__main__":
    experiment_with_widths()
