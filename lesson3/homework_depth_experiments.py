
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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

class FCRegNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            last_size = size
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def experiment_with_depths():
    train_loader, test_loader = get_mnist_loaders()
    input_size = 28 * 28
    output_size = 10

    depths = [1, 2, 3, 5, 7]
    results = []
    for depth in depths:
        hidden_sizes = [128] * depth
        print(f"Training model: {depth} layer{'s' if depth > 1 else ''}")
        model = FCNet(input_size, output_size, hidden_sizes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        history = train_model(model, train_loader, test_loader, epochs=10)
        results.append({
            "depth": depth,
            "train_acc": history["train_accs"][-1],
            "test_acc": history["test_accs"][-1],
            "history": history
        })

    for result in results:
        name = f"{result['depth']} layers"
        plt.plot(result["history"]["train_accs"], label=f"{name} train", linestyle="--")
        plt.plot(result["history"]["test_accs"], label=f"{name} test")
    plt.title("Accuracy vs Depth")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("depth_accuracy.png")
    plt.close()

if __name__ == "__main__":
    experiment_with_depths()
