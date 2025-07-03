
import torch
import torch.nn as nn
from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.trainer import train_model
import matplotlib.pyplot as plt

class FCNetRegularized(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, use_dropout=False, use_batchnorm=False, l2_lambda=0.0):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.l2_lambda = l2_lambda
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            last_size = size
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

def experiment_with_regularization():
    configs = {
        "no regularization": {"dropout": False, "bn": False, "l2": 0.0},
        "dropout": {"dropout": True, "bn": False, "l2": 0.0},
        "batchnorm": {"dropout": False, "bn": True, "l2": 0.0},
        "l2": {"dropout": False, "bn": False, "l2": 1e-4},
        "dropout + bn": {"dropout": True, "bn": True, "l2": 0.0},
        "dropout + l2": {"dropout": True, "bn": False, "l2": 1e-4},
        "bn + l2": {"dropout": False, "bn": True, "l2": 1e-4},
        "all": {"dropout": True, "bn": True, "l2": 1e-4}
    }

    train_loader, test_loader = get_mnist_loaders()
    input_size = 28 * 28
    num_classes = 10
    hidden_sizes = [256, 128, 64]

    results = {}
    for name, cfg in configs.items():
        print(f"Training: {name}")
        model = FCNetRegularized(input_size, num_classes, hidden_sizes,
                                 use_dropout=cfg["dropout"],
                                 use_batchnorm=cfg["bn"],
                                 l2_lambda=cfg["l2"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=cfg["l2"])
        criterion = nn.CrossEntropyLoss()
        history = train_model(model, train_loader, test_loader, epochs=10)
        results[name] = history

    for name, hist in results.items():
        plt.plot(hist["train_accs"], label=f"{name} train", linestyle="--")
        plt.plot(hist["test_accs"], label=f"{name} test")
    plt.title("Train/Test Accuracy with Regularization")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig("reg_accuracy.png")
    plt.close()

if __name__ == "__main__":
    experiment_with_regularization()
