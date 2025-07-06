"""
Задание 3: Кастомные слои и эксперименты (30 баллов)
"""

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.custom_layers import (
    BottleneckResidualBlock,
    CustomResidualBlock,
    get_custom_layer,
)
from utils.comparison_utils import generate_comprehensive_report
from utils.training_utils import count_parameters, train_model
from utils.visualization_utils import (
    create_summary_table,
    plot_model_comparison,
    plot_parameter_comparison,
    plot_training_history,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def experiment_3_1_custom_layers():
    """3.1 Реализация кастомных слоев"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3.1: КАСТОМНЫЕ СЛОИ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5
    batch_size = 64
    lr = 0.001
    train_loader, test_loader = get_mnist_loaders(batch_size)
    custom_layers = {
        "custom_conv": get_custom_layer(
            "custom_conv",
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
            activation="swish",
        ),
        "attention_conv": get_custom_layer(
            "attention_conv", in_channels=1, out_channels=16, kernel_size=3, padding=1
        ),
        "custom_pool": get_custom_layer("custom_pool", pool_type="lp", kernel_size=2),
        "custom_activation": get_custom_layer(
            "custom_activation", activation_type="mish"
        ),
    }

    results = {}
    for name, layer in custom_layers.items():
        print(f"\nТестируем слой: {name}")

        # Простейшая модель для теста слоя
        class TestNet(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
                self.flatten = torch.nn.Flatten()
                # Динамически определяем размер входа в fc
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, 28, 28)
                    x = self.layer(dummy)
                    x = self.flatten(x)
                    flat_dim = x.shape[1]
                self.fc = torch.nn.Linear(flat_dim, 10)

            def forward(self, x):
                x = self.layer(x)
                x = self.flatten(x)
                return self.fc(x)

        model = TestNet(layer).to(device)
        params_info = count_parameters(model)
        history = train_model(
            model, train_loader, test_loader, epochs=epochs, lr=lr, device=str(
                device)
        )
        results[name] = history
        results[name]["parameters"] = params_info

        print(
            f"Параметров: {params_info['total_params_millions']:.4f}M | Лучшая точность: {max(history['test_accs']):.4f}"
        )
        plot_training_history(
            history,
            title=f"Custom Layer: {name}",
            save_path=f"plots/custom_layers/{name}_history.png",
        )

    os.makedirs("plots/custom_layers", exist_ok=True)
    os.makedirs("results/custom_layers", exist_ok=True)

    plot_model_comparison(
        results,
        "test_accs",
        "Custom Layers: Accuracy",
        "plots/custom_layers/accuracy.png",
    )
    plot_parameter_comparison(
        {
            k: {"total_params_millions": v["parameters"]
                ["total_params_millions"]}
            for k, v in results.items()
        },
        "plots/custom_layers/params.png",
    )
    summary_df = create_summary_table(
        results, "results/custom_layers/summary.csv")

    print("\nСводная таблица:")
    print(summary_df.to_string(index=False))
    return results


def experiment_3_2_residual_blocks():
    """3.2 Эксперименты с Residual блоками"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3.2: RESIDUAL БЛОКИ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 7
    batch_size = 64
    lr = 0.001
    train_loader, test_loader = get_mnist_loaders(batch_size)

    block_variants = {
        "basic_residual": CustomResidualBlock(
            1, 16, stride=1, attention=False, activation="relu"
        ),
        "bottleneck_residual": BottleneckResidualBlock(
            1, 16, stride=1, expansion=2, attention=False, activation="relu"
        ),
        "wide_residual": CustomResidualBlock(
            1, 32, stride=1, attention=False, activation="relu", dropout=0.2
        ),
    }
    results = {}

    for name, block in block_variants.items():
        print(f"\nТестируем блок: {name}")

        class TestNet(torch.nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
                self.flatten = torch.nn.Flatten()
                # Динамически определяем размер входа в fc
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, 28, 28)
                    x = self.block(dummy)
                    x = self.flatten(x)
                    flat_dim = x.shape[1]
                self.fc = torch.nn.Linear(flat_dim, 10)

            def forward(self, x):
                x = self.block(x)
                x = self.flatten(x)
                return self.fc(x)

        model = TestNet(block).to(device)
        params_info = count_parameters(model)
        history = train_model(
            model, train_loader, test_loader, epochs=epochs, lr=lr, device=str(
                device)
        )
        results[name] = history
        results[name]["parameters"] = params_info

        print(
            f"Параметров: {params_info['total_params_millions']:.4f}M | Лучшая точность: {max(history['test_accs']):.4f}"
        )
        plot_training_history(
            history,
            title=f"Residual Block: {name}",
            save_path=f"plots/custom_layers/{name}_history.png",
        )

    plot_model_comparison(
        results,
        "test_accs",
        "Residual Blocks: Accuracy",
        "plots/custom_layers/residual_accuracy.png",
    )

    plot_parameter_comparison(
        {
            k: {"total_params_millions": v["parameters"]
                ["total_params_millions"]}
            for k, v in results.items()
        },
        "plots/custom_layers/residual_params.png",
    )

    summary_df = create_summary_table(
        results, "results/custom_layers/residual_summary.csv"
    )
    print("\nСводная таблица:")
    print(summary_df.to_string(index=False))
    return results


def main():
    print("ЗАДАНИЕ 3: КАСТОМНЫЕ СЛОИ И ЭКСПЕРИМЕНТЫ")
    print("=" * 80)

    os.makedirs("plots/custom_layers", exist_ok=True)
    os.makedirs("results/custom_layers", exist_ok=True)

    custom_layer_results = experiment_3_1_custom_layers()
    residual_block_results = experiment_3_2_residual_blocks()

    all_results = {**custom_layer_results, **residual_block_results}

    print("\nГенерируем комплексный отчет...")
    _ = generate_comprehensive_report(
        all_results, "results/custom_layers/comprehensive"
    )
    print("Эксперименты завершены! Результаты сохранены в папках:")
    print("- plots/custom_layers/: Графики и визуализации")
    print("- results/custom_layers/: Таблицы и данные")


if __name__ == "__main__":
    main()