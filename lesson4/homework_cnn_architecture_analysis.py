"""
Задание 2: Анализ архитектур CNN
"""

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.cnn_models import get_cnn_model
from utils.comparison_utils import generate_comprehensive_report
from utils.training_utils import count_parameters, measure_inference_time, train_model
from utils.visualization_utils import (
    create_summary_table,
    plot_feature_maps,
    plot_gradient_flow,
    plot_inference_time_comparison,
    plot_model_comparison,
    plot_parameter_comparison,
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


def experiment_2_1_kernel_size():
    """2.1 Влияние размера ядра свертки"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2.1: ВЛИЯНИЕ РАЗМЕРА ЯДРА СВЕРТКИ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 64
    lr = 0.001
    train_loader, test_loader = get_mnist_loaders(batch_size)

    kernel_variants = {
        "kernel_3x3": {"kernel_sizes": [3]},
        "kernel_5x5": {"kernel_sizes": [5]},
        "kernel_7x7": {"kernel_sizes": [7]},
        "kernel_combo": {"kernel_sizes": [1, 3]},
    }
    results = {}
    models_info = {}
    for name, params in kernel_variants.items():
        print(f"\nМодель: {name}")

        model = get_cnn_model(
            "different_kernels",
            input_channels=1,
            num_classes=10,
            kernel_sizes=params["kernel_sizes"],
        )
        model.to(device)
        params_info = count_parameters(model)

        history = train_model(
            model, train_loader, test_loader, epochs=epochs, lr=lr, device=str(
                device)
        )
        inference_time = measure_inference_time(
            model, test_loader, str(device))

        results[name] = history
        results[name]["parameters"] = params_info
        results[name]["inference_time"] = inference_time

        models_info[name] = {
            "total_params_millions": params_info["total_params_millions"],
            "inference_time": inference_time,
        }
        print(
            f"Параметров: {params_info['total_params_millions']:.2f}M | Время инференса: {inference_time['mean_time']:.4f}s | Лучшая точность: {max(history['test_accs']):.4f}"
        )
        # Визуализация feature maps первого слоя
        plot_feature_maps(
            model,
            test_loader,
            layer_name="conv_layers.0",
            num_images=4,
            save_path=f"plots/architecture_analysis/feature_maps_{name}.png",
        )

    os.makedirs("plots/architecture_analysis", exist_ok=True)
    os.makedirs("results/architecture_analysis", exist_ok=True)

    plot_model_comparison(
        results,
        "test_accs",
        "MNIST: Влияние размера ядра свертки",
        "plots/architecture_analysis/kernel_accuracy.png",
    )
    plot_parameter_comparison(
        models_info, "plots/architecture_analysis/kernel_params.png"
    )
    plot_inference_time_comparison(
        models_info, "plots/architecture_analysis/kernel_inference.png"
    )
    summary_df = create_summary_table(
        results, "results/architecture_analysis/kernel_summary.csv"
    )

    print("\nСводная таблица:")
    print(summary_df.to_string(index=False))
    return results


def experiment_2_2_depth():
    """2.2 Влияние глубины CNN"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2.2: ВЛИЯНИЕ ГЛУБИНЫ CNN")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 64
    lr = 0.001
    train_loader, test_loader = get_mnist_loaders(batch_size)

    depth_variants = {
        "cnn_2_layers": {"model_type": "simple", "input_channels": 1},
        "cnn_4_layers": {"model_type": "deep", "input_channels": 1},
        # deep с 6+ слоями можно реализовать через параметризацию
        "cnn_6_layers": {"model_type": "deep", "input_channels": 1},
        "cnn_residual": {"model_type": "residual", "input_channels": 1},
    }

    results = {}
    models_info = {}

    for name, params in depth_variants.items():
        print(f"\nМодель: {name}")
        model = get_cnn_model(
            params["model_type"],
            input_channels=params["input_channels"],
            num_classes=10,
        )
        model.to(device)
        params_info = count_parameters(model)

        history = train_model(
            model, train_loader, test_loader, epochs=epochs, lr=lr, device=str(
                device)
        )
        inference_time = measure_inference_time(
            model, test_loader, str(device))

        results[name] = history
        results[name]["parameters"] = params_info
        results[name]["inference_time"] = inference_time

        models_info[name] = {
            "total_params_millions": params_info["total_params_millions"],
            "inference_time": inference_time,
        }
        print(
            f"Параметров: {params_info['total_params_millions']:.2f}M | Время инференса: {inference_time['mean_time']:.4f}s | Лучшая точность: {max(history['test_accs']):.4f}"
        )

        # Визуализация feature maps
        plot_feature_maps(
            model,
            test_loader,
            layer_name="conv1",
            num_images=4,
            save_path=f"plots/architecture_analysis/feature_maps_{name}.png",
        )
        # Анализ градиентов
        plot_gradient_flow(
            model,
            train_loader,
            save_path=f"plots/architecture_analysis/gradient_flow_{name}.png",
        )
    plot_model_comparison(
        results,
        "test_accs",
        "MNIST: Влияние глубины сети",
        "plots/architecture_analysis/depth_accuracy.png",
    )
    plot_parameter_comparison(
        models_info, "plots/architecture_analysis/depth_params.png"
    )
    plot_inference_time_comparison(
        models_info, "plots/architecture_analysis/depth_inference.png"
    )

    summary_df = create_summary_table(
        results, "results/architecture_analysis/depth_summary.csv"
    )

    print("\nСводная таблица:")
    print(summary_df.to_string(index=False))
    return results


def main():
    print("ЗАДАНИЕ 2: АНАЛИЗ АРХИТЕКТУР CNN")
    print("=" * 80)

    os.makedirs("plots/architecture_analysis", exist_ok=True)
    os.makedirs("results/architecture_analysis", exist_ok=True)

    kernel_results = experiment_2_1_kernel_size()
    depth_results = experiment_2_2_depth()

    all_results = {**kernel_results, **depth_results}

    print("\nГенерируем комплексный отчет...")
    _ = generate_comprehensive_report(
        all_results, "results/architecture_analysis/comprehensive"
    )

    print("Эксперименты завершены! Результаты сохранены в папках:")
    print("- plots/architecture_analysis/: Графики и визуализации")
    print("- results/architecture_analysis/: Таблицы и данные")


if __name__ == "__main__":
    main()