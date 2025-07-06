"""
Задание 1: Сравнение CNN и полносвязных сетей
"""

from utils.comparison_utils import generate_comprehensive_report
from utils.visualization_utils import (
    plot_model_comparison, plot_confusion_matrix,
    plot_parameter_comparison, plot_inference_time_comparison, create_summary_table
)
from utils.training_utils import train_model, measure_inference_time, count_parameters, save_training_results
from models.cnn_models import get_cnn_model
from models.fc_models import get_fc_model
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import sys


# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


def get_mnist_loaders(batch_size=64):
    """Загружает данные MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar_loaders(batch_size=64):
    """Загружает данные CIFAR-10"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def experiment_1_1_mnist_comparison():
    """Задание 1.1: Сравнение на MNIST"""
    print("=" * 60)
    print("ЗАДАНИЕ 1.1: СРАВНЕНИЕ НА MNIST")
    print("=" * 60)

    # Параметры эксперимента
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 15
    batch_size = 64
    lr = 0.001

    # Загружаем данные
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Определяем модели для сравнения
    models_config = {
        'FC_Simple': {
            'type': 'fc',
            'model_type': 'simple',
            'input_size': 784,
            'num_classes': 10
        },
        'FC_Deep': {
            'type': 'fc',
            'model_type': 'deep',
            'input_size': 784,
            'num_classes': 10
        },
        'CNN_Simple': {
            'type': 'cnn',
            'model_type': 'simple',
            'input_channels': 1,
            'num_classes': 10
        },
        'CNN_Residual': {
            'type': 'cnn',
            'model_type': 'residual',
            'input_channels': 1,
            'num_classes': 10
        }
    }

    results = {}
    models_info = {}

    # Обучаем каждую модель
    for model_name, config in models_config.items():
        print(f"\nОбучаем модель: {model_name}")
        print("-" * 40)

        # Создаем модель
        if config['type'] == 'fc':
            model = get_fc_model(
                config['model_type'], config['input_size'], config['num_classes'])
        else:
            model = get_cnn_model(
                config['model_type'], config['input_channels'], config['num_classes'])

        model.to(device)

        # Подсчитываем параметры
        params_info = count_parameters(model)

        # Обучаем модель
        history = train_model(model, train_loader, test_loader,
                              epochs=epochs, lr=lr, device=str(device))

        # Измеряем время инференса
        inference_time = measure_inference_time(
            model, test_loader, str(device))

        # Сохраняем результаты
        results[model_name] = history
        results[model_name]['parameters'] = params_info
        results[model_name]['inference_time'] = inference_time

        models_info[model_name] = {
            'total_params_millions': params_info['total_params_millions'],
            'inference_time': inference_time
        }

        print(f"Параметров: {params_info['total_params_millions']:.2f}M")
        print(f"Время инференса: {inference_time['mean_time']:.4f}s")
        print(f"Лучшая точность: {max(history['test_accs']):.4f}")

    # Создаем визуализации
    os.makedirs('plots/mnist_comparison', exist_ok=True)
    os.makedirs('results/mnist_comparison', exist_ok=True)

    # График сравнения точности
    plot_model_comparison(results, 'test_accs', 'MNIST: Сравнение точности моделей',
                          'plots/mnist_comparison/accuracy_comparison.png')

    # График сравнения потерь
    plot_model_comparison(results, 'test_losses', 'MNIST: Сравнение потерь моделей',
                          'plots/mnist_comparison/loss_comparison.png')

    # Сравнение параметров
    plot_parameter_comparison(
        models_info, 'plots/mnist_comparison/parameters_comparison.png')

    # Сравнение времени инференса
    plot_inference_time_comparison(
        models_info, 'plots/mnist_comparison/inference_time_comparison.png')

    # Создаем сводную таблицу
    summary_df = create_summary_table(
        results, 'results/mnist_comparison/summary.csv')
    print("\nСводная таблица результатов:")
    print(summary_df.to_string(index=False))

    # Сохраняем результаты
    save_training_results(results, 'results/mnist_comparison/results.pkl')

    return results


def experiment_1_2_cifar_comparison():
    """Задание 1.2: Сравнение на CIFAR-10"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 1.2: СРАВНЕНИЕ НА CIFAR-10")
    print("=" * 60)

    # Параметры эксперимента
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    batch_size = 64
    lr = 0.001

    # Загружаем данные
    train_loader, test_loader = get_cifar_loaders(batch_size)

    # Определяем модели для сравнения
    models_config = {
        'FC_Deep': {
            'type': 'fc',
            'model_type': 'deep',
            'input_size': 3072,
            'num_classes': 10
        },
        'CNN_CIFAR': {
            'type': 'cnn',
            'model_type': 'cifar',
            'input_channels': 3,
            'num_classes': 10
        },
        'CNN_Deep': {
            'type': 'cnn',
            'model_type': 'deep',
            'input_channels': 3,
            'num_classes': 10
        },
        'CNN_Residual': {
            'type': 'cnn',
            'model_type': 'residual',
            'input_channels': 3,
            'num_classes': 10
        }
    }

    results = {}
    models_info = {}

    # Обучаем каждую модель
    for model_name, config in models_config.items():
        print(f"\nОбучаем модель: {model_name}")
        print("-" * 40)

        # Создаем модель
        if config['type'] == 'fc':
            model = get_fc_model(
                config['model_type'], config['input_size'], config['num_classes'])
        else:
            model = get_cnn_model(
                config['model_type'], config['input_channels'], config['num_classes'])

        model.to(device)

        # Подсчитываем параметры
        params_info = count_parameters(model)

        # Обучаем модель
        history = train_model(model, train_loader, test_loader,
                              epochs=epochs, lr=lr, device=str(device))

        # Измеряем время инференса
        inference_time = measure_inference_time(
            model, test_loader, str(device))

        # Сохраняем результаты
        results[model_name] = history
        results[model_name]['parameters'] = params_info
        results[model_name]['inference_time'] = inference_time

        models_info[model_name] = {
            'total_params_millions': params_info['total_params_millions'],
            'inference_time': inference_time
        }

        print(f"Параметров: {params_info['total_params_millions']:.2f}M")
        print(f"Время инференса: {inference_time['mean_time']:.4f}s")
        print(f"Лучшая точность: {max(history['test_accs']):.4f}")

        # Анализируем переобучение
        train_acc = max(history['train_accs'])
        test_acc = max(history['test_accs'])
        overfitting_gap = train_acc - test_acc
        print(f"Переобучение: {overfitting_gap:.4f}")

        # Создаем confusion matrix для лучшей модели
        if test_acc == max([results[m]['test_accs'][-1] for m in results.keys()]):
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
            os.makedirs('plots/cifar_comparison', exist_ok=True)
            plot_confusion_matrix(history['predictions'], history['targets'],
                                  class_names, f'Confusion Matrix - {model_name}',
                                  f'plots/cifar_comparison/confusion_matrix_{model_name}.png')

    # Создаем визуализации
    os.makedirs('plots/cifar_comparison', exist_ok=True)
    os.makedirs('results/cifar_comparison', exist_ok=True)

    # График сравнения точности
    plot_model_comparison(results, 'test_accs', 'CIFAR-10: Сравнение точности моделей',
                          'plots/cifar_comparison/accuracy_comparison.png')

    # График сравнения потерь
    plot_model_comparison(results, 'test_losses', 'CIFAR-10: Сравнение потерь моделей',
                          'plots/cifar_comparison/loss_comparison.png')

    # Сравнение параметров
    plot_parameter_comparison(
        models_info, 'plots/cifar_comparison/parameters_comparison.png')

    # Сравнение времени инференса
    plot_inference_time_comparison(
        models_info, 'plots/cifar_comparison/inference_time_comparison.png')

    # Создаем сводную таблицу
    summary_df = create_summary_table(
        results, 'results/cifar_comparison/summary.csv')
    print("\nСводная таблица результатов:")
    print(summary_df.to_string(index=False))

    # Сохраняем результаты
    save_training_results(results, 'results/cifar_comparison/results.pkl')

    return results


def analyze_gradient_flow():
    """Анализ градиентов для лучших моделей"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ ГРАДИЕНТОВ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем лучшие модели
    from utils.visualization_utils import plot_gradient_flow

    # MNIST
    train_loader, _ = get_mnist_loaders(64)

    # Лучшая CNN модель для MNIST
    cnn_model = get_cnn_model('residual', 1, 10)
    cnn_model.to(device)

    plot_gradient_flow(cnn_model, train_loader,
                       'plots/mnist_comparison/gradient_flow.png')

    # CIFAR-10
    train_loader, _ = get_cifar_loaders(64)

    # Лучшая CNN модель для CIFAR-10
    cnn_model = get_cnn_model('deep', 3, 10)
    cnn_model.to(device)

    plot_gradient_flow(cnn_model, train_loader,
                       'plots/cifar_comparison/gradient_flow.png')


def main():
    """Основная функция"""
    print("ЗАДАНИЕ 1: СРАВНЕНИЕ CNN И ПОЛНОСВЯЗНЫХ СЕТЕЙ")
    print("=" * 80)

    # Создаем директории для результатов
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Задание 1.1: Сравнение на MNIST
    mnist_results = experiment_1_1_mnist_comparison()

    # Задание 1.2: Сравнение на CIFAR-10
    cifar_results = experiment_1_2_cifar_comparison()

    # Анализ градиентов
    analyze_gradient_flow()

    # Генерируем комплексный отчет
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ КОМПЛЕКСНОГО ОТЧЕТА")
    print("=" * 60)

    # Объединяем результаты
    all_results = {**mnist_results, **cifar_results}

    # Генерируем отчет
    _ = generate_comprehensive_report(
        all_results, 'results/comparison_analysis')

    print("Эксперименты завершены! Результаты сохранены в папках:")
    print("- plots/: Графики и визуализации")
    print("- results/: Таблицы и данные")
    print("- training.log: Логи обучения")


if __name__ == "__main__":
    main()