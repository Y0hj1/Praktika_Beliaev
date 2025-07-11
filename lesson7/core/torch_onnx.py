import torch

import onnx

import onnxruntime as ort

import numpy as np

import os

import time
from typing import Tuple, Optional, Dict

import onnxconverter_common
from torch.utils.data import Dataset, DataLoader

try:
    from core.net import Resnet18
    from core.datasets import CustomImageDataset
    from core.utils import run_test
except ImportError:
    from net import Resnet18
    from datasets import CustomImageDataset
    from utils import run_test

# ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)

def convert_to_onnx(
    model_path: str,
    output_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'fp32',
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    opt_batch_size: int = 1,
    opset_version: int = 11,
    **kwargs
) -> str:
    """
    Конвертирует PyTorch модель в ONNX формат с поддержкой динамических батчей
    
    Args:
        model_path: Путь к сохраненной PyTorch модели
        output_path: Путь для сохранения ONNX модели
        input_shape: Форма входного тензора (channels, height, width)
        precision: Точность (fp32, fp16)
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        opt_batch_size: Оптимальный размер батча
        opset_version: Версия ONNX операторов
        optimize: Применять ли оптимизацию при конвертации
    
    Returns:
        Путь к сохраненной ONNX модели
    """
    # Загружаем модель
    net = Resnet18()
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Подготавливаем входные данные для разных размеров батчей
    min_shape = (min_batch_size, *input_shape)
    opt_shape = (opt_batch_size, *input_shape)
    max_shape = (max_batch_size, *input_shape)
    
    # Настройки для динамического батча
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # Настройки точности - вход всегда float32, выход может быть fp16
    dummy_input = torch.randn(max_shape, dtype=torch.float32)
    
    # Экспортируем в ONNX
    torch.onnx.export(
        net,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # Проверяем модель
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if precision == 'fp16':
        onnxconverter_common.float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=True
        )
        onnx.save(onnx_model, output_path)
    
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    
    
    print(f"Модель успешно конвертирована в ONNX: {output_path}")
    print(f"Поддерживаемые размеры батчей: {min_batch_size} - {max_batch_size}")
    print(f"Точность: {precision}")
    return output_path


def test_onnx_model_cpu_timer(
    onnx_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    **kwargs
) -> Dict[Tuple[int, int, int], float]:
    """
    Тестирует ONNX модель с использованием CPU таймера
    
    Args:
        onnx_path: Путь к ONNX модели
        input_shape: Форма входного тензора
        num_runs: Количество прогонов для усреднения
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        dataloader: Даталоадер для тестирования
    
    Returns:
        Словарь с результатами тестирования для каждого размера батча
    """
    # Создаем ONNX Runtime сессию
    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, providers=providers, sess_options=session_options)
    
    def run_inference(pictures):
        return session.run(None, {'input': pictures})

    def data_preprocess(pictures):
        return pictures.cpu().numpy()
    
    return run_test(
        model_wrapper=run_inference,
        data_preprocess=data_preprocess,
        input_shape=input_shape,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_step=batch_step,
        dataset=dataset,
        timer_type='cpu'
    )


def test_onnx_model_cuda_timer(
    onnx_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    **kwargs
) -> Dict[Tuple[int, int, int], float]:
    """
    Тестирует ONNX модель с использованием CUDA таймера
    
    Args:
        onnx_path: Путь к ONNX модели
        input_shape: Форма входного тензора
        num_runs: Количество прогонов для усреднения
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        dataloader: Даталоадер для тестирования
    
    Returns:
        Словарь с результатами тестирования для каждого размера батча
    """
    # Создаем ONNX Runtime сессию с CUDA
    providers = ['CUDAExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, providers=providers, sess_options=session_options)
    
    def run_inference(pictures):
        return session.run(None, {'input': pictures})
    
    def data_preprocess(pictures):
        return pictures.cpu().numpy()
    
    return run_test(
        model_wrapper=run_inference,
        data_preprocess=data_preprocess,
        input_shape=input_shape,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_step=batch_step,
        dataset=dataset,
        timer_type='cuda'
    )


if __name__ == '__main__':
    # Пример использования
    model_path = './weights/best_resnet18.pth'
    onnx_path = './weights/resnet18.onnx'
    optimized_path = './weights/resnet18_optimized.onnx'
    
    # Создаем dataloader если есть данные
    try:
        train_dataset = CustomImageDataset(root_dir='../lesson5_augmentations/data/train', target_size=(224, 224))
        loader_train = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=8)
        print("Используем реальный dataloader")
    except:
        loader_train = None
        print("Используем синтетический dataloader")
    
    if os.path.exists(model_path):
        # Конвертируем в ONNX с поддержкой динамических батчей и оптимизацией
        convert_to_onnx(
            model_path, 
            onnx_path, 
            precision='fp32',
            min_batch_size=32, 
            max_batch_size=32, 
            opt_batch_size=32,
        )
        
        # Тестируем модель с CPU таймером
        print("\nТестирование ONNX модели с CPU таймером:")
        cpu_results = test_onnx_model_cpu_timer(
            optimized_path, 
            dataloader=loader_train,
            min_batch_size=32, 
            max_batch_size=32, 
            input_shape=(3, 224, 224), 
            num_runs=50
        )
        
        for shape, avg_time in cpu_results.items():
            print(f"Shape: {shape}, CPU Time: {avg_time:.4f} seconds")
        
        # Тестируем модель с CUDA таймером
        if torch.cuda.is_available():
            print("\nТестирование ONNX модели с CUDA таймером:")
            cuda_results = test_onnx_model_cuda_timer(
                optimized_path, 
                dataloader=loader_train,
                min_batch_size=32, 
                max_batch_size=32, 
                input_shape=(3, 224, 224), 
                num_runs=50
            )
            
            for shape, avg_time in cuda_results.items():
                print(f"Shape: {shape}, CUDA Time: {avg_time:.4f} seconds")