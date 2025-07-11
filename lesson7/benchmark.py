import torch
from typing Tuple, List, Callable
from torch.utils.data import Dataset
from tabulate import tabulate

import matplotlib.pyplot as plt

import numpy as np

import os

from core.torch_onnx import test_onnx_model_cpu_timer, test_onnx_model_cuda_timer, convert_to_onnx
from core.torch_trt import test_torch_trt_model, convert_to_torch_trt
from core.datasets import CustomImageDataset, RandomImageDataset
from core.utils import run_test, gpu_mem_usage, cpu_mem_usage
from core.net import Resnet18


def make_res_table(results: List[Dict], test_fn_names: Dict[str, str], gpu_name: str):
    headers = [
        'function',
        'dataloader_type',
        'precision',
        'computing_device',
        'image_size',
        'batch_size',
        'time_per_batch_ms',
        'time_per_image_ms',
        'fps',
        'allocated_memory',
        'speedup'
    ]

    table = []
    base_times = {}
    speedup_data = []
    fps_data = {'torch': [], 'onnx': [], 'torch_trt': []}
    image_sizes = set()
    batch_sizes = set()

    # Сбор базовых времен (PyTorch FP32 CUDA)
    for result in results:
        if (result['test_function'] == 'test_torch_model' and
                result['precision'] == 'fp32' and
                result['timer_type'] == 'cuda'):
            for shape, time_res in result['results'].items():
                batch_size = shape[0]
                image_size = shape[2]  # H dimension
                key = (image_size, batch_size)
                base_times[key] = time_res

    # Формирование таблицы и данных для графиков
    for result in results:
        for shape, time_res in result['results'].items():
            batch_size = shape[0]
            image_size = shape[2]  # H dimension
            key = (image_size, batch_size)

            # Расчет метрик
            time_per_image = time_res / batch_size
            fps = (batch_size * 1000) / time_res
            base_time = base_times.get(key, 0)
            speedup = base_time / time_res if base_time else 0

            # Сохранение данных для графиков
            func_name = test_fn_names[result['test_function']]
            fps_data[func_name].append(fps)
            speedup_data.append({
                'function': func_name,
                'image_size': image_size,
                'batch_size': batch_size,
                'speedup': speedup
            })
            image_sizes.add(image_size)
            batch_sizes.add(batch_size)

            table.append([
                func_name,
                result['dataloader'],
                result['precision'],
                result['timer_type'],
                image_size,
                batch_size,
                f'{time_res:.3f}',
                f'{time_per_image:.3f}',
                f'{fps:.1f}',
                f'{result["allocated_memory"]:.1f} MB',
                f'{speedup:.1f}x' if speedup else 'N/A'
            ])

    # Сортировка таблицы
    table.sort(key=lambda x: (x[0], x[4], x[5]))

    # Сохранение таблицы
    with open('../ДЗ №5/results.md', 'w') as f:
        f.write(f"## GPU: {gpu_name}\n\n")
        f.write(tabulate(table, headers=headers, tablefmt='github'))

    # Построение графиков
    plot_results(image_sizes, batch_sizes, fps_data, speedup_data)

    return table


def plot_results(image_sizes, batch_sizes, fps_data, speedup_data):
    # График FPS vs Размер изображения
    plt.figure(figsize=(12, 8))
    for func, fps in fps_data.items():
        if fps:
            plt.plot(sorted(image_sizes), fps[:len(image_sizes)], 'o-', label=func)
    plt.title('FPS vs Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True)
    plt.savefig('fps_vs_image_size.png')
    plt.close()

    # График FPS vs Размер батча
    plt.figure(figsize=(12, 8))
    for func, fps in fps_data.items():
        if fps and len(fps) > len(image_sizes):
            plt.plot(sorted(batch_sizes), fps[len(image_sizes):], 'o-', label=func)
    plt.title('FPS vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True)
    plt.savefig('fps_vs_batch_size.png')
    plt.close()

    # График Ускорение vs Размер изображения
    plt.figure(figsize=(12, 8))
    speedup_by_func = {}
    for item in speedup_data:
        if item['function'] not in speedup_by_func:
            speedup_by_func[item['function']] = []
        speedup_by_func[item['function']].append(item['speedup'])

    for func, speeds in speedup_by_func.items():
        plt.plot(sorted(image_sizes), speeds[:len(image_sizes)], 'o-', label=func)
    plt.title('Speedup vs Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Speedup (x)')
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup_vs_image_size.png')
    plt.close()


def test_torch_model(
        net: torch.nn.Module,
        dataset: Dataset,
        batch_step: int = 1,
        num_runs: int = 50,
        min_batch_size: int = 1,
        max_batch_size: int = 1,
        precision: str = 'fp16',
        timer_type: str = 'cuda',
        **kwargs
):
    net.eval()
    net = net.to('cuda')
    dtype = torch.float16 if precision == 'fp16' else torch.float32

    def model_wrapper(input_data):
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
            return net(input_data)

    return run_test(
        model_wrapper=model_wrapper,
        input_shape=kwargs['input_shape'],
        batch_step=batch_step,
        dataset=dataset,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        timer_type=timer_type
    )


def test_onnx(
        model_path: str,
        dataset: Dataset,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_runs: int = 50,
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        batch_step: int = 1,
        precision: str = 'fp32',
        timer_type: str = 'cuda',
        **kwargs
):
    onnx_path = model_path.replace('.pth', '.onnx')
    if not os.path.exists(onnx_path):
        convert_to_onnx(
            model_path=model_path,
            output_path=onnx_path,
            input_shape=input_shape,
            precision=precision,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )

    if timer_type == 'cuda':
        return test_onnx_model_cuda_timer(
            onnx_path=onnx_path,
            input_shape=input_shape,
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
    elif timer_type == 'cpu' and precision == 'fp32':
        return test_onnx_model_cpu_timer(
            onnx_path=onnx_path,
            input_shape=input_shape,
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )


def test_torch_trt(
        model_path: str,
        dataset: Dataset,
        batch_step: int = 1,
        num_runs: int = 50,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        precision: str = 'fp32',
        timer_type: str = 'cuda',
        **kwargs
):
    trt_path = model_path.replace('.pth', '.trt')
    if not os.path.exists(trt_path):
        net = convert_to_torch_trt(
            model_path=model_path,
            input_shape=input_shape,
            precision=precision,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )
    else:
        net = torch.jit.load(trt_path)

    if timer_type == 'cuda':
        return test_torch_trt_model(
            net=net,
            input_shape=input_shape,
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )


def benchmark_models(
        sizes: List[int] = [224, 256, 384, 512],
        num_runs: int = 10,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        opt_batch_size: int = 32,
        batch_step: int = 32,
):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Testing on: {gpu_name}")

    results = []
    test_functions = [
        test_torch_model,
        test_onnx,
        test_torch_trt
    ]

    test_fn_names = {
        'test_torch_model': 'torch',
        'test_onnx': 'onnx',
        'test_torch_trt': 'torch_trt'
    }

    for size in sizes:
        model_path = f'./weights/best_resnet18_{size}.pth'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        net = Resnet18()
        try:
            net.load_state_dict(torch.load(model_path))
        except:
            print(f"Error loading net: {model_path}")
            continue

        real_dataset = CustomImageDataset(root_dir='../lesson5/data/test', target_size=(size, size))
        dummy_dataset = RandomImageDataset(target_size=(3, size, size))

        static_kwargs = {
            'net': net,
            'input_shape': (3, size, size),
            'model_path': model_path,
            'min_batch_size': min_batch_size,
            'max_batch_size': max_batch_size,
            'opt_batch_size': opt_batch_size,
            'batch_step': batch_step,
            'num_runs': num_runs,
        }

        kwargs = {
            'datasets': [dummy_dataset],  # Используем только синтетические данные для скорости
            'precisions': ['fp16', 'fp32'],
            'timer_types': ['cuda']
        }

        for precision in kwargs['precisions']:
            for test_function in test_functions:
                for dataset in kwargs['datasets']:
                    for timer_type in kwargs['timer_types']:
                        mem_usage = gpu_mem_usage if timer_type == 'cuda' else cpu_mem_usage
                        print(f"Testing: size={size}, func={test_function.__name__}, "
                              f"dataloader={dataset.__class__.__name__}, "
                              f"precision={precision}, timer={timer_type}")

                        try:
                            result, allocated_memory = mem_usage(test_function)(
                                **static_kwargs,
                                dataset=dataset,
                                precision=precision,
                                timer_type=timer_type
                            )
                        except Exception as e:
                            print(f"Error during testing: {e}")
                            continue

                        if result is None:
                            continue

                        data = {
                            'test_function': test_function.__name__,
                            'dataloader': 'real' if isinstance(dataset, CustomImageDataset) else 'dummy',
                            'precision': precision,
                            'timer_type': timer_type,
                            'results': result,
                            'allocated_memory': allocated_memory
                        }
                        results.append(data)

    return make_res_table(results, test_fn_names, gpu_name)


if __name__ == '__main__':
    benchmark_models(
        sizes=[224, 256, 384, 512],
        num_runs=10,
        min_batch_size=1,
        max_batch_size=64,
        opt_batch_size=32,
        batch_step=32,
    )