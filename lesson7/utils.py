import torch

import time
from typing import Callable, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np

import psutil

NUM_WARMUP_ITERATIONS = 10


def cuda_timer(func):
    def wrapper(*args, **kwargs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream=torch.cuda.current_stream())
        result = func(*args, **kwargs)
        end_time.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return result, start_time.elapsed_time(end_time)

    return wrapper


def cpu_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        return result, end_time * 1000  # ms

    return wrapper


def gpu_mem_usage(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        after = torch.cuda.memory_allocated()
        return result, (after - before) / (1024 ** 2)  # MB

    return wrapper


def cpu_mem_usage(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        before = process.memory_info().rss
        result = func(*args, **kwargs)
        after = process.memory_info().rss
        return result, (after - before) / (1024 ** 2)  # MB

    return wrapper


def run_test(
        model_wrapper: Callable,
        data_preprocess: Callable = None,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_runs: int = 1000,
        min_batch_size: int = 1,
        max_batch_size: int = 1,
        batch_step: int = 1,
        dataset: Dataset = None,
        timer_type: str = 'cuda'
) -> Dict[Tuple[int, int, int], float]:
    shapes = [(size, *input_shape) for size in range(min_batch_size, max_batch_size + 1, batch_step)]
    results = {}
    timer = cuda_timer if timer_type == 'cuda' else cpu_timer

    for shape in shapes:
        dataloader = DataLoader(dataset, batch_size=shape[0], shuffle=False, drop_last=True)

        # Warm-up
        with torch.no_grad():
            for _ in range(NUM_WARMUP_ITERATIONS):
                if data_preprocess:
                    dummy_input = data_preprocess(torch.randn(shape, computing_device='cuda' if timer_type == 'cuda' else 'cpu'))
                else:
                    dummy_input = torch.randn(shape, computing_device='cuda' if timer_type == 'cuda' else 'cpu')
                model_wrapper(dummy_input)

        times = []
        for _ in range(num_runs):
            for batch in dataloader:
                image = batch[0]
                if timer_type == 'cuda':
                    image = image.cuda()

                if data_preprocess:
                    image = data_preprocess(image)

                _, time_taken = timer(model_wrapper)(image)
                times.append(time_taken)

        # Обработка результатов: убираем выбросы
        times = np.array(times)
        q10, q90 = np.percentile(times, [10, 90])
        filtered = times[(times >= q10) & (times <= q90)]
        avg_time = np.mean(filtered).item()

        results[shape] = avg_time

    return results