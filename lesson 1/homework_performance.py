import torch
import time

# Проверка наличия GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3.1 Подготовка данных ---
sizes = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
matrices = [torch.rand(size) for size in sizes]

# --- 3.2 Функция измерения времени ---
def measure_cpu(op, *args):
    start = time.time()
    result = op(*args)
    end = time.time()
    return (end - start) * 1000  # мс

def measure_gpu(op, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = op(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # мс

# --- 3.3 Сравнение операций ---
ops = {
    "Матричное умножение": torch.matmul,
    "Поэлементное сложение": lambda a, b: a + b,
    "Поэлементное умножение": lambda a, b: a * b,
    "Транспонирование": lambda a: a.transpose(-1, -2),
    "Сумма элементов": lambda a: a.sum()
}

print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение'}")
print("-" * 60)
for name, op in ops.items():
    cpu_time = measure_cpu(op, matrices[0], matrices[0]) if "элемент" in name or "умножение" in name else measure_cpu(op, matrices[0])
    if torch.cuda.is_available():
        tensor_gpu = matrices[0].to(device)
        gpu_time = measure_gpu(op, tensor_gpu, tensor_gpu) if "элемент" in name or "умножение" in name else measure_gpu(op, tensor_gpu)
        speedup = cpu_time / gpu_time if gpu_time else float("inf")
        print(f"{name:<25} | {cpu_time:>9.2f} | {gpu_time:>9.2f} | {speedup:.1f}x")
    else:
        print(f"{name:<25} | {cpu_time:>9.2f} | {'—':>9} | GPU недоступен")

# --- 3.4 Анализ результатов ---
"""
- Операции которые получают наибольшое ускорение на GPU - это матричное умножение и сложение за счёт параллельных вычислений
- Маленькие тензоры могут работать медленнее на GPU из-за задержек копирования
- Чем больше размер тензора, тем выше эффективность параллельных вычислений на графическом процессоре. Для небольших матриц использование GPU 
  может не дать выигрыша по времени, а в некоторых случаях даже привести к замедлению из-за накладных расходов
- При передаче данных между CPU и GPU происходит копирование, что требует значительного времени. Такие передачи особенно замедляют работу,
  если выполняются часто и с небольшими объемами данных
"""
