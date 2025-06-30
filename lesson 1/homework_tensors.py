import torch

# --- 1.1 Создание тензоров ---
tensor1 = torch.rand(3, 4)
tensor2 = torch.zeros(2, 3, 4)
tensor3 = torch.ones(5, 5)
tensor4 = torch.arange(16).reshape(4, 4)

# --- 1.2 Операции с тензорами ---
A = torch.rand(3, 4)
B = torch.rand(4, 3)

A_T = A.T
matmul_result = A @ B
elementwise_mul = A * B.T
sum_elements = A.sum()

# --- 1.3 Индексация и срезы ---
tensor5 = torch.rand(5, 5, 5)
first_row = tensor5[0, :, :]
last_column = tensor5[:, :, -1]
center_slice = tensor5[2:4, 2:4, 2:4]
even_indices = tensor5[::2, ::2, ::2]

# --- 1.4 Работа с формами ---
flat_tensor = torch.arange(24)
shape_2x12 = flat_tensor.view(2, 12)
shape_3x8 = flat_tensor.view(3, 8)
shape_4x6 = flat_tensor.view(4, 6)
shape_2x3x4 = flat_tensor.view(2, 3, 4)
shape_2x2x2x3 = flat_tensor.view(2, 2, 2, 3)
