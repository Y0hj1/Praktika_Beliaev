import torch

# --- 2.1 Простые вычисления с градиентами ---
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2 * x * y * z
f.backward()

# Градиенты: x.grad, y.grad, z.grad

# --- 2.2 MSE и градиенты ---

def compute_mse_and_grads(x: torch.Tensor, y_true: torch.Tensor, w_val: float, b_val: float):
    """
    Вычисляет MSE и градиенты по параметрам w и b для линейной модели y_pred = w * x + b.

    Args:
        x (torch.Tensor): входные данные
        y_true (torch.Tensor): истинные значения
        w_val (float): начальное значение w
        b_val (float): начальное значение b

    Returns:
        tuple: mse.item(), grad_w.item(), grad_b.item()
    """
    # Преобразуем входные параметры в тензоры с градиентами
    w = torch.tensor(w_val, requires_grad=True)
    b = torch.tensor(b_val, requires_grad=True)

    # Линейная модель
    y_pred = w * x + b

    # Функция потерь (MSE)
    loss = torch.mean((y_pred - y_true) ** 2)

    # Вычисление градиентов
    grads = torch.autograd.grad(loss, [w, b])

    return loss.item(), grads[0].item(), grads[1].item()

#Пример использования:
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])
w_init = 0.0
b_init = 0.0

mse_val, grad_w, grad_b = compute_mse_and_grads(x, y_true, w_init, b_init)
print(f"MSE: {mse_val:.4f}")
print(f"Градиент по w: {grad_w:.4f}")
print(f"Градиент по b: {grad_b:.4f}")


# --- 2.3 Цепное правило ---
def composite_function_grad(x_val: float):
    """
    Вычисляет градиент функции f(x) = sin(x^2 + 1) с помощью torch.autograd.grad

    Args:
        x_val (float): значение переменной x

    Returns:
        float: значение градиента df/dx
    """
    x = torch.tensor(x_val, requires_grad=True)
    f = torch.sin(x**2 + 1)
    grad, = torch.autograd.grad(f, x)
    return grad.item()

# Пример использования:
x_value = 1.0
grad_result = composite_function_grad(x_value)
print(f"Градиент df/dx при x = {x_value}: {grad_result:.6f}")