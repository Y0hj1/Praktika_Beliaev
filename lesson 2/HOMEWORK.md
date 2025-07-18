# Домашнее задание к уроку 2: Линейная и логистическая регрессия

## Цель задания
Закрепить навыки работы с PyTorch API, изучить модификацию моделей и работу с различными датасетами.

## Задание 1: Модификация существующих моделей (30 баллов)

Создайте файл `homework_model_modification.py`:

### 1.1 Расширение линейной регрессии (15 баллов)
```python
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
# - Добавьте early stopping
```

### 1.2 Расширение логистической регрессии (15 баллов)
```python
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix
```

## Задание 2: Работа с датасетами (30 баллов)

Создайте файл `homework_datasets.py`:

### 2.1 Кастомный Dataset класс (15 баллов)
```python
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)
```

### 2.2 Эксперименты с различными датасетами (15 баллов)
```python
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию
```

## Задание 3: Эксперименты и анализ (20 баллов)

Создайте файл `homework_experiments.py`:

### 3.1 Исследование гиперпараметров (10 баллов)
```python
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц
```

### 3.2 Feature Engineering (10 баллов)
```python
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью
```

## Дополнительные требования

1. **Код должен быть модульным** - разделите на функции и классы
2. **Документация** - добавьте подробные комментарии и docstring
3. **Визуализация** - создайте графики для анализа результатов
4. **Тестирование** - добавьте unit-тесты для критических функций
5. **Логирование** - используйте logging для отслеживания процесса обучения

## Структура проекта

```
homework/
├── homework_model_modification.py
├── homework_datasets.py
├── homework_experiments.py
├── data/                    # Датасеты
├── models/                  # Сохраненные модели
├── plots/                   # Графики и визуализации
└── README.md               # Описание решения
```

## Срок сдачи
Домашнее задание должно быть выполнено до начала занятия 4.

## Полезные ссылки
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
- [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)

Удачи в выполнении задания! 🚀 