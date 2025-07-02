
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# ----- 3.1 ИССЛЕДОВАНИЕ ГИПЕРПАРАМЕТРОВ -----

def experiment_hyperparams(model_class, X, y, input_dim):
    results = []

    for lr in [0.01, 0.001]:
        for batch_size in [16, 32]:
            for opt_name in ['SGD', 'Adam', 'RMSprop']:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

                model = model_class(input_dim)
                criterion = nn.MSELoss()
                optimizer = {
                    'SGD': optim.SGD(model.parameters(), lr=lr),
                    'Adam': optim.Adam(model.parameters(), lr=lr),
                    'RMSprop': optim.RMSprop(model.parameters(), lr=lr)
                }[opt_name]

                dataset = torch.utils.data.TensorDataset(X_train, y_train)
                loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for epoch in range(20):
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = criterion(val_pred, y_val).item()

                results.append((lr, batch_size, opt_name, val_loss))

    for lr, bs, opt, loss in results:
        print(f"LR={lr}, BS={bs}, OPT={opt}, Loss={loss:.4f}")

# ----- 3.2 FEATURE ENGINEERING -----

def add_polynomial_features(X, degree=2):
    poly_features = [X]
    for d in range(2, degree + 1):
        poly_features.append(X ** d)
    return torch.cat(poly_features, dim=1)

def add_interaction_features(X):
    n = X.shape[1]
    features = [X]
    for i in range(n):
        for j in range(i + 1, n):
            features.append((X[:, i] * X[:, j]).unsqueeze(1))
    return torch.cat(features, dim=1)

def add_stat_features(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    std = torch.std(X, dim=1, keepdim=True)
    return torch.cat([X, mean, std], dim=1)

def feature_engineering_pipeline(X):
    X_poly = add_polynomial_features(X)
    X_inter = add_interaction_features(X)
    X_stat = add_stat_features(X)
    return torch.cat([X_poly, X_inter, X_stat], dim=1)
