import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ----- 2.1 КАСТОМНЫЙ КЛАСС DATASET -----

class CustomCSVDataset(Dataset):
    def __init__(self, filepath, target_column, task_type='regression'):
        data = pd.read_csv(filepath)

        self.y = data[target_column].values
        self.X = data.drop(columns=[target_column])

        self.encoders = {}
        for col in self.X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            self.encoders[col] = le

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        if task_type == 'classification':
            self.y = LabelEncoder().fit_transform(self.y)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32 if task_type == 'regression' else torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- 2.2 ОБУЧЕНИЕ НА CSV-ДАННЫХ -----
if __name__ == "__main__":
    from torch.utils.data import random_split
    from homework_model_modification import LinearRegressionWithRegularization, LogisticRegressionMulticlass, train_linear_model, evaluate_classification
    import torch.optim as optim
    import torch.nn as nn

    # Линейная регрессия на regression_data.csv
    reg_dataset = CustomCSVDataset("regression_data.csv", target_column="target", task_type="regression")
    train_len = int(len(reg_dataset) * 0.8)
    train_set, val_set = random_split(reg_dataset, [train_len, len(reg_dataset)-train_len])

    X_train = torch.stack([x for x, _ in train_set])
    y_train = torch.stack([y for _, y in train_set]).unsqueeze(1)
    X_val = torch.stack([x for x, _ in val_set])
    y_val = torch.stack([y for _, y in val_set]).unsqueeze(1)

    model_reg = LinearRegressionWithRegularization(input_dim=X_train.shape[1])
    train_linear_model(model_reg, X_train, y_train, X_val, y_val)

    # Логистическая регрессия на classification_data.csv
    clf_dataset = CustomCSVDataset("classification_data.csv", target_column="target", task_type="classification")
    train_len = int(len(clf_dataset) * 0.8)
    train_set, test_set = random_split(clf_dataset, [train_len, len(clf_dataset)-train_len])

    X_train = torch.stack([x for x, _ in train_set])
    y_train = torch.stack([y for _, y in train_set])
    X_test = torch.stack([x for x, _ in test_set])
    y_test = torch.stack([y for _, y in test_set])

    model_clf = LogisticRegressionMulticlass(input_dim=X_train.shape[1], num_classes=2)
    optimizer = optim.Adam(model_clf.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model_clf.train()
        optimizer.zero_grad()
        out = model_clf(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    evaluate_classification(model_clf, X_test, y_test)
