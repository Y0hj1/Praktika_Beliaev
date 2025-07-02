
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----- ЛИНЕЙНАЯ РЕГРЕССИЯ С L1, L2 И EARLY STOPPING -----

class LinearRegressionWithRegularization(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_model(model, X_train, y_train, X_val, y_val, 
                       epochs=100, lr=0.01, l1=0.0, l2=0.0, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        l1_penalty = sum(torch.norm(p, 1) for p in model.parameters())
        loss += l1 * l1_penalty
        loss.backward()
        optimizer.step()

        model.eval()
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model

# ----- ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ С МНОГОКЛАССОВОЙ ПОДДЕРЖКОЙ И МЕТРИКАМИ -----

class LogisticRegressionMulticlass(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def evaluate_classification(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        if len(torch.unique(y_test)) == 2:
            auc = roc_auc_score(y_test.numpy(), probs[:,1].numpy())
            print(f"ROC-AUC: {auc:.4f}")
