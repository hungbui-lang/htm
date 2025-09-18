import numpy as np
import pandas as pd
import argparse
import time
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# Hàm kích hoạt và đạo hàm
# -------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# -------------------------------
# Hàm mất mát cross-entropy
# -------------------------------
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    eps = 1e-9
    return -np.sum(y_true * np.log(y_pred + eps)) / m

# -------------------------------
# Mạng MLP tự cài đặt
# -------------------------------
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        pre_activations = []
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = relu(z)
            activations.append(a)
        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        a = softmax(z)
        activations.append(a)
        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        grads_w = []
        grads_b = []
        dz = activations[-1] - y_true
        for i in reversed(range(len(self.weights))):
            dw = activations[i].T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i != 0:
                dz = (dz @ self.weights[i].T) * relu_derivative(pre_activations[i-1])
        return grads_w, grads_b

    def update(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, y, X_val, y_val, epochs=100, batch_size=32, verbose=True, early_stop_patience=10):
        history = []
        best_val_loss = float("inf")
        patience = 0
        for epoch in range(epochs):
            # Mini-batch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]
                activations, pre_activations = self.forward(X_batch)
                grads_w, grads_b = self.backward(activations, pre_activations, y_batch)
                self.update(grads_w, grads_b)
            # Evaluate
            train_pred, _ = self.forward(X)
            train_loss = cross_entropy(y, train_pred)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y, axis=1))
            val_pred, _ = self.forward(X_val)
            val_loss = cross_entropy(y_val, val_pred)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            history.append((epoch+1, train_loss, train_acc, val_loss, val_acc))
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print("Early stopping triggered.")
                    break
        return history

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

# -------------------------------
# Tiền xử lý dữ liệu
# -------------------------------
def preprocess_data(df, target_col="OUTPUT Grade"):
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    # One-hot encode tất cả
    X_enc = pd.get_dummies(X.astype(str))
    num_classes = len(np.unique(y))
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return X_enc.values.astype(float), y_onehot

# -------------------------------
# Xuất báo cáo PDF
# -------------------------------
def export_report(results, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Báo cáo MLP - Student Performance", styles["Title"]))
    elements.append(Spacer(1, 12))
    data = [["Cấu hình", "Thời gian (s)", "Epoch", "Val Loss", "Val Acc"]]
    for r in results:
        data.append([r["config"], f"{r['time']:.2f}", r["epochs"], f"{r['val_loss']:.4f}", f"{r['val_acc']:.4f}"])
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    doc.build(elements)

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to StudentPerformance.csv")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--make_pdf", action="store_true")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    X, y = preprocess_data(df)
    # Train/val split
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    results = []
    configs = [
        {"hidden": [64, 32], "lr": 0.01, "batch": 32},
        {"hidden": [128, 64], "lr": 0.01, "batch": 64},
        {"hidden": [256, 128, 64], "lr": 0.005, "batch": 64},
    ]
    for cfg in configs:
        print("\n==== Training config:", cfg, "====")
        mlp = MLP([X_train.shape[1]] + cfg["hidden"] + [y_train.shape[1]], learning_rate=cfg["lr"])
        start = time.time()
        history = mlp.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=cfg["batch"], early_stop_patience=10)
        elapsed = time.time() - start
        last_epoch, train_loss, train_acc, val_loss, val_acc = history[-1]
        results.append({
            "config": f"{cfg['hidden']}, lr={cfg['lr']}, batch={cfg['batch']}",
            "time": elapsed,
            "epochs": last_epoch,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "results_comparison.csv"), index=False)
    if args.make_pdf:
        export_report(results, os.path.join(args.out_dir, "Bao_cao_MLP_StudentPerformance_auto.pdf"))

if __name__ == "__main__":
    main()
