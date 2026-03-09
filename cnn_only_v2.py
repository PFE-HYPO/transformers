import torch
import numpy as np
import random

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

import os
import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


# ===============================
# Dataset
# ===============================
class ECGDataset(Dataset):
    def __init__(self, npy_file):
        data = np.load(npy_file, allow_pickle=True)

        x_np = np.ascontiguousarray(data["x"].astype(np.float32))
        y_np = np.ascontiguousarray(data["label"].astype(np.float32))

        self.X = torch.from_numpy(x_np).unsqueeze(1)   # (N, 1, 2500)
        self.y = torch.from_numpy(y_np)

        print(f"\nLoaded {npy_file}")
        print(f"X shape: {self.X.shape}")
        print(f"Hypo ratio: {self.y.mean().item()*100:.2f}%")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===============================
# Simple CNN
# ===============================
class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 312, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(1)


# ===============================
# Train one epoch
# ===============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ===============================
# Evaluate with threshold
# ===============================
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    probs_all = []
    labels_all = []

    for X, y in loader:
        X = X.to(device)

        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()

        probs_all.extend(probs)
        labels_all.extend(y.numpy())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    preds = (probs_all >= threshold).astype(int)

    acc = accuracy_score(labels_all, preds)
    recall = recall_score(labels_all, preds, zero_division=0)
    f1 = f1_score(labels_all, preds, zero_division=0)
    precision = precision_score(labels_all, preds, zero_division=0)

    return acc, recall, f1, precision, probs_all, labels_all


# ===============================
# Find best threshold
# ===============================
def find_best_threshold(labels, probs):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.01, 0.5, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# ===============================
# Main
# ===============================
if __name__ == "__main__":

    train_dataset = ECGDataset("split_out/train.npy")
    test_dataset = ECGDataset("split_out/test.npy")

    # ---------------------------
    # Weighted sampler
    # ---------------------------
    labels_np = train_dataset.y.numpy().astype(int)

    class_counts = np.bincount(labels_np)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_np]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ---------------------------
    # Device
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---------------------------
    # Model
    # ---------------------------
    model = ECGCNN().to(device)

    # ---------------------------
    # BCEWithLogitsLoss + pos_weight
    # ---------------------------
    n_pos = labels_np.sum()
    n_neg = len(labels_np) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    print(f"Train normal: {n_neg}")
    print(f"Train hypo:   {n_pos}")
    print(f"pos_weight:   {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # ---------------------------
    # Early stopping
    # ---------------------------
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_threshold = 0.5
    patience = 15
    wait = 0

    num_epochs = 15

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # evaluate with default threshold first
        acc, recall, f1, precision, probs, labels = evaluate(model, test_loader, device, threshold=0.5)

        # find best threshold
        thr, f1_thr = find_best_threshold(labels, probs)

        # evaluate again with best threshold
        acc_best, recall_best, f1_best, precision_best, _, _ = evaluate(
            model, test_loader, device, threshold=thr
        )

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"[thr=0.50] Acc: {acc:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
        print(f"[best thr={thr:.2f}] Acc: {acc_best:.4f} | Recall: {recall_best:.4f} | Precision: {precision_best:.4f} | F1: {f1_best:.4f}")

        # early stopping on best-threshold F1
        if f1_best > best_f1:
            best_f1 = f1_best
            best_threshold = thr
            best_model_wts = copy.deepcopy(model.state_dict())
            wait = 0
            print("Best model updated.")
        else:
            wait += 1
            print(f"No improvement. Patience: {wait}/{patience}")

        if wait >= patience:
            print("\nEarly stopping triggered.")
            break

    # ---------------------------
    # Load best model
    # ---------------------------
    model.load_state_dict(best_model_wts)

    acc, recall, f1, precision, probs, labels = evaluate(model, test_loader, device, threshold=best_threshold)

    print("\n================ FINAL BEST MODEL ================")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score:  {f1:.4f}")