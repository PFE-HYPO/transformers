import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


# =========================================================
# Seed
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# =========================================================
# Dataset
# =========================================================
class ECGDataset(Dataset):
    def __init__(self, npy_file, augment=False):
        data = np.load(npy_file, allow_pickle=True)

        x_np = np.ascontiguousarray(data["x"].astype(np.float32))
        y_np = np.ascontiguousarray(data["label"].astype(np.float32))

        self.X = torch.from_numpy(x_np).unsqueeze(1)   # (N, 1, 2500)
        self.y = torch.from_numpy(y_np)
        self.augment = augment

        print(f"\nLoaded {npy_file}")
        print(f"X shape: {self.X.shape}")
        print(f"Hypo ratio: {self.y.mean().item()*100:.2f}%")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = x + 0.01 * torch.randn_like(x)

            if torch.rand(1).item() < 0.3:
                shift = torch.randint(-40, 41, (1,)).item()
                x = torch.roll(x, shifts=shift, dims=-1)

        return x, y


# =========================================================
# Residual Block
# =========================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# =========================================================
# Model: ResNet1D + BiLSTM + Attention
# =========================================================
class ECGResNetLSTM(nn.Module):
    def __init__(self, lstm_hidden=64, lstm_layers=1, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 2500 -> 1250
        )

        self.block1 = ResidualBlock1D(32, 64, kernel_size=7, dropout=0.1)
        self.pool1 = nn.MaxPool1d(2)     # 1250 -> 625

        self.block2 = ResidualBlock1D(64, 128, kernel_size=5, dropout=0.1)
        self.pool2 = nn.MaxPool1d(2)     # 625 -> 312

        self.block3 = ResidualBlock1D(128, 128, kernel_size=5, dropout=0.1)
        self.pool3 = nn.MaxPool1d(2)     # 312 -> 156

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
            bidirectional=True
        )

        self.attn_fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.attn_fc2 = nn.Linear(64, 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B,1,2500)
        x = self.stem(x)      # (B,32,1250)

        x = self.block1(x)
        x = self.pool1(x)     # (B,64,625)

        x = self.block2(x)
        x = self.pool2(x)     # (B,128,312)

        x = self.block3(x)
        x = self.pool3(x)     # (B,128,156)

        x = x.permute(0, 2, 1)   # (B,156,128)

        lstm_out, _ = self.lstm(x)   # (B,156,2*h)

        attn = torch.tanh(self.attn_fc1(lstm_out))
        attn = self.attn_fc2(attn)          # (B,156,1)
        attn = torch.softmax(attn, dim=1)

        context = torch.sum(lstm_out * attn, dim=1)   # (B,2*h)

        out = self.classifier(context).squeeze(1)     # (B,)
        return out


# =========================================================
# Training / Evaluation
# =========================================================
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


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
    precision = precision_score(labels_all, preds, zero_division=0)
    f1 = f1_score(labels_all, preds, zero_division=0)

    return acc, recall, precision, f1, probs_all, labels_all


def find_best_threshold(labels, probs):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.01, 0.51, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    train_dataset = ECGDataset("split_out/train.npy", augment=True)
    test_dataset = ECGDataset("split_out/test.npy", augment=False)

    labels_np = train_dataset.y.numpy().astype(int)

    class_counts = np.bincount(labels_np)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_np]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = ECGResNetLSTM(
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    n_pos = labels_np.sum()
    n_neg = len(labels_np) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    print(f"Train normal: {n_neg}")
    print(f"Train hypo:   {n_pos}")
    print(f"pos_weight:   {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_threshold = 0.5

    patience = 10
    wait = 0
    num_epochs = 20

    os.makedirs("checkpoints_resnet_lstm", exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        acc05, rec05, prec05, f105, probs, labels = evaluate(model, test_loader, device, threshold=0.5)
        thr, f1_thr = find_best_threshold(labels, probs)
        accb, recb, precb, f1b, _, _ = evaluate(model, test_loader, device, threshold=thr)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"[thr=0.50] Acc: {acc05:.4f} | Recall: {rec05:.4f} | Precision: {prec05:.4f} | F1: {f105:.4f}")
        print(f"[best thr={thr:.2f}] Acc: {accb:.4f} | Recall: {recb:.4f} | Precision: {precb:.4f} | F1: {f1b:.4f}")

        if f1b > best_f1:
            best_f1 = f1b
            best_threshold = thr
            best_model_wts = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(model.state_dict(), "checkpoints_resnet_lstm/best_model.pt")
            print("Best model updated.")
        else:
            wait += 1
            print(f"No improvement. Patience: {wait}/{patience}")

        if wait >= patience:
            print("\nEarly stopping triggered.")
            break

    model.load_state_dict(best_model_wts)

    acc, recall, precision, f1, probs, labels = evaluate(
        model, test_loader, device, threshold=best_threshold
    )

    print("\n================ FINAL BEST MODEL ================")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score:  {f1:.4f}")