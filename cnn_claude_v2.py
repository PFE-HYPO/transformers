import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import os


# ===============================
# Dataset avec augmentation
# ===============================
class ECGDataset(Dataset):
    def __init__(self, npy_file, augment=False):

        data = np.load(npy_file, allow_pickle=True)

        # conversion propre pour PyTorch
        x_np = np.ascontiguousarray(data["x"].astype(np.float32))
        y_np = np.ascontiguousarray(data["label"].astype(np.float32))

        self.X = torch.from_numpy(x_np)
        self.y = torch.from_numpy(y_np)

        self.augment = augment

        # reshape -> (N,1,2500)
        if self.X.dim() == 2:
            self.X = self.X.unsqueeze(1)

        print(f"\nLoaded {npy_file}")
        print(f"Signals shape: {self.X.shape}")
        print(f"Hypo ratio: {(self.y.mean().item()*100):.2f}%")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # bruit gaussien
            if torch.rand(1) < 0.5:
                x = x + torch.randn_like(x) * 0.02

            # scaling aléatoire ±10%
            if torch.rand(1) < 0.4:
                x = x * (0.9 + torch.rand(1) * 0.2)

            # time shift
            if torch.rand(1) < 0.3:
                shift = torch.randint(-100, 100, (1,)).item()
                x = torch.roll(x, shift, dims=-1)

        return x, y


# ===============================
# Residual Block 1D avec dilatation
# ===============================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv_block(x) + self.shortcut(x))


# ===============================
# Modèle CNN (ResNet1D) + BiLSTM + Attention
# ===============================

class ECGResNetLSTM(nn.Module):
    def __init__(self, lstm_hidden=64, lstm_layers=1, dropout=0.4):
        super().__init__()

        # --- Bloc d'entrée large pour capturer ondes P, QRS, T ---
        self.input_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=49, padding=24, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),  # 2500 -> 625
        )

        # --- Blocs Residuels avec dilatations croissantes ---
        self.stage1 = nn.Sequential(
            ResidualBlock1D(32,  64,  kernel_size=7, dilation=1),
            ResidualBlock1D(64,  64,  kernel_size=7, dilation=2),
            nn.MaxPool1d(2),   # 625 -> 312
        )

        self.stage2 = nn.Sequential(
            ResidualBlock1D(64,  128, kernel_size=5, dilation=1),
            ResidualBlock1D(128, 128, kernel_size=5, dilation=2),
            nn.MaxPool1d(2),   # 312 -> 156
        )

        self.stage3 = nn.Sequential(
            ResidualBlock1D(128, 256, kernel_size=3, dilation=1),
            ResidualBlock1D(256, 256, kernel_size=3, dilation=2),
            nn.MaxPool1d(2),   # 156 -> 78
        )

        # --- BiLSTM léger ---
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # --- Attention ---
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # --- Classifieur final ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, 1, 2500)

        x = self.input_block(x)   # (batch, 32,  625)
        x = self.stage1(x)        # (batch, 64,  312)
        x = self.stage2(x)        # (batch, 128, 156)
        x = self.stage3(x)        # (batch, 256,  78)

        # Préparer pour LSTM
        x = x.permute(0, 2, 1)           # (batch, 78, 256)
        lstm_out, _ = self.lstm(x)        # (batch, 78, lstm_hidden*2)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, 78, 1)
        context = (lstm_out * attn_weights).sum(dim=1)                 # (batch, lstm_hidden*2)

        return self.classifier(context)   # (batch, 1)


# ===============================
# Entraînement
# ===============================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, probs_all, labels = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            prob = torch.sigmoid(model(X).squeeze()).cpu().numpy()

            pred = (prob > 0.5).astype(int)
            preds.extend(pred)
            probs_all.extend(prob)
            labels.extend(y.numpy())

    acc    = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, zero_division=0)
    f1     = f1_score(labels, preds, zero_division=0)
    auc    = roc_auc_score(labels, probs_all) if len(set(labels)) > 1 else 0.0

    return acc, recall, f1, auc


# ===============================
# Main
# ===============================

if __name__ == "__main__":

    # --- Datasets ---
    train_dataset = ECGDataset("split_out/train.npy", augment=True)
    test_dataset  = ECGDataset("split_out/test.npy",  augment=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # --- Gestion du déséquilibre de classes ---
    labels_array = train_dataset.y.numpy()
    n_pos = labels_array.sum()
    n_neg = len(labels_array) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6) * 2.0 ], dtype=torch.float32)
    print(f"Classe 0: {int(n_neg)} | Classe 1: {int(n_pos)} | pos_weight: {pos_weight.item():.2f}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Modèle ---
    model = ECGResNetLSTM(lstm_hidden=64, lstm_layers=1, dropout=0.4).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables: {total_params:,}")

    # --- Loss ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # --- Optimizer + Cosine Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )

    # --- Boucle d'entraînement ---
    best_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    NUM_EPOCHS = 50

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, recall, f1, auc = evaluate(model, test_loader, device)

        scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            saved = "✅ saved"
        else:
            saved = ""

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Loss: {train_loss:.4f} | "
              f"Acc: {acc:.4f} | "
              f"Recall: {recall:.4f} | "
              f"F1: {f1:.4f} | "
              f"AUC: {auc:.4f} | "
              f"LR: {lr:.6f} {saved}")

    print(f"\n🏆 Meilleur F1: {best_f1:.4f} — modèle sauvegardé dans checkpoints/best_model.pt")

    # --- Résultats finaux ---
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    acc, recall, f1, auc = evaluate(model, test_loader, device)
    print(f"\n📊 Résultats finaux (meilleur modèle):")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")