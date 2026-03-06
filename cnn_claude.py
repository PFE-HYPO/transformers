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

        x = self.X[idx]
        y = self.y[idx]

        if self.augment:

            # bruit gaussien
            if torch.rand(1) < 0.5:
                x = x + torch.randn_like(x) * 0.01

            # time shift
            if torch.rand(1) < 0.3:
                shift = torch.randint(-50, 50, (1,)).item()
                x = torch.roll(x, shift, dims=-1)

        return x, y    

# ===============================
# Residual Block 1D
# ===============================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()

        padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # Shortcut connection (si changement de dimensions)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        out = out + self.shortcut(x)  # skip connection
        return self.relu(out)


# ===============================
# Modèle CNN (ResNet1D) + LSTM
# ===============================

class ECGResNetLSTM(nn.Module):
    def __init__(self, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # --- Bloc d'entrée ---
        self.input_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 2500 -> 1250
        )

        # --- Blocs Residuels ---
        self.res_block1 = ResidualBlock1D(32, 64, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)   # 1250 -> 625

        self.res_block2 = ResidualBlock1D(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)   # 625 -> 312

        self.res_block3 = ResidualBlock1D(128, 128, kernel_size=5)
        self.pool3 = nn.MaxPool1d(2)   # 312 -> 156

        # --- LSTM pour capturer les dépendances temporelles ---
        # Input: (batch, seq_len=156, features=128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,  # BiLSTM pour contexte passé + futur
        )

        # --- Attention sur les sorties LSTM ---
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # --- Classifieur final ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, 1, 2500)

        # CNN feature extraction
        x = self.input_block(x)          # (batch, 32, 1250)

        x = self.res_block1(x)
        x = self.pool1(x)                # (batch, 64, 625)

        x = self.res_block2(x)
        x = self.pool2(x)                # (batch, 128, 312)

        x = self.res_block3(x)
        x = self.pool3(x)                # (batch, 128, 156)

        # Préparer pour LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)          # (batch, 156, 128)

        # LSTM
        lstm_out, _ = self.lstm(x)       # (batch, 156, lstm_hidden*2)

        # Attention pooling
        attn_weights = self.attention(lstm_out)          # (batch, 156, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, lstm_hidden*2)

        # Classification
        out = self.classifier(context)   # (batch, 1)

        return out


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

        # Gradient clipping pour stabilité
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
            outputs = model(X).squeeze()
            prob = torch.sigmoid(outputs).cpu().numpy()

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

    # --- Gestion du déséquilibre de classes ---
    labels_array = train_dataset.y.numpy()
    n_pos = labels_array.sum()
    n_neg = len(labels_array) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], dtype=torch.float32)
    print(f"Classe 0: {int(n_neg)} | Classe 1: {int(n_pos)} | pos_weight: {pos_weight.item():.2f}")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Modèle ---
    model = ECGResNetLSTM(lstm_hidden=128, lstm_layers=2, dropout=0.3).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables: {total_params:,}")

    # --- Loss avec pos_weight pour données déséquilibrées ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    # --- Boucle d'entraînement ---
    best_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    NUM_EPOCHS = 40

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, recall, f1, auc = evaluate(model, test_loader, device)

        # Scheduler basé sur F1
        scheduler.step(f1)

        # Sauvegarde du meilleur modèle
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
              f"AUC: {auc:.4f} {saved}")

    print(f"\n🏆 Meilleur F1: {best_f1:.4f} — modèle sauvegardé dans checkpoints/best_model.pt")

    # --- Chargement du meilleur modèle pour évaluation finale ---
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    acc, recall, f1, auc = evaluate(model, test_loader, device)
    print(f"\n📊 Résultats finaux (meilleur modèle):")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")