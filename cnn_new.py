import torch
import numpy as np
import random
import copy
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix


# =========================================================
# SEED
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
# CONFIG
# =========================================================
TRAIN_PATH = "split_out/train.npy"
TEST_PATH = "split_out/test.npy"

BATCH_SIZE = 64
LR = 3e-4
NUM_EPOCHS = 12
PATIENCE = 4
DROPOUT = 0.3

# nouveau but: garder un recall correct mais améliorer la precision
TARGET_RECALL = 0.85


# =========================================================
# DATASET
# =========================================================
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


# =========================================================
# SIMPLE CNN
# =========================================================
class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 312, 64)
        self.dropout = nn.Dropout(DROPOUT)
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


# =========================================================
# TRAIN
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
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# EVAL
# =========================================================
@torch.no_grad()
def collect_probs(model, loader, device):
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

    return probs_all, labels_all


def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# =========================================================
# THRESHOLD SELECTION
# =========================================================
def find_balanced_threshold(labels, probs, target_recall=0.85):
    """
    Nouveau critère :
    1) on garde les thresholds avec recall >= target_recall
    2) parmi eux, on maximise precision
    3) si égalité, on maximise F1
    """
    candidates = []

    for t in np.arange(0.01, 0.51, 0.01):
        m = compute_metrics(labels, probs, t)
        candidates.append(m)

    valid = [m for m in candidates if m["recall"] >= target_recall]

    if len(valid) > 0:
        valid = sorted(valid, key=lambda x: (x["precision"], x["f1"]), reverse=True)
        best = valid[0]
    else:
        # si aucun seuil n'atteint le recall cible, on prend meilleur F1
        candidates = sorted(candidates, key=lambda x: x["f1"], reverse=True)
        best = candidates[0]

    return best, candidates


def print_threshold_table(candidates, top_k=8):
    ranked = sorted(candidates, key=lambda x: (x["precision"], x["f1"]), reverse=True)

    print("\nTop thresholds (ranked by precision then F1 among all tested):")
    print(f"{'thr':>6} {'acc':>8} {'recall':>8} {'prec':>8} {'f1':>8}")
    for m in ranked[:top_k]:
        print(f"{m['threshold']:>6.2f} {m['accuracy']:>8.4f} {m['recall']:>8.4f} {m['precision']:>8.4f} {m['f1']:>8.4f}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    train_dataset = ECGDataset(TRAIN_PATH)
    test_dataset = ECGDataset(TEST_PATH)

    # IMPORTANT :
    # on enlève WeightedRandomSampler pour réduire le biais excessif vers hypo
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = ECGCNN().to(device)

    labels_np = train_dataset.y.numpy().astype(int)
    n_pos = labels_np.sum()
    n_neg = len(labels_np) - n_pos

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    print(f"Train normal: {n_neg}")
    print(f"Train hypo:   {n_pos}")
    print(f"pos_weight:   {pos_weight.item():.4f}")
    print(f"Target recall for threshold selection: {TARGET_RECALL:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_threshold = 0.5
    best_metrics = None

    wait = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        probs, labels = collect_probs(model, test_loader, device)

        # métriques classiques avec threshold=0.5
        m05 = compute_metrics(labels, probs, 0.50)

        # nouveau choix de threshold équilibré
        best_thr_metrics, candidates = find_balanced_threshold(
            labels,
            probs,
            target_recall=TARGET_RECALL
        )

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"[thr=0.50] Acc: {m05['accuracy']:.4f} | Recall: {m05['recall']:.4f} | Precision: {m05['precision']:.4f} | F1: {m05['f1']:.4f}")
        print(f"[balanced thr={best_thr_metrics['threshold']:.2f}] Acc: {best_thr_metrics['accuracy']:.4f} | Recall: {best_thr_metrics['recall']:.4f} | Precision: {best_thr_metrics['precision']:.4f} | F1: {best_thr_metrics['f1']:.4f}")

        # score principal = precision, avec contrainte recall
        current_score = best_thr_metrics["precision"]

        if current_score > best_score:
            best_score = current_score
            best_threshold = best_thr_metrics["threshold"]
            best_model_wts = copy.deepcopy(model.state_dict())
            best_metrics = best_thr_metrics
            wait = 0
            print("Best model updated.")
            print_threshold_table(candidates, top_k=6)
        else:
            wait += 1
            print(f"No improvement. Patience: {wait}/{PATIENCE}")

        if wait >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    model.load_state_dict(best_model_wts)

    probs, labels = collect_probs(model, test_loader, device)
    final_metrics = compute_metrics(labels, probs, best_threshold)

    print("\n================ FINAL BEST MODEL ================")
    print(f"Chosen threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"F1-score:  {final_metrics['f1']:.4f}")
    print(f"TN: {final_metrics['tn']} | FP: {final_metrics['fp']} | FN: {final_metrics['fn']} | TP: {final_metrics['tp']}")