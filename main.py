import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from pathlib import Path
from dataset import ECGStructuredNPYDataset
from models import ECG_CNN1D
import torch.nn as nn

from training import evaluate, train_one_epoch


def main():
    SPLIT_DIR = os.environ.get("SPLIT_DIR", "split_out")
    train_path = os.path.join(SPLIT_DIR, "train.npy")
    test_path  = os.path.join(SPLIT_DIR, "test.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} | SPLIT_DIR={SPLIT_DIR}")

    train_ds = ECGStructuredNPYDataset(train_path)
    test_ds  = ECGStructuredNPYDataset(test_path)

    # WeightedRandomSampler to balance classes in TRAIN only
    y_train = train_ds.y
    class_counts = np.bincount(y_train, minlength=2)  # [count0, count1]
    # weight per sample = 1 / count(class)
    weights = np.where(y_train == 0, 1.0 / max(class_counts[0], 1), 1.0 / max(class_counts[1], 1)).astype(np.float64)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = ECG_CNN1D(in_ch=1, base_ch=32, dropout=0.2).to(device)

    # BCEWithLogitsLoss with pos_weight (optional bonus)
    # pos_weight > 1 increases penalty for missing positives (hypo=1)
    pos = float(class_counts[1])
    neg = float(class_counts[0])
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = -1.0
    patience = 15
    patience_left = patience
    thresh = 0.5

    for epoch in range(1, 101):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        global_m, per_patient = evaluate(model, test_loader, device, thresh=thresh)

        print(f"Epoch {epoch:3d} | TrainLoss {tr_loss:.4f} | "
              f"Acc {global_m['accuracy']*100:.2f}% | "
              f"Recall {global_m['recall']:.4f} | "
              f"Prec {global_m['precision']:.4f} | "
              f"Spec {global_m['specificity']:.4f} | "
              f"F1 {global_m['f1']:.4f}")

        # Print per-patient summary (compact)
        for pid, m in per_patient.items():
            print(f"  - {pid}: Recall {m['recall']:.4f} | F1 {m['f1']:.4f} | Acc {m['accuracy']*100:.2f}%")

        # Early stopping on F1 (more meaningful than accuracy with imbalance)
        if global_m["f1"] > best_f1 + 1e-4:
            best_f1 = global_m["f1"]
            patience_left = patience
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_f1": best_f1}, "best_cnn.pt")
            print(f"[INFO] Saved best checkpoint (best_f1={best_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[INFO] Early stopping.")
                break

    print(f"[DONE] Best F1 = {best_f1:.4f} | checkpoint: best_cnn.pt")

if __name__ == "__main__":
    main()