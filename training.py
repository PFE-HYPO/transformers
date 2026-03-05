import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import  confusion_from_probs, metrics_from_conf
import numpy as np
import pandas as pd


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, thresh=0.5):
    model.eval()
    all_probs = []
    all_y = []
    all_pid = []

    for x, y, pid in loader:
        x = x.to(device, non_blocking=True)
        y = y.cpu().numpy().astype(np.int32)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_y.append(y)
        all_pid.extend(list(pid))

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    # Global metrics
    tp, tn, fp, fn = confusion_from_probs(y_true, probs, thresh=thresh)
    global_m = metrics_from_conf(tp, tn, fp, fn)

    # Per-patient metrics
    per_patient = {}
    pids = np.array(all_pid)
    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        tp_p, tn_p, fp_p, fn_p = confusion_from_probs(y_true[idx], probs[idx], thresh=thresh)
        per_patient[pid] = metrics_from_conf(tp_p, tn_p, fp_p, fn_p)

    return global_m, per_patient

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0

    for x, y, _pid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)                       # (B,)
        loss = criterion(logits, y)             # BCEWithLogits expects logits
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)
