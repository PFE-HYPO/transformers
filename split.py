# ===========================
# GLOBAL VARIABLES
# ===========================

X_PATH = "x.npy"
META_PATH = "meta.csv"
OUTPUT_DIR = "split_out"
TEST_RATIO = 0.15
SEED = 0
TRAIN_MAX_NORMAL = 50_000
TEST_MAX_NORMAL = 7_000
NORMAL_LABEL = 0
HYPO_LABEL = 1

# ===========================
# IMPORTS
# ===========================

import os
import numpy as np
import pandas as pd

# ===========================
# HELPERS
# ===========================


def load_meta(meta_path):
    if meta_path.lower().endswith(".csv"):
        df = pd.read_csv(meta_path, low_memory=False)
        if "label" not in df.columns or "patient_id" not in df.columns:
            raise ValueError("CSV must contain 'label' and 'patient_id' columns.")
        labels = df["label"].to_numpy(dtype=np.int64)
        patient_ids = df["patient_id"].astype(str).to_numpy()
        return labels, patient_ids

    raw = np.load(meta_path, allow_pickle=True)

    # Structured npy with named fields
    if raw.dtype.names is not None:
        if "label" not in raw.dtype.names or "patient_id" not in raw.dtype.names:
            raise ValueError("Structured META npy must contain 'label' and 'patient_id' fields.")
        labels = raw["label"].astype(np.int64)
        patient_ids = raw["patient_id"].astype(str)
        return labels, patient_ids

    # Plain 2D npy with first two columns [label, patient_id]
    if raw.ndim == 1:
        raw = raw.reshape(-1, 2)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("META npy must have at least 2 columns: [label, patient_id].")

    labels = raw[:, 0].astype(np.int64)
    patient_ids = raw[:, 1].astype(str)
    return labels, patient_ids


def split_by_patient(patient_ids, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    rng.shuffle(unique_patients)

    target_test = int(len(patient_ids) * test_ratio)
    test_patients = []
    count = 0

    for p in unique_patients:
        n = np.sum(patient_ids == p)
        if count < target_test:
            test_patients.append(p)
            count += n
        else:
            break

    test_patients = np.asarray(test_patients)
    train_patients = unique_patients[~np.isin(unique_patients, test_patients)]
    return train_patients, test_patients


def apply_normal_cap(indices, labels, max_normal, normal_label=0, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices)
    labels_subset = labels[idx]

    normal_idx = idx[labels_subset == normal_label]
    non_normal_idx = idx[labels_subset != normal_label]

    if len(normal_idx) > max_normal:
        normal_idx = rng.choice(normal_idx, size=max_normal, replace=False)

    balanced_idx = np.concatenate([normal_idx, non_normal_idx], axis=0)
    rng.shuffle(balanced_idx)
    return balanced_idx


def summarize_split(name, indices, labels, patient_ids):
    idx = np.asarray(indices)
    n_total = len(idx)
    n_normal = int(np.sum(labels[idx] == NORMAL_LABEL))
    n_hypo = int(np.sum(labels[idx] == HYPO_LABEL))
    n_patients = int(np.unique(patient_ids[idx]).shape[0])
    ratio = (100.0 * n_hypo / max(n_total, 1))
    print(f"\n{name}:")
    print(f"- segments: {n_total}")
    print(f"- normal (label=0): {n_normal}")
    print(f"- hypo (label=1): {n_hypo}")
    print(f"- hypo ratio: {ratio:.4f}%")
    print(f"- unique patients: {n_patients}")


def build_structured_dataset(x_rows, labels_rows, patient_rows):
    max_pid_len = max(1, int(max(len(str(p)) for p in patient_rows)))
    dtype = np.dtype(
        [
            ("x", np.float32, (x_rows.shape[1],)),
            ("label", np.int8),
            ("patient_id", f"U{max_pid_len}"),
        ]
    )

    out = np.empty(len(labels_rows), dtype=dtype)
    out["x"] = np.asarray(x_rows, dtype=np.float32)
    out["label"] = np.asarray(labels_rows, dtype=np.int8)
    out["patient_id"] = np.asarray(patient_rows, dtype=f"U{max_pid_len}")
    return out


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    X = np.load(X_PATH, mmap_mode="r")
    labels, patient_ids = load_meta(META_PATH)

    if X.shape[0] != len(labels):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows but META has {len(labels)} rows.")

    train_patients, test_patients = split_by_patient(patient_ids, test_ratio=TEST_RATIO, seed=SEED)

    all_indices = np.arange(len(labels))
    train_idx_all = all_indices[np.isin(patient_ids, train_patients)]
    test_idx_all = all_indices[np.isin(patient_ids, test_patients)]

    # Keep all hypo (label=1), cap normal (label=0)
    train_idx = apply_normal_cap(
        train_idx_all,
        labels,
        max_normal=TRAIN_MAX_NORMAL,
        normal_label=NORMAL_LABEL,
        seed=SEED + 1,
    )
    test_idx = apply_normal_cap(
        test_idx_all,
        labels,
        max_normal=TEST_MAX_NORMAL,
        normal_label=NORMAL_LABEL,
        seed=SEED + 2,
    )

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    train_array = build_structured_dataset(X[train_idx], labels[train_idx], patient_ids[train_idx])
    test_array = build_structured_dataset(X[test_idx], labels[test_idx], patient_ids[test_idx])

    train_path = os.path.join(OUTPUT_DIR, "train.npy")
    test_path = os.path.join(OUTPUT_DIR, "test.npy")

    np.save(train_path, train_array)
    np.save(test_path, test_array)

    print("=== SPLIT DONE ===")
    print(f"Train shape: {train_array.shape}")
    print(f"Test shape: {test_array.shape}")
    print(f"Train dtype: {train_array.dtype}")
    print(f"Test dtype: {test_array.dtype}")
    summarize_split("TRAIN", train_idx, labels, patient_ids)
    summarize_split("TEST", test_idx, labels, patient_ids)
    print("\nSaved files:")
    print(f"- {train_path}")
    print(f"- {test_path}")
