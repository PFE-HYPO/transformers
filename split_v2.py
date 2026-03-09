# ===========================
# GLOBAL VARIABLES
# ===========================

X_PATH = "x.npy"
META_PATH = "meta.csv"
OUTPUT_DIR = "split_out"

TEST_RATIO = 0.15
SEED = 42

NORMAL_LABEL = 0
HYPO_LABEL = 1

# Target class ratios after capping normals
TRAIN_TARGET_HYPO_RATIO = 0.40   # e.g. 40% hypo, 60% normal
TEST_TARGET_HYPO_RATIO = 0.50    # e.g. balanced test

# Search bounds for number of test patients
MIN_TEST_PATIENTS = 2
MAX_TEST_PATIENTS = 3

# ===========================
# IMPORTS
# ===========================

import os
import itertools
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

    if raw.dtype.names is not None:
        if "label" not in raw.dtype.names or "patient_id" not in raw.dtype.names:
            raise ValueError("Structured META npy must contain 'label' and 'patient_id' fields.")
        labels = raw["label"].astype(np.int64)
        patient_ids = raw["patient_id"].astype(str)
        return labels, patient_ids

    if raw.ndim == 1:
        raw = raw.reshape(-1, 2)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("META npy must have at least 2 columns: [label, patient_id].")

    labels = raw[:, 0].astype(np.int64)
    patient_ids = raw[:, 1].astype(str)
    return labels, patient_ids


def summarize_split(name, indices, labels, patient_ids):
    idx = np.asarray(indices)
    n_total = len(idx)
    n_normal = int(np.sum(labels[idx] == NORMAL_LABEL))
    n_hypo = int(np.sum(labels[idx] == HYPO_LABEL))
    n_patients = int(np.unique(patient_ids[idx]).shape[0])
    ratio = 100.0 * n_hypo / max(n_total, 1)

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


def patient_stats(labels, patient_ids):
    stats = {}
    for pid in np.unique(patient_ids):
        mask = (patient_ids == pid)
        yy = labels[mask]
        n0 = int(np.sum(yy == NORMAL_LABEL))
        n1 = int(np.sum(yy == HYPO_LABEL))
        total = n0 + n1
        hypo_ratio = n1 / total if total > 0 else 0.0
        stats[pid] = {
            "normal": n0,
            "hypo": n1,
            "total": total,
            "hypo_ratio": hypo_ratio,
        }
    return stats


def choose_best_test_patients(patient_ids, labels,
                              test_ratio=0.15,
                              min_test_patients=2,
                              max_test_patients=3,
                              target_test_hypo_ratio=0.50):
    """
    Brute-force search over patient combinations.
    Objective:
    - test size close to desired TEST_RATIO
    - test hypo ratio close to target_test_hypo_ratio
    """
    total_segments = len(labels)
    unique_patients = np.unique(patient_ids)
    stats = patient_stats(labels, patient_ids)

    best_score = float("inf")
    best_test_patients = None

    for k in range(min_test_patients, min(max_test_patients, len(unique_patients)) + 1):
        for combo in itertools.combinations(unique_patients, k):
            n_total = sum(stats[p]["total"] for p in combo)
            n_hypo = sum(stats[p]["hypo"] for p in combo)

            size_ratio = n_total / total_segments
            hypo_ratio = n_hypo / max(n_total, 1)

            # weighted objective
            score = (
                3.0 * abs(size_ratio - test_ratio) +
                2.0 * abs(hypo_ratio - target_test_hypo_ratio)
            )

            if score < best_score:
                best_score = score
                best_test_patients = np.array(combo)

    train_patients = np.array([p for p in unique_patients if p not in set(best_test_patients)])
    return train_patients, best_test_patients


def cap_normals_to_target_ratio(indices, labels, target_hypo_ratio, seed=0):
    """
    Keep all hypo.
    Select only enough normals to reach target_hypo_ratio.
    """
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices)

    normal_idx = idx[labels[idx] == NORMAL_LABEL]
    hypo_idx = idx[labels[idx] == HYPO_LABEL]

    H = len(hypo_idx)

    # desired total T = H / r  => desired normals N = T - H = H*(1-r)/r
    desired_normals = int(round(H * (1.0 - target_hypo_ratio) / target_hypo_ratio))

    if desired_normals < len(normal_idx):
        normal_idx = rng.choice(normal_idx, size=desired_normals, replace=False)

    out = np.concatenate([normal_idx, hypo_idx], axis=0)
    rng.shuffle(out)
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

    train_patients, test_patients = choose_best_test_patients(
        patient_ids=patient_ids,
        labels=labels,
        test_ratio=TEST_RATIO,
        min_test_patients=MIN_TEST_PATIENTS,
        max_test_patients=MAX_TEST_PATIENTS,
        target_test_hypo_ratio=TEST_TARGET_HYPO_RATIO,
    )

    print("Chosen train patients:", sorted(train_patients.tolist()))
    print("Chosen test patients :", sorted(test_patients.tolist()))

    all_indices = np.arange(len(labels))
    train_idx_all = all_indices[np.isin(patient_ids, train_patients)]
    test_idx_all = all_indices[np.isin(patient_ids, test_patients)]

    # Cap normals to target ratios
    train_idx = cap_normals_to_target_ratio(
        train_idx_all,
        labels,
        target_hypo_ratio=TRAIN_TARGET_HYPO_RATIO,
        seed=SEED + 1,
    )

    test_idx = cap_normals_to_target_ratio(
        test_idx_all,
        labels,
        target_hypo_ratio=TEST_TARGET_HYPO_RATIO,
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

    print("\n=== SPLIT DONE ===")
    print(f"Train shape: {train_array.shape}")
    print(f"Test shape: {test_array.shape}")
    print(f"Train dtype: {train_array.dtype}")
    print(f"Test dtype: {test_array.dtype}")

    summarize_split("TRAIN", train_idx, labels, patient_ids)
    summarize_split("TEST", test_idx, labels, patient_ids)

    print("\nSaved files:")
    print(f"- {train_path}")
    print(f"- {test_path}")