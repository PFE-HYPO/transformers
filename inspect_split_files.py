from pathlib import Path

import pandas as pd
import numpy as np


# =========================
# Global configuration only
# =========================
TRAIN_NPY_PATH = Path("split_out/train.npy")
TEST_NPY_PATH = Path("split_out/test.npy")
NORMAL_LABEL = 0
HYPO_LABEL = 1
SORT_BY_TOTAL_DESC = True


def load_structured_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.load(path, allow_pickle=False)

    if arr.dtype.names is None:
        raise ValueError(
            f"{path} is not a structured npy. Expected fields: 'label' and 'patient_id'."
        )
    if "label" not in arr.dtype.names or "patient_id" not in arr.dtype.names:
        raise ValueError(
            f"{path} must contain fields 'label' and 'patient_id'. Found: {arr.dtype.names}"
        )

    df = pd.DataFrame(
        {
            "label": arr["label"].astype(int),
            "patient_id": arr["patient_id"].astype(str),
        }
    )
    return df


def summarize_counts(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    total = len(df)
    n_normal = int((df["label"] == NORMAL_LABEL).sum())
    n_hypo = int((df["label"] == HYPO_LABEL).sum())
    hypo_pct = 100.0 * n_hypo / max(total, 1)
    n_patients = df["patient_id"].nunique()

    print(f"\n=== {split_name} SUMMARY ===")
    print(f"Total segments: {total}")
    print(f"Normal (label={NORMAL_LABEL}): {n_normal}")
    print(f"Hypo (label={HYPO_LABEL}): {n_hypo}")
    print(f"Hypo percent: {hypo_pct:.4f}%")
    print(f"Unique patients: {n_patients}")

    grouped = pd.crosstab(df["patient_id"], df["label"], dropna=False)
    if NORMAL_LABEL not in grouped.columns:
        grouped[NORMAL_LABEL] = 0
    if HYPO_LABEL not in grouped.columns:
        grouped[HYPO_LABEL] = 0

    out = pd.DataFrame(
        {
            "patient_id": grouped.index.astype(str),
            "normal_segments": grouped[NORMAL_LABEL].to_numpy(dtype=int),
            "hypo_segments": grouped[HYPO_LABEL].to_numpy(dtype=int),
        }
    )
    out["total_segments"] = out["normal_segments"] + out["hypo_segments"]
    out["hypo_percent"] = 100.0 * out["hypo_segments"] / out["total_segments"].clip(lower=1)

    if SORT_BY_TOTAL_DESC:
        out = out.sort_values("total_segments", ascending=False)
    else:
        out = out.sort_values("patient_id", ascending=True)

    out = out.reset_index(drop=True)

    print("\nGrouped by patient_id:")
    print(out.to_string(index=False))
    return out


def main() -> None:
    train_df = load_structured_split(TRAIN_NPY_PATH)
    test_df = load_structured_split(TEST_NPY_PATH)

    train_grouped = summarize_counts(train_df, "TRAIN")
    test_grouped = summarize_counts(test_df, "TEST")

    print("\n=== GLOBAL (TRAIN + TEST) ===")
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    total = len(all_df)
    n_normal = int((all_df["label"] == NORMAL_LABEL).sum())
    n_hypo = int((all_df["label"] == HYPO_LABEL).sum())
    print(f"Total segments: {total}")
    print(f"Normal (label={NORMAL_LABEL}): {n_normal}")
    print(f"Hypo (label={HYPO_LABEL}): {n_hypo}")
    print(f"Hypo percent: {100.0 * n_hypo / max(total, 1):.4f}%")

    overlap = set(train_grouped["patient_id"]).intersection(set(test_grouped["patient_id"]))
    print(f"Patient overlap between train/test: {len(overlap)}")
    if len(overlap) > 0:
        print(f"Overlapping IDs: {sorted(overlap)}")


if __name__ == "__main__":
    main()
