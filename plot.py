import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_PATH = "x.npy"
META_PATH = "meta.csv"
HYPO_LABEL = 1
NORMAL_LABEL = 0


def main() -> None:
    x = np.load(X_PATH, mmap_mode="r")
    meta = pd.read_csv(META_PATH, usecols=["label", "patient_id"])

    labels = meta["label"].to_numpy(dtype=np.int64)
    patient_ids = meta["patient_id"].astype(str).to_numpy()

    if x.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch rows: x.npy={x.shape[0]} vs meta.csv={labels.shape[0]}")

    hypo_indices = np.flatnonzero(labels == HYPO_LABEL)
    normal_indices = np.flatnonzero(labels == NORMAL_LABEL)

    if hypo_indices.size == 0 or normal_indices.size == 0:
        raise ValueError("Impossible de tirer au hasard: il manque une des classes (0 ou 1).")

    rng = np.random.default_rng()
    hypo_idx = int(rng.choice(hypo_indices))
    normal_idx = int(rng.choice(normal_indices))

    hypo_signal = np.asarray(x[hypo_idx], dtype=np.float32)
    normal_signal = np.asarray(x[normal_idx], dtype=np.float32)

    print(f"Hypo sample -> idx={hypo_idx}, patient_id={patient_ids[hypo_idx]}, label={labels[hypo_idx]}")
    print(f"Normal sample -> idx={normal_idx}, patient_id={patient_ids[normal_idx]}, label={labels[normal_idx]}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(hypo_signal, color="crimson", linewidth=1.2)
    axes[0].set_title(f"Signal Hypo (label=1) - idx={hypo_idx} - patient={patient_ids[hypo_idx]}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(normal_signal, color="royalblue", linewidth=1.2)
    axes[1].set_title(f"Signal Normal (label=0) - idx={normal_idx} - patient={patient_ids[normal_idx]}")
    axes[1].set_xlabel("Time (samples)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Random ECG Samples from x.npy + meta.csv", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
