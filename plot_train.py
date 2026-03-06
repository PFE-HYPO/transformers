import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = "split_out/train.npy"
HYPO_LABEL = 1
NORMAL_LABEL = 0


def main() -> None:
    data = np.load(TRAIN_PATH, allow_pickle=True)

    expected_fields = {"x", "label", "patient_id"}
    if data.dtype.names is None or not expected_fields.issubset(set(data.dtype.names)):
        raise ValueError("train.npy doit contenir les champs: x, label, patient_id.")

    signals = data["x"].astype(np.float32)
    labels = data["label"].astype(np.int64)
    patient_ids = data["patient_id"].astype(str)

    hypo_indices = np.flatnonzero(labels == HYPO_LABEL)
    normal_indices = np.flatnonzero(labels == NORMAL_LABEL)

    if hypo_indices.size == 0 or normal_indices.size == 0:
        raise ValueError("Impossible de tirer au hasard: une classe (0 ou 1) est absente.")

    rng = np.random.default_rng()
    hypo_idx = int(rng.choice(hypo_indices))
    normal_idx = int(rng.choice(normal_indices))

    hypo_signal = signals[hypo_idx]
    normal_signal = signals[normal_idx]

    print(
        f"Hypo sample -> idx={hypo_idx}, "
        f"patient_id={patient_ids[hypo_idx]}, label={labels[hypo_idx]}"
    )
    print(
        f"Normal sample -> idx={normal_idx}, "
        f"patient_id={patient_ids[normal_idx]}, label={labels[normal_idx]}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(hypo_signal, color="crimson", linewidth=1.2)
    axes[0].set_title(
        f"Signal Hypo (label=1) - idx={hypo_idx} - patient={patient_ids[hypo_idx]}"
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(normal_signal, color="royalblue", linewidth=1.2)
    axes[1].set_title(
        f"Signal Normal (label=0) - idx={normal_idx} - patient={patient_ids[normal_idx]}"
    )
    axes[1].set_xlabel("Time (samples)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Random ECG Samples from train.npy", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
