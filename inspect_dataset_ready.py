import numpy as np


TRAIN_PATH = "split_out/train.npy"
TEST_PATH = "split_out/test.npy"


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def class_counts(labels):
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    total = len(labels)
    p0 = 100.0 * n0 / total if total else 0.0
    p1 = 100.0 * n1 / total if total else 0.0
    return n0, n1, p0, p1


def describe_split(name, arr):
    x = arr["x"].astype(np.float32)
    y = arr["label"].astype(np.int32)
    p = arr["patient_id"].astype(str)

    n0, n1, p0, p1 = class_counts(y)

    print(f"\n[{name}]")
    print(f"Segments           : {len(arr)}")
    print(f"Signal shape       : {x.shape}")
    print(f"Unique patients    : {len(np.unique(p))}")
    print(f"Normal (0)         : {n0} ({p0:.2f}%)")
    print(f"Hypo   (1)         : {n1} ({p1:.2f}%)")

    print("\nGlobal normalization stats")
    print(f"Mean               : {x.mean():.8f}")
    print(f"Std                : {x.std():.8f}")
    print(f"Min                : {x.min():.6f}")
    print(f"Max                : {x.max():.6f}")

    seg_means = x.mean(axis=1)
    seg_stds = x.std(axis=1)

    print("\nPer-segment stats")
    print(f"Mean(seg means)    : {seg_means.mean():.8f}")
    print(f"Std(seg means)     : {seg_means.std():.8f}")
    print(f"Mean(seg stds)     : {seg_stds.mean():.8f}")
    print(f"Std(seg stds)      : {seg_stds.std():.8f}")
    print(f"Min(seg stds)      : {seg_stds.min():.8f}")
    print(f"Max(seg stds)      : {seg_stds.max():.8f}")

    x0 = x[y == 0]
    x1 = x[y == 1]

    print("\nPer-class stats")
    print(f"Normal mean/std    : {x0.mean():.8f} / {x0.std():.8f}")
    print(f"Hypo mean/std      : {x1.mean():.8f} / {x1.std():.8f}")

    # Simple morphology/energy summaries
    abs_mean = np.mean(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    ptp = np.ptp(x, axis=1)

    print("\nPer-segment feature summaries")
    print(f"AbsMean normal/hypo: {abs_mean[y==0].mean():.6f} / {abs_mean[y==1].mean():.6f}")
    print(f"RMS normal/hypo    : {rms[y==0].mean():.6f} / {rms[y==1].mean():.6f}")
    print(f"PTP normal/hypo    : {ptp[y==0].mean():.6f} / {ptp[y==1].mean():.6f}")

    # Per-patient table
    header(f"{name} - per patient summary")
    patients = np.unique(p)
    print(f"{'patient':<10} {'normal':>8} {'hypo':>8} {'total':>8} {'hypo_%':>10}")
    for pid in patients:
        mask = (p == pid)
        yy = y[mask]
        n0p = int(np.sum(yy == 0))
        n1p = int(np.sum(yy == 1))
        tot = n0p + n1p
        hp = 100.0 * n1p / tot if tot else 0.0
        print(f"{pid:<10} {n0p:>8} {n1p:>8} {tot:>8} {hp:>9.2f}")

    return x, y, p


def centroid_separability(x, y, name):
    header(f"{name} - simple class separability")

    x0 = x[y == 0]
    x1 = x[y == 1]

    c0 = x0.mean(axis=0)
    c1 = x1.mean(axis=0)

    d_centroids = np.linalg.norm(c0 - c1)

    # Distance of samples to their class centroid on a subset for speed
    rng = np.random.default_rng(42)
    n_sub = min(1000, len(x0), len(x1))

    idx0 = rng.choice(len(x0), size=n_sub, replace=False)
    idx1 = rng.choice(len(x1), size=n_sub, replace=False)

    d0 = np.linalg.norm(x0[idx0] - c0, axis=1).mean()
    d1 = np.linalg.norm(x1[idx1] - c1, axis=1).mean()
    intra = 0.5 * (d0 + d1)

    ratio = d_centroids / intra if intra > 0 else 0.0

    print(f"Centroid distance          : {d_centroids:.6f}")
    print(f"Mean intra-class distance  : {intra:.6f}")
    print(f"Separation ratio           : {ratio:.6f}")

    print("\nInterpretation:")
    print("- ratio << 1  : classes are highly overlapping in raw signal")
    print("- ratio ~ 1   : moderate separation")
    print("- ratio > 1   : strong separation")

    # Correlation between class mean signals
    corr = np.corrcoef(c0, c1)[0, 1]
    print(f"Mean-signal correlation    : {corr:.6f}")
    print("Interpretation:")
    print("- corr close to 1 means average hypo and normal waveforms are very similar")


def train_test_shift(x_train, y_train, x_test, y_test):
    header("TRAIN vs TEST distribution shift")

    print("Overall")
    print(f"Train mean/std : {x_train.mean():.8f} / {x_train.std():.8f}")
    print(f"Test  mean/std : {x_test.mean():.8f} / {x_test.std():.8f}")

    for cls, cls_name in [(0, "Normal"), (1, "Hypo")]:
        tr = x_train[y_train == cls]
        te = x_test[y_test == cls]

        ctr = tr.mean(axis=0)
        cte = te.mean(axis=0)

        d = np.linalg.norm(ctr - cte)
        corr = np.corrcoef(ctr, cte)[0, 1]

        print(f"\n{cls_name}")
        print(f"Train count      : {len(tr)}")
        print(f"Test count       : {len(te)}")
        print(f"Centroid L2 dist : {d:.6f}")
        print(f"Centroid corr    : {corr:.6f}")

    print("\nInterpretation:")
    print("- very large train/test shift can make generalization difficult")
    print("- centroid corr near 1 is usually a good sign")


def overlap_info(p_train, p_test):
    header("Patient overlap check")
    s_train = set(p_train.tolist())
    s_test = set(p_test.tolist())
    overlap = sorted(list(s_train.intersection(s_test)))

    print(f"Train patients : {sorted(list(s_train))}")
    print(f"Test patients  : {sorted(list(s_test))}")
    print(f"Overlap        : {overlap}")
    print(f"Overlap count  : {len(overlap)}")

    if len(overlap) == 0:
        print("Good: no patient leakage between train and test.")
    else:
        print("Warning: patient leakage detected.")


def main():
    header("LOADING DATA")
    train = np.load(TRAIN_PATH, allow_pickle=True)
    test = np.load(TEST_PATH, allow_pickle=True)

    x_train, y_train, p_train = describe_split("TRAIN", train)
    x_test, y_test, p_test = describe_split("TEST", test)

    overlap_info(p_train, p_test)
    centroid_separability(x_train, y_train, "TRAIN")
    centroid_separability(x_test, y_test, "TEST")
    train_test_shift(x_train, y_train, x_test, y_test)

    header("FINAL QUICK CHECK")
    print("A dataset is usually acceptable for training if:")
    print("1) no patient overlap")
    print("2) train/test class ratios are not extremely different")
    print("3) normalization is stable (mean~0, std~1)")
    print("4) there is not too much train/test shift")
    print("5) class separability ratio is not almost zero")
    print("\nSend me the full output, and I will interpret it for you.")


if __name__ == "__main__":
    main()