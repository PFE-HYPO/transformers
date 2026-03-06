import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Charger dataset
# ===============================

data = np.load("split_out/train.npy", allow_pickle=True)

signals = data["x"]      # (N, 2500)
labels = data["label"]

print("Dataset shape:", signals.shape)

# ===============================
# Statistiques globales
# ===============================

print("\nGLOBAL STATS")
print("Mean:", np.mean(signals))
print("Std:", np.std(signals))
print("Min:", np.min(signals))
print("Max:", np.max(signals))

# ===============================
# Statistiques par segment
# ===============================

segment_means = np.mean(signals, axis=1)
segment_stds = np.std(signals, axis=1)

print("\nSEGMENT STATS")
print("Mean of segment means:", np.mean(segment_means))
print("Std of segment means:", np.std(segment_means))

print("Mean of segment std:", np.mean(segment_stds))
print("Std of segment std:", np.std(segment_stds))


normal = signals[labels==0]
hypo = signals[labels==1]

print("Normal mean:", np.mean(normal))
print("Normal std:", np.std(normal))

print("Hypo mean:", np.mean(hypo))
print("Hypo std:", np.std(hypo))
# ===============================
# Histogram amplitude
# ===============================

plt.figure(figsize=(10,4))

plt.hist(signals.flatten(), bins=200)
plt.title("Amplitude distribution")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")

plt.show()

# ===============================
# Histogram std segments
# ===============================

plt.figure(figsize=(10,4))

plt.hist(segment_stds, bins=100)
plt.title("Std per segment distribution")
plt.xlabel("Segment STD")

plt.show()