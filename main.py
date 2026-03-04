import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
from models import DMSTransformer
from utils import ECGTensorDataset, FocalLoss
from training import fit_evaluate_stop_with_recall

# -----------------------------
# Constants (from the article)
# -----------------------------
PATCH_SIZE = 5
NUM_LEADS = 1
MAX_SEQ_LEN = 2500  # points per segment (10s)
D_MODEL = 512
D_FF = 1024
HIDDEN = 32
DROPOUT = 0.2

WINDOW_SIZES = [0.2, 0.4, 0.8, 1.0]
NUM_HEADS = len(WINDOW_SIZES)  # 4
LAYER_SIZES = [1.0]            # one block

LR = 0.001
ALPHA = 0.9
GAMMA = 3
EPOCHS = 100
PATIENCE = 10
DELTA = 0.001
BATCH = 64

# -----------------------------
# Load data
# -----------------------------
def resolve_numpy_path(path: str) -> str:
    """
    Resolve a split file path.
    Supports:
    - real .npy/.npz files
    - symlinks
    - text pointer files containing an absolute path (frequent on Drive)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")

    if p.is_symlink():
        target = str(p.resolve())
        print(f"[INFO] Following symlink: {p} -> {target}")
        return target

    try:
        with open(p, "rb") as f:
            header = f.read(8)
    except OSError:
        return str(p)

    # .npy magic: b'\\x93NUMPY', .npz zip magic: b'PK'
    if header.startswith(b"\x93NUMPY") or header.startswith(b"PK"):
        return str(p)

    # If not numpy header, try path-pointer content.
    try:
        pointer = p.read_text(encoding="utf-8").strip()
    except Exception:
        return str(p)

    if pointer and (pointer.startswith("/") or (len(pointer) > 2 and pointer[1:3] == ":\\")):
        target = Path(pointer)
        if target.exists():
            print(f"[INFO] Pointer file detected: {p} -> {target}")
            return str(target)

    return str(p)


def load_split(path: str):
    """Load split arrays with a safe default and a pickle fallback for legacy files."""
    resolved_path = resolve_numpy_path(path)
    try:
        return np.load(resolved_path, allow_pickle=False)
    except ValueError as exc:
        if "Cannot load file containing pickled data" in str(exc):
            print(f"[INFO] {resolved_path} requires allow_pickle=True (legacy/object format).")
            try:
                return np.load(resolved_path, allow_pickle=True)
            except Exception as nested_exc:
                header = b""
                try:
                    with open(resolved_path, "rb") as f:
                        header = f.read(16)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Failed to load '{resolved_path}'. The file is not a valid .npy/pickle payload "
                    f"(header={header!r}). If you used a symlink on Drive, use a real copy "
                    "or set SPLIT_DIR to the real dataset folder."
                ) from nested_exc
        raise


SPLIT_DIR = os.environ.get("SPLIT_DIR", "split_out")
print(f"[INFO] Using SPLIT_DIR={SPLIT_DIR}")
train = load_split(os.path.join(SPLIT_DIR, "train.npy"))
test = load_split(os.path.join(SPLIT_DIR, "test.npy"))

def extract_xy(arr):
    # New format: structured npy with fields ('x', 'label', 'patient_id')
    if arr.dtype.names is not None and "x" in arr.dtype.names and "label" in arr.dtype.names:
        x_np = arr["x"].astype(np.float32)
        y_np = arr["label"].astype(np.float32)
        return x_np, y_np

    # Old format fallback: dense matrix [features..., label, patient_code]
    if arr.ndim == 2 and arr.shape[1] >= 2 and arr.dtype != object:
        x_np = arr[:, :-2].astype(np.float32)
        y_np = arr[:, -2].astype(np.float32)
        return x_np, y_np

    # Legacy pickle/object formats
    if arr.dtype == object:
        # Case A: 2D object matrix with rows like [x, label, patient_id]
        if arr.ndim == 2 and arr.shape[1] >= 2:
            first_x = arr[0, 0]
            if isinstance(first_x, (list, tuple, np.ndarray)):
                x_np = np.stack([np.asarray(row[0], dtype=np.float32) for row in arr], axis=0)
                y_np = np.asarray([row[1] for row in arr], dtype=np.float32)
                return x_np, y_np

        # Case B: 1D object array with dicts/tuples
        if arr.ndim == 1 and len(arr) > 0:
            first = arr[0]

            if isinstance(first, dict) and "x" in first and "label" in first:
                x_np = np.stack([np.asarray(item["x"], dtype=np.float32) for item in arr], axis=0)
                y_np = np.asarray([item["label"] for item in arr], dtype=np.float32)
                return x_np, y_np

            if isinstance(first, (list, tuple)) and len(first) >= 2:
                x_np = np.stack([np.asarray(item[0], dtype=np.float32) for item in arr], axis=0)
                y_np = np.asarray([item[1] for item in arr], dtype=np.float32)
                return x_np, y_np

    raise ValueError(
        "Unsupported split format. Expected structured npy with fields ('x','label') "
        "or a 2D matrix [features..., label, patient_code], "
        "or legacy object rows with [x, label, patient_id]/dicts."
    )


# Extract ECG points and labels
X_train_np, y_train_np = extract_xy(train)
X_test_np, y_test_np = extract_xy(test)

# -----------------------------
# Reshape to patches of size 5
# -----------------------------
def to_patches(x_np, patch_size=5, num_leads=1):
    # x_np: (N, 2500)
    N, L = x_np.shape
    assert L % patch_size == 0, f"L={L} must be divisible by patch_size={patch_size}"
    S = L // patch_size  # tokens
    x = torch.tensor(x_np, dtype=torch.float32).view(N, S, patch_size, num_leads)
    return x

X_train = to_patches(X_train_np, PATCH_SIZE, NUM_LEADS)  # (N, 500, 5, 1)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

X_test = to_patches(X_test_np, PATCH_SIZE, NUM_LEADS)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

train_loader = DataLoader(ECGTensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(ECGTensorDataset(X_test, y_test), batch_size=BATCH, shuffle=False)

# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DMSTransformer(
    input_dim=PATCH_SIZE * NUM_LEADS,   # 5
    d_model=D_MODEL,                    # 512
    num_heads=NUM_HEADS,                # 4
    d_ff=D_FF,                          # 1024
    layer_sizes=LAYER_SIZES,            # [1.0]
    window_sizes=WINDOW_SIZES,          # [0.2, 0.4, 0.8, 1.0]
    max_seq_len=MAX_SEQ_LEN,            # 2500
    hidden=HIDDEN,                      # 32
    dropout=DROPOUT                     # 0.2
).to(device)

criterion = FocalLoss(alpha=ALPHA, gamma=GAMMA)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

fit_evaluate_stop_with_recall(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    train_data_loader=train_loader,
    test_data_loader=val_loader,
    embedding=True,
    num_epochs=EPOCHS,
    patience=PATIENCE,
    delta=DELTA
)
