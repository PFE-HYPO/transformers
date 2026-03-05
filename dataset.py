import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -------------------------
# Dataset
# -------------------------
class ECGStructuredNPYDataset(Dataset):
    """
    Expects structured numpy array with fields:
    - 'x' shape (2500,)
    - 'label' int
    - 'patient_id' string
    Returns:
    x: torch.FloatTensor (1, 2500)
    y: torch.FloatTensor scalar (0/1)
    pid: python str
    """
    def __init__(self, npy_path: str):
        arr = np.load(npy_path, allow_pickle=True)
        assert "x" in arr.dtype.names and "label" in arr.dtype.names and "patient_id" in arr.dtype.names, \
            f"Expected fields ('x','label','patient_id') in {npy_path}"
        self.x = arr["x"].astype(np.float32)            # (N, 2500)
        self.y = arr["label"].astype(np.int64)          # (N,)
        self.pid = arr["patient_id"].astype(str)        # (N,)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).unsqueeze(0)  # (1, 2500)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        pid = self.pid[idx]
        return x, y, pid
