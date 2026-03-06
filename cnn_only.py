import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score


# ===============================
# Dataset
# ===============================

class ECGDataset(Dataset):
    def __init__(self, npy_file):
        data = np.load(npy_file, allow_pickle=True)

        self.X = torch.from_numpy(np.array(data["x"], dtype=np.float32, copy=True))
        self.y = torch.from_numpy(np.array(data["label"], dtype=np.float32, copy=True))

        # reshape -> (N,1,2500)
        self.X = self.X.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===============================
# CNN Model
# ===============================

class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 312, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.flatten(1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ===============================
# Load dataset
# ===============================

train_dataset = ECGDataset("split_out/train.npy")
test_dataset = ECGDataset("split_out/test.npy")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# ===============================
# Device
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECGCNN().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ===============================
# Training loop
# ===============================

for epoch in range(20):

    model.train()
    train_loss = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X).squeeze()

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ===============================
    # Evaluation
    # ===============================

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            outputs = model(X).squeeze()

            prob = torch.sigmoid(outputs)

            pred = (prob > 0.5).cpu().numpy()

            preds.extend(pred)
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")