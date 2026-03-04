import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import pandas as pd

class ECGDataset(Dataset):
    def __init__(self, csv_file=None, data=None, label_column='label', transform=None):
        """
        Args:
            csv_file (str, optional): Path to the CSV file.
            data (pd.DataFrame, optional): DataFrame containing the data.
            label_column (str): Name of the column containing labels.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either `csv_file` or `data` must be provided.")

        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Ensure the label column exists
        if label_column not in self.data.columns:
            raise KeyError(f"Label column '{label_column}' not found in DataFrame.")

        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row at index `idx`
        row = self.data.iloc[idx]

        # Extract features and label
        features = row.drop(self.label_column).values.astype(float)  # Exclude the label column
        label = row[self.label_column]  # Get the label

        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Apply any transformations (if provided)
        if self.transform:
            features = self.transform(features)

        return features, label
    
class ConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]
    
class ECGTensorDataset(Dataset):    
    def __init__(self, X_tensor, y_tensor):
        """
        Custom dataset for ECG data.
        :param X_tensor: torch.Tensor of shape (num_samples, num_patches, patch_size, num_leads)
        :param y_tensor: torch.Tensor of shape (num_samples,) with binary labels (0 or 1)
        """
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.X)  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def split_dataset(dataset_loader: Dataset, train_ratio=0.8, batch_size=32):
    train_size = int(train_ratio * len(dataset_loader))
    test_size = len(dataset_loader) - train_size

    train_dataset, test_dataset = random_split(dataset_loader, [train_size, test_size])

    batch_size = 64
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        return model

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate Binary Cross-Entropy Loss for each sample
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute pt (model confidence on true class)
        pt = torch.exp(-BCE_loss)
        
        # Apply the focal adjustment
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
