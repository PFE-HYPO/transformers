import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
# from xformers.ops import memory_efficient_attention
    
 # -------------------------
# Model: 1D CNN
# -------------------------
class ECG_CNN1D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, dropout=0.2):
        super().__init__()
        # A small but strong 1D CNN backbone
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True),

            # Global Average Pooling over time
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),                         # (B, C)
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)                       # logits
        )

    def forward(self, x):
        # x: (B, 1, 2500)
        feats = self.net(x)
        logits = self.head(feats)
        return logits.squeeze(-1)                 # (B,)
