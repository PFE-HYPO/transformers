import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
# from xformers.ops import memory_efficient_attention
    
        

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x = self.gelu(self.fc1(x))
        return self.fc2(self.gelu(self.fc1(x)))
    


   
########################################################        
    
class MultiScaleWindowAttentionOptimized(nn.Module):
    def __init__(self, d_model, num_heads, window_sizes, max_seq_len):
        """
        Args:
            d_model (int): Dimensionality of the input embeddings.
            num_heads (int): Total number of attention heads.
            window_sizes (list): List of window sizes for each group of heads.
        """
        super(MultiScaleWindowAttentionOptimized, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_sizes = window_sizes        # Contains portions of the input
        self.num_groups = len(window_sizes)
        self.max_seq_len = max_seq_len

        # Ensure the number of heads is divisible by the number of groups
        assert num_heads % self.num_groups == 0, "num_heads must be divisible by the number of window sizes"

        # Linear layers for Q, K, V transformations
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable relative positional bias
        self.rel_pos_bias = nn.Parameter(torch.randn(2 * max_seq_len - 1, dtype=torch.float32))

    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: Input tensors of shape (batch_size, seq_len, d_model).
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = Q.shape

        # Apply linear transformations
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Reshape Q, K, V for multi-head attention to (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        K = rearrange(K, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        V = rearrange(V, "b s (h d_k) -> b h s d_k", h=self.num_heads)

        # Compute relative positional bias
        rel_pos = self.get_relative_positions(seq_len).to(Q.device)
        B = self.rel_pos_bias[rel_pos + self.max_seq_len - 1]  # Center the indices

        # Split heads into groups based on window sizes
        group_size = self.num_heads // self.num_groups
        outputs = []
        for i, window_size in enumerate(self.window_sizes):
            start = i * group_size
            end = start + group_size
            Q_group = Q[:, start:end, :, :]
            K_group = K[:, start:end, :, :]
            V_group = V[:, start:end, :, :]

            # Apply sliding window masking
            mask = self.create_sliding_window_mask(seq_len, int(seq_len * window_size)).to(Q.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

            # Compute memory-efficient attention with xFormers
            # output_group = memory_efficient_attention(
            #     Q_group, K_group, V_group, attn_bias=mask
            # )
            # outputs.append(output_group)

            # Compute attention scores with relative positional bias
            attn_scores = torch.matmul(Q_group, K_group.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
            attn_scores = attn_scores + B.unsqueeze(0).unsqueeze(0)  # Add relative positional bias

            # Apply mask (optional, depending on the use case)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

            # Compute attention probabilities
            attn_probs = torch.softmax(attn_scores, dim=-1)

            # Compute attention output
            output_group = torch.matmul(attn_probs, V_group)
            outputs.append(output_group)

        # Concatenate outputs from all groups
        output = torch.cat(outputs, dim=1)

        # Reshape back to original shape
        output = rearrange(output, "b h s d_k -> b s (h d_k)")

        # Apply final linear transformation
        output = self.W_o(output)
        return output
    
    def get_relative_positions(self, seq_len):
        """
        Compute relative positions for all pairs of tokens in the sequence.
        Args:
            seq_len (int): Length of the sequence.
        Returns:
            rel_pos (torch.Tensor): Relative position matrix of shape (seq_len, seq_len).
        """
        range_vec = torch.arange(seq_len)
        rel_pos = range_vec[:, None] - range_vec[None, :]  # Shape: (seq_len, seq_len)
        return rel_pos

    def create_sliding_window_mask(self, seq_len, window_size):
        """
        Create a sliding window mask for attention.
        Args:
            seq_len (int): Length of the sequence.
            window_size (int): Size of the sliding window.
        Returns:
            mask (torch.Tensor): Mask tensor of shape (seq_len, seq_len).
        """
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask

class MSWTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, window_sizes: list[float], max_seq_len: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.msa = MultiScaleWindowAttentionOptimized(d_model, num_heads, window_sizes, max_seq_len)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.msa(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))       # Residual connection
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))         # Residual connection
        return x

class DMSTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_heads: int, d_ff: int,
                 layer_sizes: list[float], window_sizes: list[float], max_seq_len: int,
                 hidden=32, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)

        # Compute window sizes for each block
        self.layer_window_sizes = [
            [ls * ws * max_seq_len for ws in window_sizes]
            for ls in layer_sizes
        ]

        # Create transformer blocks in parallel (each gets same input)
        self.transformer_blocks = nn.ModuleList([
            MSWTransformerBlock(d_model, num_heads, d_ff, self.layer_window_sizes[i], max_seq_len, dropout)
            for i in range(len(layer_sizes))
        ])

        # Project concatenated output from all blocks back to d_model
        self.output_projection = nn.Linear(d_model * len(layer_sizes), d_model)

        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input shape: (batch_size, seq_length, patch_size, num_leads)
        batch_size, seq_length, patch_size, num_leads = x.shape

        # Flatten the last two dims
        x = x.view(batch_size, seq_length, -1)  # (batch_size, seq_length, input_dim)

        # Embed to (batch_size, seq_length, d_model)
        x = self.embed(x)

        # Parallel processing through all transformer blocks
        block_outputs = [block(x) for block in self.transformer_blocks]  # list of (batch_size, seq_length, d_model)

        # Concatenate along feature dimension
        combined = torch.cat(block_outputs, dim=-1)  # (batch_size, seq_length, d_model * num_blocks)

        # Project back to original d_model
        x = self.output_projection(combined)  # (batch_size, seq_length, d_model)

        # Use CLS token (first position) as sequence representation
        x = x[:, 0, :]  # (batch_size, d_model)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
