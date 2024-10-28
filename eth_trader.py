import math
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

def positional_encoding(positions, d_model=64):
    batch_size = positions.size(0)
    pe = torch.zeros(batch_size, d_model, device=positions.device)
    position = positions.float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float().to(positions.device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def normalize_per_batch(tensor):
    batch_mean = tensor.mean(dim=tuple(range(1, tensor.dim())), keepdim=True)
    batch_std = tensor.std(dim=tuple(range(1, tensor.dim())), keepdim=True)
    normalized_tensor = (tensor - batch_mean) / (batch_std + 1e-8)
    return normalized_tensor

class TradingTransformer(nn.Module):
    def __init__(self, n_actions=100, d_model=64, num_embeddings=128, nhead=4, num_encoder_layers=4, dim_feedforward=256, dropout=0.1):
        super(TradingTransformer, self).__init__()

        self.n_actions = n_actions
        self.num_embeddings = num_embeddings
        self.window_embeds = nn.Embedding(num_embeddings, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.d_model = d_model

        # Change this to be 1D conv and add some non-locality
        self.obs_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.obs_transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.balance_ff = nn.Linear(d_model, d_model)
        self.eth_ff = nn.Linear(d_model, d_model)
        self.networth_ff = nn.Linear(d_model, d_model)

        self.policy = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 128),
            nn.GELU(),
            nn.Linear(128, n_actions + num_embeddings)
        )

        self.value = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, balance, eth, networth, window=None):
        obs = normalize_per_batch(obs)
        obs = obs.unsqueeze(1)  # Shape: [batch_size, 1, 50]
        obs = self.obs_conv(obs)  # Shape: [batch_size, d_model, 50]
        obs = obs.permute(0, 2, 1)  # Shape: [batch_size, 50, d_model]
        obs = self.obs_transformer(obs)  # Shape: [batch_size, 50, d_model]
        obs = obs.mean(dim=1)  # Global average pooling, Shape: [batch_size, d_model]
        obs = obs.unsqueeze(1)  # Shape: [batch_size, 1, d_model]

        balance = self.balance_ff(positional_encoding(balance, d_model=self.d_model)).unsqueeze(1)
        eth = self.eth_ff(positional_encoding(eth, d_model=self.d_model)).unsqueeze(1)
        networth = self.networth_ff(positional_encoding(networth, d_model=self.d_model)).unsqueeze(1)

        # Concatenate observations with balance, eth, and networth
        x = torch.cat([obs, balance, eth, networth], dim=1)

        # If window tokens are provided, embed them and concatenate along dimension 1
        if window is not None and window.shape[1] > 0:
            window_embed = self.window_embeds(window)  # Embed window tokens
            x = torch.cat([x, window_embed], dim=1)  # Concatenate along dim 1

        # Normalize x
        x_min = x.amin(dim=(1, 2), keepdim=True)
        x_max = x.amax(dim=(1, 2), keepdim=True)
        epsilon = 1e-8
        x = 2 * (x - x_min) / (x_max - x_min + epsilon) - 1

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        return F.softmax(self.policy(x[:, 0, :]), dim=-1), self.value(x[:, 1, :])