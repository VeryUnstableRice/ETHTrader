import math

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def positional_encoding(positions, d_model=64):
    # Create a tensor to store the positional encodings
    batch_size = positions.size(0)
    pe = torch.zeros(batch_size, d_model)

    # Compute the positional encodings
    position = positions.float()  # Convert positions to float for division
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def normalize_per_batch(tensor):
    """
    Normalizes a float tensor on a per-batch basis.

    Args:
        tensor (torch.Tensor): Input tensor of shape [batch, ...]

    Returns:
        torch.Tensor: Normalized tensor with mean 0 and std 1 for each batch.
    """
    batch_mean = tensor.mean(dim=tuple(range(1, tensor.dim())), keepdim=True)  # Mean per batch
    batch_std = tensor.std(dim=tuple(range(1, tensor.dim())), keepdim=True)  # Std per batch

    # Normalize each batch separately
    normalized_tensor = (tensor - batch_mean) / (batch_std + 1e-8)  # Adding a small epsilon to prevent division by zero

    return normalized_tensor


class TradingTransformer(nn.Module):
    def __init__(self, n_actions=100, d_model=64, nhead=4, num_encoder_layers=1, dim_feedforward=256, dropout=0.1):
        super(TradingTransformer, self).__init__()


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.d_model = d_model

        self.obs_ff = nn.Linear(50, d_model)
        self.balance_ff = nn.Linear(d_model, d_model)
        self.eth_ff = nn.Linear(d_model, d_model)
        self.networth_ff = nn.Linear(d_model, d_model)

        self.policy = nn.Linear(d_model, n_actions)
        self.value = nn.Linear(d_model, 1)

    def forward(self, obs, balance, eth, networth):
        obs = self.obs_ff(normalize_per_batch(obs)).unsqueeze(1)
        balance = self.balance_ff(positional_encoding(balance, d_model=self.d_model)).unsqueeze(1)
        eth = self.eth_ff(positional_encoding(eth, d_model=self.d_model)).unsqueeze(1)
        networth = self.networth_ff(positional_encoding(networth, d_model=self.d_model)).unsqueeze(1)

        x = torch.cat([obs, balance, eth, networth], dim=1)

        x = self.transformer_encoder(x)

        return F.softmax(self.policy(x[:,0,:]), dim=-1), self.value(x[:,1,:])