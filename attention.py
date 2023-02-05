import os
import torch
import librosa
import torchaudio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, value):
        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim) # batch matrix multiplication
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, value)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.key_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads)

        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value, prev_attn=None):
        batch_size = value.size(0)
        residual = query

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)

        context, attn = self.scaled_dot(query, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)
        combined = torch.cat([context, residual], dim=2)

        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1, self.hidden_dim)
        return output, context