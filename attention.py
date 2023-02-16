import torch
import numpy as np
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.dim) # batch matrix multiplication
        attn = F.softmax(score, dim=-1)
        output = torch.bmm(attn, value)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = hidden_dim // num_heads
        
        assert hidden_dim % self.num_heads == 0
        
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        
        # WQ, WK, WV
        self.query_projection = nn.Linear(hidden_dim, self.hidden_dim)
        self.key_projection   = nn.Linear(hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, self.hidden_dim)
        
        # WO
        self.out_projection = nn.Linear(hidden_dim << 1, self.hidden_dim, bias=True)
        
    def split_heads(self, inputs, batch_size):
        inputs = inputs.view(batch_size, -1, self.num_heads, self.dim)
        return inputs.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)

    def forward(self, query, value, prev_attn=None):
        batch_size = value.size(0)
        residual = query

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim)
        key = self.key_projection(value).view(batch_size, -1, self.num_heads, self.dim)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        context, attn = self.scaled_dot(query, key, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)
        
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)
        combined = torch.cat([context, residual], dim=2)

        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1, self.hidden_dim)
        
        return output, context