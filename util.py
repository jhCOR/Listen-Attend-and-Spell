import matplotlib.pyplot as plt
import math
import time
import torch
import numpy as np
from torch import nn

def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3, 4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError


def init_gate(bias):
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
    return bias