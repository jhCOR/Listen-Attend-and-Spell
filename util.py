import matplotlib.pyplot as plt
import math
import time
import torch
import numpy as np
from torch import nn

def label_to_string(labels, id2char):
    """
    Converts label to string (number => Hangeul)

    Args:
        labels (list): number label
        id2char (dict): id2char[id] = ch
        eos_id (int): identification of <end of sequence>

    Returns: sentence
        - **sentence** (str or list): Hangeul representation of labels
    """
    sos_id = char2id['<sos>']
    eos_id = char2id['<eos>']
    if len(labels.shape) == 1:
        sentence = str()
        for label in labels:
            if label.item() == sos_id:
                continue
            if label.item() == eos_id:
                break
            sentence += id2char[label.item()]
        return sentence

    elif len(labels.shape) == 2:
        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == sos_id:
                    continue
                if label.item() == eos_id:
                    break
                sentence += id2char[label.item()]
            sentences.append(sentence)
        return sentences
    
def scheduler_sampling(epoch, e_min=1, ratio_s=0.9, ratio_e=0, n_epoch_ramp=10):
    if epoch>e_min:
        epoch -= e_min
        teacher_forcing_ratio = max(ratio_s - (ratio_s-ratio_e)*epoch/n_epoch_ramp, ratio_e)
    else:
        teacher_forcing_ratio = 0.9
    return teacher_forcing_ratio

def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr