import torch.nn as nn

class LAS(nn.Module):
    def __init__(self, listener, speller):
        super(LAS, self).__init__()
        self.listener = listener
        self.speller = speller
        
    def forward(self, inputs, ground_truth=None, teacher_forcing_rate=0.9, use_beam=False, beam_size=16):
        listener_features, hidden = self.listener(inputs)
        logits = self.speller(listener_features, ground_truth, 
                              teacher_forcing_rate=teacher_forcing_rate, use_beam=use_beam, beam_size=beam_size)
        
        return logits