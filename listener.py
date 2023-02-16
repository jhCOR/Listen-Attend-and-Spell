import torch
import torch.nn as nn
from torch.autograd import Variable

class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0, n_layers=1, bidirectional=True):
        super(BLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.LSTM(
            input_size=self.input_dim << 1,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            bidirectional = self.bidirectional,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs, hc = inputs
        
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        input_size = inputs.size(2)
        
        if seq_len % 2:
            zeros = torch.zeros((inputs.size(0), 1, inputs.size(2))).cuda()
            inputs = torch.cat([inputs, zeros], dim = 1)
            seq_len += 1
        inputs = inputs.contiguous().view(batch_size, int(seq_len / 2), input_size * 2)
        
        output, hc = self.rnn(inputs, hc)
        return (output, hc)

class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0, n_layers=1, bidirectional=True):
        super(Listener, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = True
        
        self.pblstm = nn.Sequential(
            BLSTM(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
                n_layers=n_layers,
                bidirectional = self.bidirectional
            ),
            BLSTM(
                input_dim=self.hidden_dim << 1 if self.bidirectional else 0,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
                n_layers=n_layers,
                bidirectional = self.bidirectional
            ),
            BLSTM(
                input_dim=self.hidden_dim << 1 if self.bidirectional else 0,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
                n_layers=n_layers,
                bidirectional = self.bidirectional
            ))
    
    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else 1, batch_size, self.hidden_dim))
        cell = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else 1, batch_size, self.hidden_dim))
        return (hidden.cuda(),cell.cuda())
    
    def forward(self, inputs):
        hc = self.init_hidden(inputs.size(0))
        output, state = self.pblstm((inputs, hc))
        return output, state
