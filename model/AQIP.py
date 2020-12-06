import numpy as np
import torch
from torch import nn
import config
from torch.autograd import Variable

from model.GAT import GATLayer
from model.STBlock import STConvLayer
from model.GConvLSTM import GConvLSTM
import os


class AQIP(nn.Module):
    def __init__(self, adj: torch.tensor, seq_len: int, kt: int, with_aqi: bool = True, act_fun="GLU"):
        super().__init__()
        self.hid_size = 64
        self.num_layers = 1
        self.adj = adj
        # TODO Change hid_size to 64 cause Error of:
        #  The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 2
        self.seq_len = seq_len
        self.GConvLSTMlayers = nn.ModuleList([
            GConvLSTM(input_size=16 + int(with_aqi), hidden_size=self.hid_size, adj=self.adj),
            GConvLSTM(input_size=self.hid_size, hidden_size=self.hid_size, adj=self.adj),
        ])
        self.linear = nn.Linear(in_features=self.hid_size, out_features=1, bias=True)

        # For GAT + LSTM performance, use the code below:
        '''
        self.gat_layers = nn.ModuleList([
            GATLayer(input_dim=16 + int(with_aqi), output_dim=64, adj=adj),
            GATLayer(input_dim=64, output_dim=128, adj=adj),
        ])
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=128, hidden_size=self.hid_size, num_layers=self.num_layers,
                    batch_first=True),
            nn.LSTM(input_size=self.hid_size, hidden_size=self.hid_size, num_layers=self.num_layers,
                    batch_first=True),
        ])
        self.linear = nn.Linear(in_features=self.hid_size * self.num_layers, out_features=1, bias=True)
        '''

        # For Spatial Temporal Convolution net, use the code below:
        '''
        self.st_conv_blocks = nn.ModuleList([
            STConvLayer(input_dim=16 + int(with_aqi), output_dim=128, kt=kt, adj=adj, act_fun=act_fun),
            STConvLayer(input_dim=128, output_dim=128, kt=kt, adj=adj, act_fun='sigmoid'),
        ])
        time_step_length = self.seq_len - 2 * len(self.st_conv_blocks) * (kt - 1)
        self.linear = nn.Linear(in_features=128 * time_step_length, out_features=1, bias=True)
        '''



    def forward(self, x: torch.Tensor, site_idx: int):
        h = Variable(torch.zeros(x.shape[0], self.adj.shape[0], self.hid_size).cuda())
        c = Variable(torch.zeros(x.shape[0], self.adj.shape[0], self.hid_size).cuda())
        for GConvlstm in self.GConvLSTMlayers:
            x, (h, c) = GConvlstm(x, (h, c))
        h = h[:, site_idx, :].clone()
        h = h.reshape(x.shape[0], -1)
        h = self.linear(h)
        h = h.squeeze()
        return h

        # GAT + LSTM code:
        '''
        for gat in self.gat_layers:
            x = gat(x)
        h = Variable(torch.zeros(size=(self.num_layers, x.shape[0], self.hid_size)).cuda())
        c = Variable(torch.zeros(size=(self.num_layers, x.shape[0], self.hid_size)).cuda())
        y = x[:, :, site_idx, :].clone()
        for rnn in self.rnns:
            y, (h, c) = rnn(y, (h, c))
            h = torch.tanh(h)
            c = torch.tanh(c)
            y = torch.tanh(y)
        h = h.reshape(x.shape[0], -1)
        h = self.linear(h)
        h = h.squeeze()
        return h
        '''

        # For the Spatial Temporal Convolution Net, use the following code
        '''
        x = self.st_conv_blocks[0](x)
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        x = self.st_conv_blocks[1](x)
        x = x[:, :, site_idx, :].clone()
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = x.squeeze()
        return x
        '''



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adj = torch.tensor(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), dtype=torch.float).cuda()
    exp = torch.randn(config.BATCH_SIZE, config.SEQ_LENGTH, 3, 17).cuda()
    model = AQIP(adj, seq_len=config.SEQ_LENGTH, kt=3)
    model = model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    pred = model(exp, 1)
    print(pred.shape)
    target = torch.randn(config.BATCH_SIZE).cuda()
    loss = criterion(pred, target)
    loss.backward()
    # TODO: verify
