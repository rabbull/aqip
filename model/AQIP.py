import numpy as np
import torch
from torch import nn

from model.GAT import GATLayer
import os


class AQIP(nn.Module):
    def __init__(self, adj: torch.tensor, seq_len: int, with_aqi: bool = True):
        super().__init__()
        self.hid_size = 128
        # TODO Change hid_size to 64 cause Error of:
        #  The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 2
        self.seq_len = seq_len
        self.gat_layers = nn.ModuleList([
            GATLayer(input_dim=16 + int(with_aqi), output_dim=128, adj=adj),
            GATLayer(input_dim=128, output_dim=128, adj=adj),
        ])
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=128, hidden_size=self.hid_size, num_layers=4, bias=True, batch_first=True),
            nn.LSTM(input_size=128, hidden_size=self.hid_size, num_layers=4, bias=True, batch_first=True)
        ])
        self.linear = nn.Linear(in_features=128 * 4, out_features=1, bias=True)

    def forward(self, x: torch.Tensor, site_idx: int):
        torch.autograd.set_detect_anomaly(True)
        h = torch.zeros(size=(4, x.size(0), self.hid_size)).cuda()
        c = torch.zeros(size=(4, x.size(0), self.hid_size)).cuda()
        for gat in self.gat_layers:
            x = gat(x)
        for rnn in self.rnns:
            y = x[:, :, site_idx, :].clone()
            x[:, :, site_idx, :], (h, c) = rnn(y, (h, c))
        h = h.permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)
        h = self.linear(h)
        h = h.squeeze()
        return h


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adj = torch.tensor(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]]), dtype=torch.bool).cuda()
    exp = torch.randn(3, 8, 3, 17).cuda()
    gpus = [0]
    #with torch.autograd.set_detect_anomaly(True):
    model = AQIP(adj, seq_len=8)
    model = model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    pred = model(exp, 1)
    target = torch.randn(3).cuda()
    loss = criterion(pred, target)
    print(loss.backward())
    # TODO: verify
