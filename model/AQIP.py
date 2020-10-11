import numpy as np
import torch
from torch import nn

from model.GAT import GATLayer
import os


class AQIP(nn.Module):
    def __init__(self, adj: torch.tensor, seq_len: int, with_aqi: bool = True):
        super().__init__()
        self.hid_size = 128
        self.seq_len = seq_len
        # self.gat_layers = [
        #     GATLayer(input_dim=16 + int(with_aqi), output_dim=128, adj=adj).cuda(),
        #     GATLayer(input_dim=128, output_dim=128, adj=adj).cuda(),
        # ]
        self.GAT_1 = GATLayer(input_dim=16 + int(with_aqi), output_dim=128, adj=adj)
        self.GAT_2 = GATLayer(input_dim=128, output_dim=128, adj=adj)
        # self.rnns = [
        #     nn.LSTM(input_size=128, hidden_size=128, num_layers=4, bias=True, batch_first=True).cuda(),
        # ]
        self.RNN_1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=4, bias=True, batch_first=True)
        self.linear = nn.Linear(in_features=128 * 4, out_features=1, bias=True)
        #self.linear = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x: torch.Tensor, site_idx: int):
        h = torch.zeros(size=(4, x.size(0), 128)).cuda()
        c = torch.zeros(size=(4, x.size(0), 128)).cuda()
        # for gat in self.gat_layers:
        #     x = gat(x)
        x = self.GAT_1(x)
        x = self.GAT_2(x)
        # for rnn in self.rnns:
        #     x[:, :, site_idx, :], (h, c) = rnn(x[:, :, site_idx, :], (h, c))
        x[:, :, site_idx, :], (h, c) = self.RNN_1(x[:, :, site_idx, :], (h, c))
        h = h.permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)
        return self.linear(h).squeeze()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adj = torch.tensor(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]]), dtype=torch.bool).cuda()
    exp = torch.randn(3, 8, 3, 17).cuda()
    gpus = [0]
    model = AQIP(adj, seq_len=8)
    model = model.to(device)
    print(model(exp, 1))
    # TODO: verify
