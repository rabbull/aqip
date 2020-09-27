import numpy as np
import torch
from torch import nn

from model.GAT import GATLayer


class AQIP(nn.Module):
    def __init__(self, adj: np.array, seq_len: int, with_aqi: bool = True):
        super().__init__()
        self.hid_size = 128
        self.gat_layers = [
            GATLayer(input_dim=16 + int(with_aqi), output_dim=128, adj=adj),
            GATLayer(input_dim=128, output_dim=128, adj=adj),
        ]
        self.rnns = [
            nn.LSTM(input_size=128, hidden_size=128, num_layers=8, bias=True, batch_first=True),
        ]
        self.linear = nn.Linear(in_features=8, out_features=1, bias=True)

    def forward(self, x: torch.Tensor):
        h, c = torch.zeros(size=(8, x.size(0), 128)), torch.zeros(size=(8, x.size(0), 128))
        for gat in self.gat_layers:
            x = gat(x)
        for i in range(x.size(2)):
            for rnn in self.rnns:
                x[:, :, i, :], (h, c) = rnn(x[:, :, i, :], (h, c))
            print(h.shape)


if __name__ == '__main__':
    model = AQIP(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]]), seq_len=8)
    print(model(torch.randn(10, 8, 3, 17)))