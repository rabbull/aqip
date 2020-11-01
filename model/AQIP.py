import numpy as np
import torch
from torch import nn

from model.STBlock import STConvLayer
import os


class AQIP(nn.Module):
    def __init__(self, adj: torch.tensor, seq_len: int, kt: int, with_aqi: bool = True, act_fun="GLU"):
        super().__init__()
        self.hid_size = 128
        # TODO Change hid_size to 64 cause Error of:
        #  The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 2
        self.seq_len = seq_len
        # self.gat_layers = nn.ModuleList([
        #     GATLayer(input_dim=16 + int(with_aqi), output_dim=128, adj=adj),
        #     GATLayer(input_dim=128, output_dim=128, adj=adj),
        # ])
        # self.rnns = nn.ModuleList([
        #     nn.LSTM(input_size=128, hidden_size=self.hid_size, num_layers=4, bias=True, batch_first=True),
        #     nn.LSTM(input_size=128, hidden_size=self.hid_size, num_layers=4, bias=True, batch_first=True)
        # ])
        self.st_conv_blocks = nn.ModuleList([
            STConvLayer(input_dim=16 + int(with_aqi), output_dim=128, kt=kt, adj=adj, act_fun=act_fun),
            STConvLayer(input_dim=128, output_dim=128, kt=kt, adj=adj, act_fun=act_fun)
        ])
        time_step_length = self.seq_len - 2 * len(self.st_conv_blocks) * (kt - 1)
        print(time_step_length)
        self.linear = nn.Linear(in_features=128 * time_step_length, out_features=1, bias=True)

    def forward(self, x: torch.Tensor, site_idx: int):
        # torch.autograd.set_detect_anomaly(True)
        # h = torch.zeros(size=(4, x.size(0), self.hid_size)).cuda()
        # c = torch.zeros(size=(4, x.size(0), self.hid_size)).cuda()
        # for gat in self.gat_layers:
        #     x = gat(x)
        # for rnn in self.rnns:
        #     y = x[:, :, site_idx, :].clone()
        #     x[:, :, site_idx, :], (h, c) = rnn(y, (h, c))
        # h = h.permute(1, 0, 2)
        # h = h.reshape(h.size(0), -1)
        # h = self.linear(h)
        # h = h.squeeze()
        for st_conv in self.st_conv_blocks:
            x = st_conv(x)
        x = x[:, :, site_idx, :].clone()
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x.squeeze()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adj = torch.tensor(np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]]), dtype=torch.bool).cuda()
    exp = torch.randn(3, 50, 3, 17).cuda()
    model = AQIP(adj, seq_len=50, kt=3)
    model = model.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    pred = model(exp, 1)
    target = torch.randn(3).cuda()
    loss = criterion(pred, target)
    print(loss.backward())
    # TODO: verify
