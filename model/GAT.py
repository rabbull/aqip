import numpy as np
import torch
from torch import nn
from torch.nn import functional


class GATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adj: torch.tensor):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(size=(output_dim, input_dim)))
        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim,)))
        self.adj = adj
        self.n_points = adj.shape[0]

    def forward(self, h: torch.Tensor):
        B, T, N, F = h.size()
        hh = functional.linear(h, self.W)
        output = torch.zeros_like(hh)
        for i in range(self.n_points):
            hhj = hh[:, :, self.adj[i], :]
            hhi = torch.cat([hh[:, :, i:i + 1, :].clone()] * hhj.size(2), 2)
            hhij = torch.cat([hhi, hhj], 3)
            e = torch.mm(hhij.reshape(B * T * hhj.size(2), -1), self.a.reshape(self.a.size(0), 1)).reshape(B, T, -1)
            alpha = functional.softmax(e, dim=2)
            output[:, :, i, :] = torch.sum(hhj * torch.cat([torch.unsqueeze(alpha, 3)] * hhj.size(3), 3), dim=2)
        return output


if __name__ == '__main__':
    model = GATLayer(3, 1024, torch.tensor(np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]], dtype='bool')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model(torch.randn(5, 5, 3, 3)).shape)
