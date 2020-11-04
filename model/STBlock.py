import torch
from torch import nn
from model.GAT import GATLayer
from model.TempConv import TempConvLayer
from model.MyPermute import MyPermuteLayer
import numpy as np


class STConvLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kt: int, adj: torch.tensor, act_fun: str = 'GLU'):
        super().__init__()
        self.layers = nn.ModuleList([
            TempConvLayer(input_dim, input_dim, kt, act_fun),
            MyPermuteLayer([0, 3, 1, 2]),  # The BatchNorm Layer supports [Batch_size, Channel, T, Sites] format
            torch.nn.BatchNorm2d(input_dim),
            MyPermuteLayer([0, 2, 3, 1]),  # Change it back to [Batch_size, T, Sites, Channel]
            GATLayer(input_dim, output_dim, adj),
            TempConvLayer(output_dim, output_dim, kt, act_fun),
            MyPermuteLayer([0, 3, 1, 2]),  # The BatchNorm Layer supports [Batch_size, Channel, T, Sites] format
            torch.nn.BatchNorm2d(output_dim),
            MyPermuteLayer([0, 2, 3, 1]),  # Change it back to [Batch_size, T, Sites, Channel]
        ])

    def forward(self, x: torch.tensor):
        for layer in self.layers:
            x = layer(x)
        # print(x.shape)
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        return x


if __name__ == '__main__':
    model = STConvLayer(17, 64, 3, torch.tensor(np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]], dtype='bool')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    exp = torch.randn(3, 20, 3, 17).cuda()
    print(model(exp))
