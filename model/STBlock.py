import torch
from torch import nn
from model.GAT import GATLayer
from model.TempConv import TempConvLayer
import numpy as np

class STConvLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kt: int, adj: torch.tensor, act_fun: str = 'GLU'):
        super().__init__()
        self.layers = nn.ModuleList([
            TempConvLayer(input_dim, input_dim, kt, act_fun),
            GATLayer(input_dim, output_dim, adj),
            TempConvLayer(output_dim, output_dim, kt, act_fun)
        ])

    def forward(self, x: torch.tensor):
        for layer in self.layers:
            x = layer(x)
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        return x

if __name__ == '__main__':
    model = STConvLayer(17, 17, 3, torch.tensor(np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]], dtype='bool')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    exp = torch.randn(3, 20, 3, 17).cuda()
    print(model(exp))
