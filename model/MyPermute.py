import torch
from torch import nn

class MyPermuteLayer(nn.Module):
    def __init__(self, permute_arg):
        super().__init__()
        self.permute_arg = permute_arg

    def forward(self, x: torch.tensor):
        return x.permute(self.permute_arg)
