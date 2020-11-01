import numpy as np
import torch
from torch import nn
from torch.nn import functional


class TempConvLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kt: int, act_fun: str = 'GLU'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kt = kt  # The size of the conv kernel
        self.act_fun = act_fun  # Indicating the activation function choice, GLU as default
        # Filter required for torch (out,in,H,W)
        if self.act_fun == 'GLU':
            self.W = nn.Parameter(torch.zeros(size=(2 * output_dim, input_dim, kt, 1)))
            self.a = nn.Parameter(torch.zeros(2 * output_dim))
        else:
            self.W = nn.Parameter(torch.zeros(size=(output_dim, input_dim, kt, 1)))
            self.a = nn.Parameter(torch.zeros(output_dim))
        if self.input_dim > self.output_dim:
            self.w_input = nn.Parameter(torch.zeros(size=(output_dim, input_dim, 1, 1)))

    def forward(self, x: torch.tensor):
        x = x.permute(0, 3, 1, 2)  # Change the data to shape (batch, channels, time, station) to meet the conv2d input
        _, _, T, n = x.shape
        if self.input_dim > self.output_dim:
            # bottleneck down-sampling
            # Manually padding SAME:  kernel_size − (input_length % stride)
            x_input = torch.nn.functional.conv2d(x, self.w_input, stride=1, padding=(self.kt - T, 0))
        elif self.input_dim < self.output_dim:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
            x_input = torch.cat([x, torch.zeros([x.shape[0], T, n, self.output_dim - self.input_dim])], 3)
        else:
            x_input = x

        x_input = x_input[:, :, self.kt - 1:T, :].clone()

        x_conv = torch.nn.functional.conv2d(x, self.W, self.a, stride=1)

        if self.act_fun == "GLU":
            return torch.nn.functional.glu(x_conv, dim=1).permute(0, 2, 3, 1)
        if self.act_fun == 'linear':
            return x_conv.permute(0, 2, 3, 1)
        elif self.act_fun == 'sigmoid':
            return torch.nn.functional.sigmoid(x_conv).permute(0, 2, 3, 1)
        elif self.act_fun == 'relu':
            return torch.nn.functional.relu(x_conv + x_input).permute(0, 2, 3, 1)
        else:
            raise ValueError(f'ERROR: activation function "{self.act_fun}" is not defined.')


if __name__ == '__main__':
    model = TempConvLayer(20, 10, 3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model(torch.randn(5, 5, 10, 20).to(device)).shape)
