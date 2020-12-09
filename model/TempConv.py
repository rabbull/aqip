import torch
from torch import nn



class TempConvLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kt: int, act_fun: str = 'GLU'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kt = kt  # The size of the temp conv kernel
        self.act_fun = act_fun  # Indicating the activation function choice, GLU as default
        # Filter required for torch (out,in,H,W)
        if self.act_fun == 'GLU':
            self.W = nn.Parameter(torch.randn(size=(2 * output_dim, input_dim, kt, 1)))
            self.a = nn.Parameter(torch.randn(2 * output_dim))
        else:
            self.W = nn.Parameter(torch.randn(size=(output_dim, input_dim, kt, 1)))
            self.a = nn.Parameter(torch.randn(output_dim))
        if self.input_dim > self.output_dim:
            self.w_input = nn.Parameter(torch.zeros(size=(output_dim, input_dim, 1, 1)))

    def forward(self, x: torch.tensor):
        x = x.permute(0, 3, 1, 2)  # Change the data to shape (batch, channels, time, station) to meet the conv2d input
        _, _, T, n = x.shape
        if self.input_dim > self.output_dim:
            # bottleneck down-sampling
            # Manually padding SAME:  kernel_size − (input_length % stride)
            # TODO Padding not correct when output dim is smaller
            x_input = torch.nn.functional.conv2d(x, self.w_input, stride=(1, 1), padding=(self.kt - T, 0))
        elif self.input_dim < self.output_dim:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.output_dim - self.input_dim, T, n]).cuda()], 1)
        else:
            x_input = x

        x_input = x_input[:, :, self.kt - 1:T, :].clone()

        x_conv = torch.nn.functional.conv2d(x, self.W, self.a, stride=(1, 1))

        if self.act_fun == "GLU":
            return ((x_conv[:, 0:self.output_dim, :, :] + x_input) * torch.sigmoid(
                x_conv[:, -self.output_dim:, :, :])).permute(0, 2, 3, 1)
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
    exp = torch.randn(5, 5, 3, 20).cuda()
    print(model(exp).shape)
