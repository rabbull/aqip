import numpy as np
import torch
from torch import nn
import math

'''
    The GConvLSTM replaces the matrix multiplication in the regular LSTM networks by the Graph Convolution 
    Here we only tried the dense representation of the adjacency matrix, which did not out perform the naive lstm.
'''


class GConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, adj: torch.tensor):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.adj = nn.Parameter(adj)  # The Graph convolution lstm requires the adjacency matrix to be a parameter
        self.n_sites = self.adj.shape[0]
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """ Assumes x is of shape (batch, sequence,n_stations, feature) """
        batch_size, seq_size, _, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.n_sites, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.n_sites, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_size):
            x_t = x[:, t, :, :]
            # batch the computations into a single matrix multiplication (from github)
            gates = GConv(x_t, self.W, self.adj) + GConv(h_t, self.U, self.adj) + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :, :HS]),  # input
                torch.sigmoid(gates[:, :, HS:HS * 2]),  # forget
                torch.tanh(gates[:, :, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, :, HS * 3:]),  # output
            )

            c_t = torch.mul(f_t, c_t) + torch.mul(i_t, g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


def GConv(H: torch.tensor, W: torch.tensor, adj: torch.tensor):
    """
    The Graph convolution in the paper is described as H * W = AHW
        where A is the adjacency matrix and H is the input with features and W is the weight of the layer
    """
    AH = torch.matmul(adj, H)
    AHW = torch.matmul(AH, W)

    return AHW


if __name__ == '__main__':
    model = GConvLSTM(17, 1024, torch.tensor(np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]], dtype='float32')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out, (h, c) = model(torch.randn(5, 7, 3, 17).to(device))
    print(out.shape)
