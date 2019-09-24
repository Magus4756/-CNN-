import torch
from torch.nn import *


class PCA(Module):

    def __init__(self, k=0.99):
        super(PCA, self).__init__()
        self.k = k

    def forward(self, x):
        x_mean = torch.mean(x, 0)
        x = x - x_mean.expand_as(x)

        # svd
        U, S, V = torch.svd(torch.t(x))
        len_ = 1
        w = 0
        reserved_weight = sum(S) * self.k
        for _ in S:
            if w > reserved_weight:
                break
            w += _
            len_ += 1
        return torch.mm(x, U[:, :len_])
