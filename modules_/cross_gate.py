import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc_gate1 = nn.Linear(d_model, d_model, bias=False)
        self.fc_gate2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.fc_gate1(x1))
        x2_ = g1 * x2
        g2 = torch.sigmoid(self.fc_gate2(x2))
        x1_ = g2 * x1
        return x1_, x2_
