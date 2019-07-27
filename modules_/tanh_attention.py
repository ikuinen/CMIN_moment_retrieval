import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhAttention(nn.Module):
    def __init__(self, d_model, dropout=0.0, direction=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=False)
        self.wst = nn.Linear(d_model, 1, bias=False)
        self.direction = direction

    def forward(self, x, memory, memory_mask=None):
        item1 = self.ws1(x)  # [nb, len1, d]
        item2 = self.ws2(memory)  # [nb, len2, d]
        # print(item1.shape, item2.shape)
        item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
        S = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S = S.masked_fill(memory_mask == 0, -1e30)
            # for forward, backward, S: [nb, len, len]
            if self.direction == 'forward':
                length = S.size(1)
                forward_mask = torch.ones(length, length)
                for i in range(1, length):
                    forward_mask[i, 0:i] = 0
                S = S.masked_fill(forward_mask.cuda().unsqueeze(0) == 0, -1e30)
            elif self.direction == 'backward':
                length = S.size(1)
                backward_mask = torch.zeros(length, length)
                for i in range(0, length):
                    backward_mask[i, 0:i + 1] = 1
                S = S.masked_fill(backward_mask.cuda().unsqueeze(0) == 0, -1e30)
        S = self.dropout(F.softmax(S, -1))
        return torch.matmul(S, memory)  # [nb, len1, d]
