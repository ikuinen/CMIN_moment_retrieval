import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones


class GraphConvolution(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.num_relations = 40
        self.fc_dir_weight = clones(nn.Linear(d_model, d_model, bias=False), 3)
        self.fc_dir_bias = [nn.Parameter(torch.zeros(d_model))
                            for _ in range(self.num_relations * 2 - 1)]
        self.fc_dir_bias1 = nn.ParameterList(self.fc_dir_bias[-1:])
        self.fc_dir_bias2 = nn.ParameterList(self.fc_dir_bias[:self.num_relations - 1])
        self.fc_dir_bias3 = nn.ParameterList(self.fc_dir_bias[self.num_relations - 1:-1])

        self.fc_gate_weight = clones(nn.Linear(d_model, d_model, bias=False), 3)
        self.fc_gate_bias = [nn.Parameter(torch.zeros(d_model))
                             for _ in range(self.num_relations * 2 - 1)]
        self.fc_gate_bias1 = nn.ParameterList(self.fc_gate_bias[-1:])
        self.fc_gate_bias2 = nn.ParameterList(self.fc_gate_bias[:self.num_relations - 1])
        self.fc_gate_bias3 = nn.ParameterList(self.fc_gate_bias[self.num_relations - 1:-1])

    def _compute_one_direction(self, x, fc, biases, adj_mat, relations, fc_gate, biases_gate):
        x = fc(x)
        g = fc_gate(x)
        out = None
        for r, bias, bias_gate in zip(relations, biases, biases_gate):
            mask = (adj_mat == r).float()
            g1 = torch.sigmoid(g + bias_gate)
            res = torch.matmul(mask, (x + bias) * g1)
            if out is None:
                out = res
            else:
                out += res
        return out

    def forward(self, node, node_mask, adj_mat):
        out = self._compute_one_direction(node, self.fc_dir_weight[1], self.fc_dir_bias2,
                                          adj_mat, range(2, self.num_relations + 1),
                                          self.fc_gate_weight[1], self.fc_gate_bias2)
        adj_mat = adj_mat.transpose(-1, -2)
        out += self._compute_one_direction(node, self.fc_dir_weight[2], self.fc_dir_bias3,
                                           adj_mat, range(2, self.num_relations + 1),
                                           self.fc_gate_weight[2], self.fc_gate_bias3)
        # adj_mat = torch.eye(adj_mat.size(1)).type_as(adj_mat)
        out += self._compute_one_direction(node, self.fc_dir_weight[0], self.fc_dir_bias1,
                                           adj_mat, [1],
                                           self.fc_gate_weight[0], self.fc_gate_bias1)
        return F.relu(out)


class GraphConvolution1(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, node, node_mask, adj_mat):
        yes = (adj_mat + adj_mat.transpose(-1, -2)) > 0
        degrees = torch.sum(yes.float(), -1)
        degrees_inv = 1.0 / torch.sqrt(degrees + 1 - yes.float())
        x = torch.eye(self.max_num_keys).type_as(degrees_inv).unsqueeze(0) * degrees_inv.unsqueeze(-1)
        normalize = torch.matmul(torch.matmul(x, yes.float()), x)
        x = node
        x = torch.matmul(normalize, x)  # [nb, len, hid]
        x = F.relu(self.fc(x))
        return x
