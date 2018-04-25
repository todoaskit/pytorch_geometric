from __future__ import division
from torch.nn import Module, Parameter
from torch.autograd import Variable
import torch


class CartesianSphereAdj(Module):
    """Concatenates Cartesian spatial relations based on the position
    :math:`P \in \mathbb{R}^{N x D}` of graph nodes to the graph's edge
    attributes."""
    def __call__(self, data):
        row, col = data.index
        # Compute Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]
        lengths = torch.sqrt((weight**2).sum(1, keepdim=True))
        weight = weight * (1 / (2 * lengths))
        weight = weight + 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
