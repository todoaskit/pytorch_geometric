from __future__ import division
from torch.nn import Module, Parameter
from math import log

import torch


class LogCartesianAdj(Module):
    def __init__(self, scale=1, trainable=False):
        super(LogCartesianAdj, self).__init__()
        self.trainable = trainable
        if trainable:
            scale = torch.FloatTensor([scale]).cuda()
            self.scale = Parameter(scale)
        else:
            self.scale = scale

    def __call__(self, data):
        row, col = data.index
        if self.trainable:
            norm = 1 / torch.log(1 + self.scale)
        else:
            norm = 1 / log(1 + self.scale)


        # Compute Log-Cartesian pseudo-coordinates.
        weight = data.pos[col] - data.pos[row]
        mask = 1 - 2 * (weight < 0).type_as(weight)
        weight /= weight.abs().max()
        weight = weight.abs()

        weight = torch.log(self.scale * weight.abs() + 1) * norm
        weight *= mask
        weight *= 0.5
        weight += 0.5

        if data.weight is None:
            data.weight = weight
        else:
            data.weight = torch.cat([weight, data.weight.unsqueeze(1)], dim=1)

        return data
