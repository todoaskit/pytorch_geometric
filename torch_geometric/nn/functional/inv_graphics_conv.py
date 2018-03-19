import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .spline_conv.spline_conv import spline_conv
from torch_scatter import scatter_mean
import numpy as np


def inv_graphics_conv(
        adj,  # Pytorch Tensor (!bp_to_adj) or Pytorch Variable (bp_to_adj)
        input,  # Pytorch Variable
        local_stn,
        conv):

    ones = Variable(torch.ones(input.size(0), 1).cuda())

    # Compute params for transformation matrices
    transform = F.elu(local_stn[0](adj, ones))
    transform = F.elu(local_stn[1](adj, transform))
    transform = F.elu(local_stn[2](transform))
    transform_params = np.pi * F.tanh(local_stn[3](transform))

    # Locally transform inputs

    values = adj['values']
    _, col = adj['indices']

    transform_params_edge_wise = transform_params[col]

    # Center to zero
    t_values = values - 0.5

    # Rotate X
    cos = torch.cos(transform_params_edge_wise[:, :3])
    sin = torch.sin(transform_params_edge_wise[:, :3])
    #print(cos.data.min(), cos.data.max(), sin.data.min(), sin.data.max())
    t_values = torch.cat([t_values[:, 0].unsqueeze(1),
                          (cos[:, 0] * t_values[:, 1] -
                           sin[:, 0] * t_values[:, 2]).unsqueeze(1),
                          (sin[:, 0] * t_values[:, 1] +
                           cos[:, 0] * t_values[:, 2]).unsqueeze(1)],
                         dim=1)

    # Rotate Y
    t_values = torch.cat([(cos[:, 1] * t_values[:, 0] -
                           sin[:, 1] * t_values[:, 2]).unsqueeze(1),
                          t_values[:, 1].unsqueeze(1),
                          (sin[:, 1] * t_values[:, 0] +
                           cos[:, 1] * t_values[:, 2]).unsqueeze(1)],
                         dim=1)

    # Rotate Z
    t_values = torch.cat([(cos[:, 2] * t_values[:, 0] -
                           sin[:, 2] * t_values[:, 1]).unsqueeze(1),
                          (sin[:, 2] * t_values[:, 0] +
                           cos[:, 2] * t_values[:, 1]).unsqueeze(1),
                          t_values[:, 2].unsqueeze(1)],
                         dim=1)

    # CovarianceMatrix
    C = torch.matmul(t_values.view(-1, 3, 1), t_values.view(-1, 1, 3))

    C = scatter_mean(Variable(col.view(-1, 1, 1).expand_as(C)), C)
    C_mean = torch.abs(C).mean(0).view(9)
    C_loss = C_mean[1:4].sum() + C_mean[5:8].sum() + \
             torch.clamp(C_mean[4] - C_mean[0], min=0) + \
             torch.clamp(C_mean[8] - C_mean[4], min=0)


    # Scaling
    #t_values = F.sigmoid(transform_params_edge_wise[:, 3:]) * t_values
    '''
    t_values = torch.matmul(transform_params_edge_wise.view(-1,3,3),
                            t_values.view(-1,3,1)).squeeze()
    '''


    # Back to [0,1]
    t_values = t_values + 0.5
    t_values = torch.clamp(t_values, min=0.0, max=1.0)

    transformed_adj = {'values': t_values.detach(), 'indices': adj['indices'],
                       'size': adj['size']}

    # Conv layer on transformed input
    output = conv(transformed_adj, input)

    rotation = torch.sin(transform_params[:, :3])
    #scaling_factors = F.sigmoid(transform_params[:, 3:])


    output = torch.cat([F.elu(output),
                        #scaling_factors.detach(),
                        rotation.detach()], dim=1)
    #output = torch.cat([F.elu(output), transform_params], dim=1)

    return output, C_loss
