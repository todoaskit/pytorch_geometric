from torch_cluster import sparse_grid_cluster, dense_grid_cluster
import torch
from torch.autograd import Variable
from ...datasets.dataset import Data
from ..modules.spline_conv import repeat_to
from torch_scatter import scatter_add, scatter_mean, scatter_add_, scatter_mean_
import torch.nn.functional as F
from torch_unique import unique
import numpy as np

def normalize(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    return tensor / torch.sqrt(squared_norm)

def capsule_layer(data, W_matrices, bias, beta, alpha,
                         num_caps_in, num_caps_out, channels_in, channels_out,
                         iterations, dim):

    values = data.adj['values']
    row, col = data.adj['indices']
    input_e = data.input[col]
    pose_vectors_e = data.pose_vectors[col]
    pos_e = data.pos[col]

    assert pose_vectors_e.size(0) == input_e.size(0) == pos_e.size(0)

    try:
        pose_vectors_e = pose_vectors_e.view(pose_vectors_e.size(0), num_caps_in,
                                     channels_in)
    except:
        raise ValueError

    local_positions = Variable(data.pos[col] - data.pos[row])

    pose_vectors_e = torch.cat([pose_vectors_e, Variable(local_positions.view(-1,1,dim).
                                                         expand(-1,num_caps_in,-1),
                                                         requires_grad=False)],
                               dim=2)

    #pose_pos = pose_vectors_e[:, :, 0:dim]
    #pose_pos = pose_pos + local_positions.view(-1, 1, dim).expand(-1,num_caps_in,-1)
    #pose_vectors_e = torch.cat([pose_pos, pose_vectors_e[:, :, dim:]], dim=2)

    # Routing
    #logits = Variable(torch.zeros(input_e.size(0), num_caps_out,
    #                              num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = (pose_vectors_e[:, None, :, None, :] @ \
                       W_matrices[None, :, :, :, :]).squeeze(3)


    #votes1 = votes[:, :, :, 0:dim]
    #votes1 = votes1 - local_positions.view(-1, 1, 1, dim)
    #votes = torch.cat([votes1, votes[:, :, :, dim:]], dim=3)

    # Reshape Cluster for scatter operations
    row_exp = Variable(row.view(-1, 1, 1).
                            expand(-1, num_caps_out,
                                   channels_out))
    row_exp2 = Variable(row.view(-1, 1, 1).
                       expand(-1, num_caps_out, -1))
    #weights = F.softmax(logits,dim=1)

    prev_act = input_e.view(-1, 1, num_caps_in, 1)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    zeros = Variable(torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
    zeros2 = Variable(torch.ones(data.pos.size(0), num_caps_out, 1).cuda())

    weights_sum = scatter_add_(zeros2, row_exp2, weights.sum(2))
    new_poses = scatter_add_(zeros, row_exp, (votes_exp * weights).
                             sum(2))/weights_sum
    new_poses.squeeze()

    # Compute activations as  variance of votes


    beta = beta.view(1,-1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, row_exp).\
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = torch.sqrt(torch.clamp(((votes_exp-new_poses_gath)**2).
                                              sum(dim=-1), min=0, max=50))
        #delta_logits = torch.abs(votes_exp - new_poses_gath).mean(dim=-1)
        weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3)
        weights = weights * prev_act

        # Sum in num_caps_in dimension and cluster
        zeros = Variable(
            torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
        zeros2 = Variable(
            torch.ones(data.pos.size(0), num_caps_out, 1).cuda())

        weights_sum = scatter_add_(zeros2, row_exp2, weights.sum(2))
        new_poses = scatter_add_(zeros, row_exp, (votes_exp * weights).
                                 sum(2)) / weights_sum
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, row_exp). \
        view(-1, num_caps_out, 1, channels_out)

    avg_dist = torch.sqrt(torch.clamp(((votes_exp - new_poses_gath) ** 2).
                                      mean(dim=-1), min=0, max=50)).mean(dim=-1)
    #avg_dist = torch.abs(votes_exp - new_poses_gath).mean(dim=-1).mean(dim=-1)
    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out).cuda())
    variance = scatter_mean_(zeros, Variable(row.view(-1, 1).expand(-1, num_caps_out)),
                            avg_dist)
    # print('variance:',variance.size())

    alpha = alpha.view(1,-1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha*variance)

    data.pose_vectors = new_poses.squeeze()

    if data.pose_vectors.size(0) != data.pos.size(0):
        print('ERROR caps:', data.pose_vectors.size(0), data.input.size(0),
              data.pos.size(0), data.index.size(0))
        #overhead = data.pose_vectors.size(0) - data.pos.size(0)
        #data.pos = data.pos[:overhead, :]
        #data.index = data.index[col<data.pose_vectors.size(0)]
        #print(overhead)
        #print(data.batch.size())
        #data.batch = data.batch[:overhead]
        #print('after:',data.batch.size(), data.pos.size(0), data.index.size(0))

    return data


def capsule_layer_dense(data, W_matrices, bias, beta, alpha,
                  num_caps_in, num_caps_out, channels_in, channels_out,
                  iterations, dim):

    pose_vectors = data.pose_vectors
    input = data.input

    assert pose_vectors.size(0) == input.size(0)

    try:
        pose_vectors = pose_vectors.view(pose_vectors.size(0),
                                             num_caps_in,
                                             channels_in)
    except:
        raise ValueError


    # Routing
    #logits = Variable(torch.zeros(input.size(0), num_caps_out,
   #                               num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = pose_vectors[:, None, :, None, :] @ \
            W_matrices[None, :, :, :, :]

    #weights = F.softmax(logits, dim=1)
    prev_act = input.view(-1, 1, num_caps_in, 1)
    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    new_poses = (votes_exp * weights).sum(2)/weights.sum(2)
    new_poses.squeeze()
    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    # print('variance:',variance.size())

    for it in range(1, iterations):
        #print(votes_exp.size())
        #print(new_poses.view(-1, num_caps_out, 1, channels_out).size())
        delta_logits = torch.sqrt(torch.clamp(((votes_exp -
                         new_poses.view(-1, num_caps_out, 1, channels_out))
                        ** 2).sum(dim=-1), min=0, max=50))
        #delta_logits = torch.abs(votes_exp -
        #                         new_poses.view(-1, num_caps_out, 1,
        #                                        channels_out)).mean(dim=-1)
        #logits = logits + (1 - delta_logits.unsqueeze(4))
        #weights = F.softmax(logits, dim=1)

        weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3)
        # print(transformed_pose.size())
        # Sum in num_caps_in dimension and cluster
        weights = weights * prev_act

        new_poses = (votes_exp * weights).sum(2)/weights.sum(2)
        new_poses.squeeze()


    avg_dist = torch.sqrt(torch.clamp(((votes_exp -
                 new_poses.view(-1, num_caps_out, 1, channels_out)) ** 2) \
        .mean(dim=-1),min=0, max=50)).mean(dim=-1)
    #avg_dist = torch.abs(votes_exp -
    #                     new_poses.view(-1, num_caps_out, 1, channels_out)).\
    #    mean(dim=-1).mean(dim=-1)
    data.pose_vectors = new_poses.squeeze()

    alpha = alpha.view(1,-1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha*avg_dist)
    #print(data.input.data[0].min(),data.input.data[0].max())

    return data


def capsule_layer_unit(data, W_matrices, bias, beta, alpha,
                  num_caps_in, num_caps_out, channels_in, channels_out,
                  iterations, dim):
    values = data.adj['values']
    row, col = data.adj['indices']
    input_e = data.input[col]
    pose_vectors_e = data.pose_vectors[col]
    pos_e = data.pos[col]

    assert pose_vectors_e.size(0) == input_e.size(0) == pos_e.size(0)

    try:
        pose_vectors_e = pose_vectors_e.view(pose_vectors_e.size(0),
                                             num_caps_in,
                                             channels_in)
    except:
        raise ValueError

    local_positions = Variable(data.pos[col] - data.pos[row])

    pose_vectors_e = torch.cat(
        [pose_vectors_e, Variable(local_positions.view(-1, 1, dim).
                                  expand(-1, num_caps_in, -1),
                                  requires_grad=False)],
        dim=2)

    # pose_pos = pose_vectors_e[:, :, 0:dim]
    # pose_pos = pose_pos + local_positions.view(-1, 1, dim).expand(-1,num_caps_in,-1)
    # pose_vectors_e = torch.cat([pose_pos, pose_vectors_e[:, :, dim:]], dim=2)

    # Routing
    # logits = Variable(torch.zeros(input_e.size(0), num_caps_out,
    #                              num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = (pose_vectors_e[:, None, :, None, :] @ \
             W_matrices[None, :, :, :, :]).squeeze(3)

    votes = normalize(votes)

    # Reshape Cluster for scatter operations
    row_exp = Variable(row.view(-1, 1, 1).
                       expand(-1, num_caps_out,
                              channels_out))

    prev_act = input_e.view(-1, 1, num_caps_in, 1)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())

    new_poses = normalize(scatter_add_(zeros, row_exp, (votes_exp * weights).
                             sum(2)))
    new_poses.squeeze()

    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, row_exp). \
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = (votes_exp * new_poses_gath).sum(dim=-1)
        # delta_logits = torch.abs(votes_exp - new_poses_gath).mean(dim=-1)
        weights = F.sigmoid(beta - alpha * delta_logits).unsqueeze(3)
        weights = weights * prev_act

        # Sum in num_caps_in dimension and cluster
        zeros = Variable(
            torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
        new_poses = normalize(scatter_add_(zeros, row_exp, (votes_exp * weights).
                                 sum(2)))
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, row_exp). \
        view(-1, num_caps_out, 1, channels_out)

    avg_dist = (votes_exp * new_poses_gath).sum(dim=-1).mean(dim=-1)

    # avg_dist = torch.abs(votes_exp - new_poses_gath).mean(dim=-1).mean(dim=-1)
    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out).cuda())
    variance = scatter_mean_(zeros,
                             Variable(row.view(-1, 1).expand(-1, num_caps_out)),
                             avg_dist)
    # print('variance:',variance.size())

    alpha = alpha.view(1, -1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha * variance)

    data.pose_vectors = new_poses.squeeze()

    if data.pose_vectors.size(0) != data.pos.size(0):
        print('ERROR caps:', data.pose_vectors.size(0), data.input.size(0),
              data.pos.size(0), data.index.size(0))
        # overhead = data.pose_vectors.size(0) - data.pos.size(0)
        # data.pos = data.pos[:overhead, :]
        # data.index = data.index[col<data.pose_vectors.size(0)]
        # print(overhead)
        # print(data.batch.size())
        # data.batch = data.batch[:overhead]
        # print('after:',data.batch.size(), data.pos.size(0), data.index.size(0))

    return data


def capsule_layer_dense_unit(data, W_matrices, bias, beta, alpha,
                        num_caps_in, num_caps_out, channels_in, channels_out,
                        iterations, dim):
    pose_vectors = data.pose_vectors
    input = data.input

    assert pose_vectors.size(0) == input.size(0)

    try:
        pose_vectors = pose_vectors.view(pose_vectors.size(0),
                                         num_caps_in,
                                         channels_in)
    except:
        raise ValueError

    # Routing
    # logits = Variable(torch.zeros(input.size(0), num_caps_out,
    #                               num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = pose_vectors[:, None, :, None, :] @ \
            W_matrices[None, :, :, :, :]
    votes = normalize(votes)

    # weights = F.softmax(logits, dim=1)
    prev_act = input.view(-1, 1, num_caps_in, 1)
    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    new_poses = normalize((votes_exp * weights).sum(2))
    new_poses.squeeze()
    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    # print('variance:',variance.size())

    for it in range(1, iterations):
        # print(votes_exp.size())
        # print(new_poses.view(-1, num_caps_out, 1, channels_out).size())
        delta_logits = (votes_exp * new_poses.view(-1, num_caps_out, 1,
                                                   channels_out)).sum(dim=-1)
        # delta_logits = torch.abs(votes_exp -
        #                         new_poses.view(-1, num_caps_out, 1,
        #                                        channels_out)).mean(dim=-1)
        # logits = logits + (1 - delta_logits.unsqueeze(4))
        # weights = F.softmax(logits, dim=1)

        weights = F.sigmoid(beta - alpha * delta_logits).unsqueeze(3)
        # print(transformed_pose.size())
        # Sum in num_caps_in dimension and cluster
        weights = weights * prev_act

        new_poses = normalize((votes_exp * weights).sum(2))
        new_poses.squeeze()

    avg_dist = (votes_exp - new_poses.view(-1, num_caps_out, 1,
                                           channels_out)).sum(dim=-1).mean(dim=-1)
    # avg_dist = torch.abs(votes_exp -
    #                     new_poses.view(-1, num_caps_out, 1, channels_out)).\
    #    mean(dim=-1).mean(dim=-1)
    data.pose_vectors = new_poses.squeeze()

    alpha = alpha.view(1, -1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha * avg_dist)
    # print(data.input.data[0].min(),data.input.data[0].max())

    return data


def capsule_layer_matrix(data, W_matrices, bias, beta, alpha,
                  num_caps_in, num_caps_out, channels_in, channels_out,
                  iterations, dim):
    values = data.adj['values']
    row, col = data.adj['indices']
    input_e = data.input[col]
    pose_matrices_e = data.pose_vectors[col]
    pos_e = data.pos[col]

    channels_in = 16
    channels_out = 16

    assert pose_matrices_e.size(0) == input_e.size(0) == pos_e.size(0)

    try:
        pose_matrices_e = pose_matrices_e.view(pose_matrices_e.size(0),
                                             num_caps_in,
                                             4, 4)
    except:
        raise ValueError

    local_positions = Variable(data.pos[col] - data.pos[row])
    #
    # ones = Variable(torch.ones(local_positions.size(0),1).cuda())
    #h_local_positions = torch.cat([local_positions,ones], dim=1)
    #transformed_positions = pose_matrices_e @ h_local_positions[:, None, :, None]
    #pose_matrices_e = torch.cat([pose_matrices_e[:, :, :, 0:3],
    #                             transformed_positions], dim=3)
    trans_matrices = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()). \
        expand(pose_matrices_e.size(0), -1, 4, 4)
    trans_matrices[:, :, 0:3, 3] = trans_matrices[:, :, 0:3, 3] - \
                                   local_positions.view(-1, 1, 3)
    pose_matrices_e = pose_matrices_e @ trans_matrices
    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = pose_matrices_e[:, None, :, :, :] @ W_matrices[None, :, :, :, :]


    row_exp = Variable(row.view(-1, 1, 1).
                       expand(-1, num_caps_out,
                              channels_out))
    row_exp2 = Variable(row.view(-1, 1, 1).
                        expand(-1, num_caps_out, -1))
    # weights = F.softmax(logits,dim=1)

    prev_act = input_e.view(-1, 1, num_caps_in, 1)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
    zeros2 = Variable(torch.ones(data.pos.size(0), num_caps_out, 1).cuda())

    weights_sum = scatter_add_(zeros2, row_exp2, weights.sum(2))
    new_poses = scatter_add_(zeros, row_exp, (votes_exp * weights).
                             sum(2)) / weights_sum
    new_poses.squeeze()

    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, row_exp). \
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = torch.sqrt(
            torch.clamp(((votes_exp - new_poses_gath) ** 2).
                        sum(dim=-1), min=0, max=50))
        # delta_logits = torch.abs(votes_exp - new_poses_gath).mean(dim=-1)
        weights = F.sigmoid(beta - alpha * delta_logits).unsqueeze(3)
        weights = weights * prev_act

        # Sum in num_caps_in dimension and cluster
        zeros = Variable(
            torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
        zeros2 = Variable(
            torch.ones(data.pos.size(0), num_caps_out, 1).cuda())

        weights_sum = scatter_add_(zeros2, row_exp2, weights.sum(2))
        new_poses = scatter_add_(zeros, row_exp, (votes_exp * weights).
                                 sum(2)) / weights_sum
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, row_exp). \
        view(-1, num_caps_out, 1, channels_out)

    avg_dist = torch.sqrt(torch.clamp(((votes_exp - new_poses_gath) ** 2).
                                      mean(dim=-1), min=0, max=50)).mean(dim=-1)
    # avg_dist = torch.abs(votes_exp - new_poses_gath).mean(dim=-1).mean(dim=-1)
    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out).cuda())
    variance = scatter_mean_(zeros,
                             Variable(row.view(-1, 1).expand(-1, num_caps_out)),
                             avg_dist)
    # print('variance:',variance.size())

    alpha = alpha.view(1, -1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha * variance)

    data.pose_vectors = new_poses.squeeze()

    return data


def capsule_layer_dense_matrix(data, W_matrices, bias, beta, alpha,
                        num_caps_in, num_caps_out, channels_in, channels_out,
                        iterations, dim):
    pose_matrices = data.pose_vectors
    input = data.input

    assert pose_matrices.size(0) == input.size(0)

    try:
        pose_matrices = pose_matrices.view(pose_matrices.size(0),
                                         num_caps_in,
                                         4,4)
    except:
        raise ValueError

    channels_in = 16
    channels_out = 16

    # Routing
    # logits = Variable(torch.zeros(input.size(0), num_caps_out,
    #                               num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = pose_matrices[:, None, :, :, :] @ W_matrices[None, :, :, :, :]



    prev_act = input.view(-1, 1, num_caps_in, 1)
    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    new_poses = (votes_exp * weights).sum(2) / weights.sum(2)
    new_poses.squeeze()
    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    # print('variance:',variance.size())

    for it in range(1, iterations):
        #print(votes_exp.size())
        #print(new_poses.view(-1, num_caps_out, 1, channels_out).size())
        delta_logits = torch.sqrt(torch.clamp(((votes_exp -
                         new_poses.view(-1, num_caps_out, 1, channels_out))
                        ** 2).sum(dim=-1), min=0, max=50))
        #delta_logits = torch.abs(votes_exp -
        #                         new_poses.view(-1, num_caps_out, 1,
        #                                        channels_out)).mean(dim=-1)
        #logits = logits + (1 - delta_logits.unsqueeze(4))
        #weights = F.softmax(logits, dim=1)

        weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3)
        # print(transformed_pose.size())
        # Sum in num_caps_in dimension and cluster
        weights = weights * prev_act

        new_poses = (votes_exp * weights).sum(2)/weights.sum(2)
        new_poses.squeeze()


    avg_dist = torch.sqrt(torch.clamp(((votes_exp -
                 new_poses.view(-1, num_caps_out, 1, channels_out)) ** 2) \
        .mean(dim=-1),min=0, max=50)).mean(dim=-1)
    #avg_dist = torch.abs(votes_exp -
    #                     new_poses.view(-1, num_caps_out, 1, channels_out)).\
    #    mean(dim=-1).mean(dim=-1)
    data.pose_vectors = new_poses.squeeze()

    alpha = alpha.view(1,-1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha*avg_dist)
    #print(data.input.data[0].min(),data.input.data[0].max())

    return data


def build_W(quats, scales, trans):
    # Build rotation matrices from quaternions
    quats = normalize(quats)
    pow = (quats ** 2).view(quats.size())
    ri = (quats[:, :, 0] * quats[:, :, 1]).unsqueeze(2)
    rj = (quats[:, :, 0] * quats[:, :, 2]).unsqueeze(2)
    rk = (quats[:, :, 0] * quats[:, :, 3]).unsqueeze(2)
    ij = (quats[:, :, 1] * quats[:, :, 2]).unsqueeze(2)
    ik = (quats[:, :, 1] * quats[:, :, 3]).unsqueeze(2)
    jk = (quats[:, :, 2] * quats[:, :, 3]).unsqueeze(2)

    zeros = Variable(torch.zeros(quats.size(0), quats.size(1), 1).cuda())

    R = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()) \
        + 2 * torch.cat(
        [(-pow[:, :, 2] - pow[:, :, 3]).unsqueeze(2), ij - rk, ik + rj, zeros,
         ij + rk, (-pow[:, :, 1] - pow[:, :, 3]).unsqueeze(2), jk - ri, zeros,
         ik - rj, jk + ri, (-pow[:, :, 1] - pow[:, :, 2]).unsqueeze(2), zeros,
         zeros, zeros, zeros, zeros], dim=2).view(quats.size(0),
                                                  quats.size(1),
                                                  4, 4)

    S = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()) * scales.unsqueeze(3)

    T = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()).\
        expand(trans.size(0), trans.size(1), -1, -1)
    T[:, :, 0:3, 3] = T[:, :, 0:3, 3] + trans
    return T @ S @ R


def capsule_layer_matrix_quat(data, quat, scale, trans, bias, beta, alpha,
                  num_caps_in, num_caps_out, channels_in, channels_out,
                  iterations, dim):
    values = data.adj['values']
    row, col = data.adj['indices']
    input_e = data.input[col]
    pose_matrices_e = data.pose_vectors[col]
    pos_e = data.pos[col]

    channels_in = 16
    channels_out = 16

    assert pose_matrices_e.size(0) == input_e.size(0) == pos_e.size(0)

    try:
        pose_matrices_e = pose_matrices_e.view(pose_matrices_e.size(0),
                                             num_caps_in,
                                             4, 4)
    except:
        raise ValueError

    local_positions = Variable(data.pos[col] - data.pos[row])

    trans_matrices = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()). \
        expand(pose_matrices_e.size(0), -1, 4, 4)
    trans_matrices[:, :, 0:3, 3] = trans_matrices[:, :, 0:3, 3] - \
                                      local_positions.view(-1, 1, 3)
    pose_matrices_e = trans_matrices @ pose_matrices_e

    #ones = Variable(torch.ones(local_positions.size(0),1).cuda())
    #h_local_positions = torch.cat([local_positions,ones], dim=1)
    #transformed_positions = pose_matrices_e @ h_local_positions[:, None, :, None]
    #pose_matrices_e = torch.cat([pose_matrices_e[:, :, :, 0:3],
    #                             transformed_positions], dim=3)
    #pose_matrices_e[:, :, 0:3, 3] = pose_matrices_e[:, :, 0:3, 3] + \
    #                              local_positions.view(-1, 1, 3)
    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]
    W_matrices = build_W(quat, scale, trans)
    votes = pose_matrices_e[:, None, :, :, :] @ W_matrices[None, :, :, :, :]

    #trans_matrices = Variable(torch.eye(4).view(1,1,1,4,4).cuda()).\
    #    expand(votes.size(0), -1, -1, 4, 4)
    #trans_matrices[:, :, :, 0:3, 3] = trans_matrices[:, :, :, 0:3, 3] + \
    #                               local_positions.view(-1, 1, 1, 3)
    #votes = trans_matrices @ votes

    row_exp = Variable(row.view(-1, 1, 1).
                       expand(-1, num_caps_out,
                              channels_out))
    row_exp2 = Variable(row.view(-1, 1, 1).
                        expand(-1, num_caps_out, -1))
    # weights = F.softmax(logits,dim=1)

    prev_act = input_e.view(-1, 1, num_caps_in, 1)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
    zeros2 = Variable(torch.ones(data.pos.size(0), num_caps_out, 1).cuda())

    weights_sum = scatter_add_(zeros2, row_exp2, weights.sum(2))
    new_poses = scatter_add_(zeros, row_exp, (votes_exp * weights).
                             sum(2)) / weights_sum
    new_poses.squeeze()

    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, row_exp). \
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = torch.sqrt(
            torch.clamp(((votes_exp - new_poses_gath) ** 2).
                        sum(dim=-1), min=0, max=50))
        # delta_logits = torch.abs(votes_exp - new_poses_gath).mean(dim=-1)
        weights = F.sigmoid(beta - alpha * delta_logits).unsqueeze(3)
        weights = weights * prev_act

        # Sum in num_caps_in dimension and cluster
        zeros = Variable(
            torch.zeros(data.pos.size(0), num_caps_out, channels_out).cuda())
        zeros2 = Variable(
            torch.ones(data.pos.size(0), num_caps_out, 1).cuda())

        weights_sum = scatter_add_(zeros2, row_exp2, weights.sum(2))
        new_poses = scatter_add_(zeros, row_exp, (votes_exp * weights).
                                 sum(2)) / weights_sum
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, row_exp). \
        view(-1, num_caps_out, 1, channels_out)

    avg_dist = torch.sqrt(torch.clamp(((votes_exp - new_poses_gath) ** 2).
                                      mean(dim=-1), min=0, max=50)).mean(dim=-1)
    # avg_dist = torch.abs(votes_exp - new_poses_gath).mean(dim=-1).mean(dim=-1)
    zeros = Variable(
        torch.zeros(data.pos.size(0), num_caps_out).cuda())
    variance = scatter_mean_(zeros,
                             Variable(row.view(-1, 1).expand(-1, num_caps_out)),
                             avg_dist)
    # print('variance:',variance.size())

    alpha = alpha.view(1, -1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha * variance)

    data.pose_vectors = new_poses.squeeze()

    return data


def capsule_layer_dense_matrix_quat(data, quat, scale, trans, bias, beta, alpha,
                        num_caps_in, num_caps_out, channels_in, channels_out,
                        iterations, dim):
    pose_matrices = data.pose_vectors
    input = data.input

    assert pose_matrices.size(0) == input.size(0)

    try:
        pose_matrices = pose_matrices.view(pose_matrices.size(0),
                                         num_caps_in,
                                         4,4)
    except:
        raise ValueError

    channels_in = 16
    channels_out = 16

    # Routing
    # logits = Variable(torch.zeros(input.size(0), num_caps_out,
    #                               num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    W_matrices = build_W(quat, scale, trans)
    votes = pose_matrices[:, None, :, :, :] @ W_matrices[None, :, :, :, :]



    prev_act = input.view(-1, 1, num_caps_in, 1)
    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    new_poses = (votes_exp * weights).sum(2) / weights.sum(2)
    new_poses.squeeze()
    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    # print('variance:',variance.size())

    for it in range(1, iterations):
        #print(votes_exp.size())
        #print(new_poses.view(-1, num_caps_out, 1, channels_out).size())
        delta_logits = torch.sqrt(torch.clamp(((votes_exp -
                         new_poses.view(-1, num_caps_out, 1, channels_out))
                        ** 2).sum(dim=-1), min=0, max=50))
        #delta_logits = torch.abs(votes_exp -
        #                         new_poses.view(-1, num_caps_out, 1,
        #                                        channels_out)).mean(dim=-1)
        #logits = logits + (1 - delta_logits.unsqueeze(4))
        #weights = F.softmax(logits, dim=1)

        weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3)
        # print(transformed_pose.size())
        # Sum in num_caps_in dimension and cluster
        weights = weights * prev_act

        new_poses = (votes_exp * weights).sum(2)/weights.sum(2)
        new_poses.squeeze()


    avg_dist = torch.sqrt(torch.clamp(((votes_exp -
                 new_poses.view(-1, num_caps_out, 1, channels_out)) ** 2) \
        .mean(dim=-1),min=0, max=50)).mean(dim=-1)
    #avg_dist = torch.abs(votes_exp -
    #                     new_poses.view(-1, num_caps_out, 1, channels_out)).\
    #    mean(dim=-1).mean(dim=-1)
    data.pose_vectors = new_poses.squeeze()

    alpha = alpha.view(1,-1)
    data.input = F.sigmoid(beta.view(1, -1) - alpha*avg_dist)
    #print(data.input.data[0].min(),data.input.data[0].max())

    return data




# Notation like in Hinton et al. EM-routing paper:
# R: [batch, caps_out, caps_in] - Agreement
# a_old: [batch, caps_in], Activations of layer l
# a_new: [batch, caps_out], Activations of layer l+1
# V: [batch, caps_out, caps_in, channels_out] - Votes
# M: [batch, caps_out, channels_out] - Means of Gaussians
# S: [batch, caps_out, channels_out] - Variances of Gaussians
# P: [batch, caps_out, caps_in] - Gaussian density values of votes



def m_step(R, a_old, V, beta_v, beta_a, inverse_temperature):
    R = R*a_old.unsqueeze(1)
    R_sum = R.unsqueeze(3).sum(2)
    M = (V*R.unsqueeze(3)).sum(2)/R_sum
    S = (((V-M.unsqueeze(2))**2)*R.unsqueeze(3)).sum(2)/R_sum

    #print('S:',S.data[0].min(),S.data[0].max())
    cost = (beta_v+torch.log(torch.sqrt(S)))*R_sum

    cost = cost.sum(-1, keepdim=True)
    cost_mean = cost.mean(-2, keepdim=True)
    cost_stdv = torch.sqrt(((cost - cost_mean)**2).mean(-2,keepdim=True))

    activations_cost = beta_a + (cost_mean - cost) / (
            cost_stdv + 0.0001)

    a_new = F.sigmoid(inverse_temperature*(beta_a - activations_cost)).squeeze()
    return M, S, a_new

def e_step(M, S, a_new, V):
    M = M.unsqueeze(2)  # [batch, caps_out, 1, channels_out]
    S = S.unsqueeze(2)  # [batch, caps_out, 1, channels_out]
    a_new = a_new.unsqueeze(2)  # [batch, caps_out, 1]

    P_1 = -(((V-M)**2)/(2*S)).sum(-1)
    P_2 = torch.log(torch.sqrt(S + 0.0001)).sum(-1)
    P = P_1 + P_2

    #P = torch.exp(-(((V-M)**2)/(2*S)).sum(-1))/torch.sqrt(2*np.pi*S.prod(-1))

    R_pre = torch.log(a_new + 0.0001) + P

    R = F.softmax(R_pre, dim=1)
    #R = (a_new*P)/(a_new*P).sum(dim=1, keepdim=True)

    return R


def capsule_layer_em(data, W_matrices, beta_v, beta_a,
                  num_caps_in, num_caps_out, channels_in, channels_out,
                  iterations, dim):

    pose_vectors = data.pose_vectors
    a_old = data.input

    assert pose_vectors.size(0) == a_old.size(0)

    try:
        pose_vectors = pose_vectors.view(pose_vectors.size(0),
                                             num_caps_in,
                                             channels_in)
    except:
        raise ValueError


    # Routing
    R = Variable(torch.zeros(a_old.size(0), num_caps_out,
                                  num_caps_in).cuda()).fill_(1/num_caps_out)

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, 1, channels_out]
    V = torch.matmul(pose_vectors[:, None, :, None, :],
                  W_matrices[None, :, :, :, :])

    V = V.squeeze(3)
    it_max = min(iterations,3.0)
    it_min = 1.0

    for it in range(iterations):
        inverse_temperature = it_min + (it_max - it_min) * it / max(1.0,
                                                            iterations - 1.0)
        M, S, a_new = m_step(R, a_old, V, beta_v, beta_a, inverse_temperature)
        #print('a_new:',a_new.data[0].min(), a_new.data[0].max())
        if it != iterations-1:
            R = e_step(M, S, a_new, V)

    data.pose_vectors = M.squeeze()
    data.input = a_new

    return data
