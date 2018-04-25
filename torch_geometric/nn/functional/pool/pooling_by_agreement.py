from torch_cluster import sparse_grid_cluster, dense_grid_cluster
import torch
from torch.autograd import Variable
from .pool import max_pool
from .pool import _pool
from ....datasets.dataset import Data
from ...modules.spline_conv import repeat_to
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
from torch_unique import unique
from .pool import avg_pool


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)

def normalize(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    return tensor / (torch.sqrt(squared_norm)+0.0001)

def pooling_by_agreement_traditional(data, W_matrices, size, start, transform,
                         num_caps_in, num_caps_out, channels_in, channels_out,
                         iterations, dim):

    pos_tensor = data.pos if torch.is_tensor(data.pos) else data.pos.data
    size = pos_tensor.new(repeat_to(size, data.pos.size(1)))

    assert dim == data.pos.size(1)

    try:
        data.input = data.input.view(data.input.size(0), num_caps_in,
                                     channels_in)
    except:
        raise ValueError


    if start is not None:
        start = pos_tensor.new(repeat_to(start, data.pos.size(1)))

    output = sparse_grid_cluster(pos_tensor, size, data.batch, start)
    cluster = output[0] if isinstance(output, tuple) else output
    batch = output[1] if isinstance(output, tuple) else None

    index, pos = _pool(data.index, data.pos, cluster, weight=None)
    cluster = Variable(cluster) if not torch.is_tensor(pos) else cluster
    local_positions = data.pos - torch.gather(pos, 0, cluster.view(-1,1).
                                                               expand(-1, dim))
    local_positions = local_positions.unsqueeze(1).expand(-1, num_caps_in, -1)

    if torch.is_tensor(local_positions):
        local_positions = Variable(local_positions, requires_grad=False)

    pose_vectors = torch.cat([data.input, local_positions], dim=2)

    # Routing
    logits = Variable(torch.zeros(data.input.size(0), num_caps_out,
                                  num_caps_in, 1, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    transformed_pose = pose_vectors[:, None, :, None, :] @ \
                       W_matrices[None, :, :, :, :]

    # Reshape Cluster for scatter operations
    cluster_exp = Variable(cluster.view(-1, 1, 1).
                            expand(-1, num_caps_out,
                                   channels_out))

    weights = F.softmax(logits,dim=1)

    # Sum in num_caps_in dimension and cluster
    new_poses = squash(scatter_add(cluster_exp, (transformed_pose * weights).
                                   sum(2).squeeze()))

    for it in range(1, iterations):
        pos_vec_exp = transformed_pose.view(-1, num_caps_out, num_caps_in, channels_out)

        new_poses_gath = torch.gather(new_poses, 0, cluster_exp).\
            view(-1, num_caps_out, 1, channels_out)
        #print(pos_vec_exp.size())
        #print(new_poses_gath.size())
        delta_logits = (pos_vec_exp*new_poses_gath).sum(dim=-1, keepdim=True)

        logits = logits + torch.abs(delta_logits.unsqueeze(4))

        weights = F.softmax(logits, dim=1)
        #print(weights.size())
        #print(transformed_pose.size())
        # Sum in num_caps_in dimension and cluster
        new_poses = squash(scatter_add(cluster_exp,
                                       (transformed_pose * weights).
                                       sum(2).squeeze()))

    data = Data(new_poses.squeeze(), pos, index, None, data.target, batch)

    if transform is not None:
        data = transform(data)

    return data, cluster


def pooling_by_agreement(data, W_matrices, bias, beta, alpha, size, start, transform,
                         num_caps_in, num_caps_out, channels_in, channels_out,
                         iterations, dim):

    pos_tensor = data.pos if torch.is_tensor(data.pos) else data.pos.data
    size = pos_tensor.new(repeat_to(size, data.pos.size(1)))

    assert dim == data.pos.size(1)
    assert data.pose_vectors.size(0) == data.input.size(0) == data.pos.size(0)

    try:
        data.pose_vectors = data.pose_vectors.view(data.input.size(0), num_caps_in,
                                     channels_in)
    except:
        raise ValueError


    if start is not None:
        start = pos_tensor.new(repeat_to(start, data.pos.size(1)))

    output = sparse_grid_cluster(pos_tensor, size, data.batch, start)
    cluster = output[0] if isinstance(output, tuple) else output
    batch = output[1] if isinstance(output, tuple) else None

    index, pos = _pool(data.index, data.pos, cluster, weight=None,
                       rm_self_loops=True)

    cluster = Variable(cluster) if not torch.is_tensor(pos) else cluster
    local_positions = data.pos - torch.gather(pos, 0, cluster.view(-1,1).
                                                               expand(-1, dim))
    local_positions = local_positions.unsqueeze(1).expand(-1, num_caps_in, -1)

    if torch.is_tensor(local_positions):
        local_positions = Variable(local_positions)

    pose_vectors = torch.cat([data.pose_vectors, local_positions], dim=2)
    #pose_vectors = data.pose_vectors

    #pose_pos = pose_vectors[:, :, 0:dim]
    #pose_pos = pose_pos + local_positions.view(-1, 1, dim).expand(-1,
    #                                                              num_caps_in,
    #                                                              -1)
    #pose_vectors = torch.cat([pose_pos, pose_vectors[:, :, dim:]], dim=2)

    #pose_vectors = data.pose_vectors

    # Routing
    #logits = Variable(torch.zeros(data.input.size(0), num_caps_out,
    #                              num_caps_in, 1).cuda())

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = (pose_vectors[:, None, :, None, :] @ \
                       W_matrices[None, :, :, :, :]).squeeze(3)

    #votes1 = votes[:, :, :, 0:dim]
    #votes1 = votes1 - local_positions.view(-1, 1, 1, dim)

    #votes = torch.cat([votes1, votes[:, :, :, dim:]], dim=3)
    # Reshape Cluster for scatter operations
    cluster_exp = Variable(cluster.view(-1, 1, 1).
                            expand(-1, num_caps_out,
                                   channels_out))
    cluster_exp2 = Variable(cluster.view(-1, 1, 1).
                           expand(-1, num_caps_out,-1))

    #weights = F.softmax(logits,dim=1)

    prev_act = data.input.view(-1, 1, num_caps_in, 1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)
    # Sum in num_caps_in dimension and cluster

    weights_sum = scatter_add(cluster_exp2, weights.sum(2))
    new_poses = scatter_add(cluster_exp, (votes_exp * weights).
                            sum(2)) / weights_sum
    new_poses.squeeze()

    # Compute activations as  variance of votes

    beta = beta.view(1,-1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, cluster_exp).\
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = torch.sqrt(torch.clamp(
            ((votes_exp - new_poses_gath) ** 2).sum(dim=-1),min=0,max=50))
        #delta_logits = torch.abs(votes_exp - new_poses_gath).mean(dim=-1)
        #logits = logits + (1 - delta_logits.unsqueeze(4))
        #weights = F.softmax(logits, dim=1)

        weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3)
        weights = weights * prev_act
        # Sum in num_caps_in dimension and cluster

        weights_sum = scatter_add(cluster_exp2, weights.sum(2))
        new_poses = scatter_add(cluster_exp, (votes_exp * weights).
                                 sum(2)) / weights_sum
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, cluster_exp). \
        view(-1, num_caps_out, 1, channels_out)
    avg_dist = torch.sqrt(torch.clamp(((votes_exp - new_poses_gath) ** 2).
                                      mean(dim=-1), min=0, max=50)).mean(dim=-1)
    #avg_dist = torch.abs(votes_exp - new_poses_gath).mean(dim=-1).mean(dim=-1)
    avg_dist = scatter_mean(
        Variable(cluster.view(-1, 1).expand(-1, num_caps_out)),
        avg_dist)

    alpha = alpha.view(1,-1)
    activations = F.sigmoid(beta.view(1, -1) - alpha*avg_dist)

    data = Data(activations, pos, index, None, data.target, batch)


    if transform is not None:
        data = transform(data)

    data.pose_vectors = new_poses

    if data.pose_vectors.size(0) != data.pos.size(0):
        print('ERROR pool:', data.pose_vectors.size(0), data.input.size(0),
              data.pos.size(0))
    #print(data.pose_vectors.size())
    #print(data.input.size())
    #print(data.input.data[0].min(),data.input.data[0].max())
    return data, cluster



def pooling_by_agreement_unit(data, W_matrices, bias, beta, alpha, size, start, transform,
                         num_caps_in, num_caps_out, channels_in, channels_out,
                         iterations, dim):

    pos_tensor = data.pos if torch.is_tensor(data.pos) else data.pos.data
    size = pos_tensor.new(repeat_to(size, data.pos.size(1)))

    assert dim == data.pos.size(1)
    assert data.pose_vectors.size(0) == data.input.size(0) == data.pos.size(0)

    try:
        data.pose_vectors = data.pose_vectors.view(data.input.size(0), num_caps_in,
                                     channels_in)
    except:
        raise ValueError


    if start is not None:
        start = pos_tensor.new(repeat_to(start, data.pos.size(1)))

    output = sparse_grid_cluster(pos_tensor, size, data.batch, start)
    cluster = output[0] if isinstance(output, tuple) else output
    batch = output[1] if isinstance(output, tuple) else None

    index, pos = _pool(data.index, data.pos, cluster, weight=None,
                       rm_self_loops=True)

    cluster = Variable(cluster) if not torch.is_tensor(pos) else cluster
    local_positions = data.pos - torch.gather(pos, 0, cluster.view(-1,1).
                                                               expand(-1, dim))
    local_positions = local_positions.unsqueeze(1).expand(-1, num_caps_in, -1)

    if torch.is_tensor(local_positions):
        local_positions = Variable(local_positions)

    pose_vectors = torch.cat([data.pose_vectors, local_positions], dim=2)

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = (pose_vectors[:, None, :, None, :] @ \
                       W_matrices[None, :, :, :, :]).squeeze(3)

    votes = normalize(votes)
    # Reshape Cluster for scatter operations
    cluster_exp = Variable(cluster.view(-1, 1, 1).
                            expand(-1, num_caps_out,
                                   channels_out))
    cluster_exp2 = Variable(cluster.view(-1, 1, 1).
                           expand(-1, num_caps_out,-1))

    #weights = F.softmax(logits,dim=1)

    prev_act = data.input.view(-1, 1, num_caps_in, 1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)
    # Sum in num_caps_in dimension and cluster

    new_poses = normalize(scatter_add(cluster_exp, (votes_exp * weights).
                            sum(2)))
    new_poses.squeeze()

    # Compute activations as  variance of votes

    beta = beta.view(1,-1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, cluster_exp).\
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = (votes_exp * new_poses_gath).sum(dim=-1)

        weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3)
        weights = weights * prev_act
        # Sum in num_caps_in dimension and cluster

        new_poses = normalize(scatter_add(cluster_exp, (votes_exp * weights).
                                 sum(2)))
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, cluster_exp). \
        view(-1, num_caps_out, 1, channels_out)
    avg_dist = (votes_exp * new_poses_gath).sum(dim=-1).mean(dim=-1)

    avg_dist = scatter_mean(
        Variable(cluster.view(-1, 1).expand(-1, num_caps_out)),
        avg_dist)

    alpha = alpha.view(1,-1)
    activations = F.sigmoid(beta.view(1, -1) - alpha*avg_dist)

    data = Data(activations, pos, index, None, data.target, batch)


    if transform is not None:
        data = transform(data)

    data.pose_vectors = new_poses

    if data.pose_vectors.size(0) != data.pos.size(0):
        print('ERROR pool:', data.pose_vectors.size(0), data.input.size(0),
              data.pos.size(0))
    #print(data.pose_vectors.size())
    #print(data.input.size())
    #print(data.input.data[0].min(),data.input.data[0].max())
    return data, cluster


def pooling_by_agreement_matrix(data, W_matrices, bias, beta, alpha, size, start, transform,
                         num_caps_in, num_caps_out, channels_in, channels_out,
                         iterations, dim):

    pos_tensor = data.pos if torch.is_tensor(data.pos) else data.pos.data
    size = pos_tensor.new(repeat_to(size, data.pos.size(1)))

    assert dim == data.pos.size(1)
    assert data.pose_vectors.size(0) == data.input.size(0) == data.pos.size(0)

    try:
        data.pose_vectors = data.pose_vectors.view(data.input.size(0), num_caps_in,
                                     4, 4)
    except:
        raise ValueError


    if start is not None:
        start = pos_tensor.new(repeat_to(start, data.pos.size(1)))

    output = sparse_grid_cluster(pos_tensor, size, data.batch, start)
    cluster = output[0] if isinstance(output, tuple) else output
    batch = output[1] if isinstance(output, tuple) else None

    index, pos = _pool(data.index, data.pos, cluster, weight=None,
                       rm_self_loops=True)

    cluster = Variable(cluster) if not torch.is_tensor(pos) else cluster
    local_positions = data.pos - torch.gather(pos, 0, cluster.view(-1,1).
                                                               expand(-1, dim))

    if torch.is_tensor(local_positions):
        local_positions = Variable(local_positions)

    channels_in = 16
    channels_out = 16
    pose_matrices = data.pose_vectors

    #ones = Variable(torch.ones(local_positions.size(0), 1).cuda())
    #h_local_positions = torch.cat([local_positions, ones], dim=1)
    #transformed_positions = pose_matrices @ h_local_positions[:, None, :, None]
    #pose_matrices_e = torch.cat([pose_matrices[:, :, :, 0:3],
    #                             transformed_positions], dim=3)

    trans_matrices = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()). \
        expand(pose_matrices.size(0), -1, 4, 4)
    trans_matrices[:, :, 0:3, 3] = trans_matrices[:, :, 0:3, 3] - \
                                   local_positions.view(-1, 1, 3)
    pose_matrices = pose_matrices @ trans_matrices

    # 5-dimensional matrix fu
    # pose vectors: [bs, num_caps_in, channels_in+dim]
    # -> [bs, 1, num_caps_in, 1, channels_in+dim]
    # times
    # W_matrices: [num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # -> [1, num_caps_out, num_caps_in, channels_in+dim, channels_out]
    # equals (first 3 dims batch_wise with broadcasting)
    # -> [bs, num_caps_out, num_caps_in, channels_out, 1]

    votes = pose_matrices[:, None, :, :, :] @ W_matrices[None, :, :, :, :]

    # votes = torch.cat([votes1, votes[:, :, :, dim:]], dim=3)
    # Reshape Cluster for scatter operations
    cluster_exp = Variable(cluster.view(-1, 1, 1).
                           expand(-1, num_caps_out,
                                  channels_out))
    cluster_exp2 = Variable(cluster.view(-1, 1, 1).
                            expand(-1, num_caps_out, -1))

    # weights = F.softmax(logits,dim=1)

    prev_act = data.input.view(-1, 1, num_caps_in, 1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, channels_out)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)
    # Sum in num_caps_in dimension and cluster

    weights_sum = scatter_add(cluster_exp2, weights.sum(2))
    new_poses = scatter_add(cluster_exp, (votes_exp * weights).
                            sum(2)) / weights_sum
    new_poses.squeeze()

    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        new_poses_gath = torch.gather(new_poses, 0, cluster_exp). \
            view(-1, num_caps_out, 1, channels_out)

        delta_logits = torch.sqrt(torch.clamp(
            ((votes_exp - new_poses_gath) ** 2).sum(dim=-1), min=0, max=50))


        weights = F.sigmoid(beta - alpha * delta_logits).unsqueeze(3)
        weights = weights * prev_act
        # Sum in num_caps_in dimension and cluster

        weights_sum = scatter_add(cluster_exp2, weights.sum(2))
        new_poses = scatter_add(cluster_exp, (votes_exp * weights).
                                sum(2)) / weights_sum
        new_poses.squeeze()

    new_poses_gath = torch.gather(new_poses, 0, cluster_exp). \
        view(-1, num_caps_out, 1, channels_out)
    avg_dist = torch.sqrt(torch.clamp(((votes_exp - new_poses_gath) ** 2).
                                      mean(dim=-1), min=0, max=50)).mean(dim=-1)

    avg_dist = scatter_mean(
        Variable(cluster.view(-1, 1).expand(-1, num_caps_out)),
        avg_dist)

    alpha = alpha.view(1, -1)
    activations = F.sigmoid(beta.view(1, -1) - alpha * avg_dist)

    data = Data(activations, pos, index, None, data.target, batch)

    if transform is not None:
        data = transform(data)

    data.pose_vectors = new_poses

    if data.pose_vectors.size(0) != data.pos.size(0):
        print('ERROR pool:', data.pose_vectors.size(0), data.input.size(0),
              data.pos.size(0))

    return data, cluster


def b_quat_mul(q1, q2):
    q1 = q1.expand(-1,q2.size()[1],-1,-1)
    q2 = q2.expand(q1.size()[0],-1,-1,-1)
    result_a = q1[:,:,:,0]*q2[:,:,:,0] - q1[:,:,:,1]*q2[:,:,:,1] - \
               q1[:,:,:,2]*q2[:,:,:,2] - q1[:,:,:,3]*q2[:,:,:,3]
    result_i = q1[:,:,:,0]*q2[:,:,:,1] + q1[:,:,:,1]*q2[:,:,:,0] - \
               q1[:,:,:,2]*q2[:,:,:,3] + q1[:,:,:,3]*q2[:,:,:,2]
    result_j = q1[:,:,:,0]*q2[:,:,:,2] + q1[:,:,:,1]*q2[:,:,:,3] + \
               q1[:,:,:,2]*q2[:,:,:,0] - q1[:,:,:,3]*q2[:,:,:,1]
    result_k = q1[:,:,:,0]*q2[:,:,:,3] - q1[:,:,:,1]*q2[:,:,:,2] + \
               q1[:,:,:,2]*q2[:,:,:,1] + q1[:,:,:,3]*q2[:,:,:,0]
    result = torch.stack([result_a, result_i, result_j, result_k], dim=3)

    return result


def build_W(quats, scales, trans):
    # Build rotation matrices from quaternions
    quats = normalize(quats)
    pow = (quats**2).view(quats.size())
    ri = (quats[:,:,0] * quats[:,:,1]).unsqueeze(2)
    rj = (quats[:,:,0] * quats[:,:,2]).unsqueeze(2)
    rk = (quats[:,:,0] * quats[:,:,3]).unsqueeze(2)
    ij = (quats[:,:,1] * quats[:,:,2]).unsqueeze(2)
    ik = (quats[:,:,1] * quats[:,:,3]).unsqueeze(2)
    jk = (quats[:,:,2] * quats[:,:,3]).unsqueeze(2)

    zeros = Variable(torch.zeros(quats.size(0), quats.size(1), 1).cuda())

    R = Variable(torch.eye(4).view(1,1,4,4).cuda()) \
        + 2*torch.cat([(-pow[:, :, 2]-pow[:, :, 3]).unsqueeze(2), ij-rk, ik+rj, zeros,
                         ij+rk, (-pow[:, :, 1]-pow[:, :, 3]).unsqueeze(2), jk-ri, zeros,
                         ik-rj, jk + ri, (-pow[:, :, 1]-pow[:, :, 2]).unsqueeze(2), zeros,
                         zeros, zeros, zeros, zeros], dim=2).view(quats.size(0),
                                                                 quats.size(1),
                                                                 4, 4)

    S = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()) * scales.unsqueeze(3)

    T = Variable(torch.eye(4).view(1, 1, 4, 4).cuda()). \
        expand(trans.size(0), trans.size(1), -1, -1)
    T[:, :, 0:3, 3] = T[:, :, 0:3, 3] + trans
    return T @ S @ R


def pooling_by_agreement_quat(data, quaternions, alpha, beta, size, start, transform,
                         num_caps_in, num_caps_out,
                         iterations, dim):

    pos_tensor = data.pos if torch.is_tensor(data.pos) else data.pos.data
    size = pos_tensor.new(repeat_to(size, data.pos.size(1)))

    assert dim == data.pos.size(1)
    assert data.pose_vectors.size(0) == data.input.size(0) == data.pos.size(0)

    try:
        data.pose_vectors = data.pose_vectors.view(data.input.size(0),
                                                   num_caps_in, 4)
    except:
        raise ValueError


    if start is not None:
        start = pos_tensor.new(repeat_to(start, data.pos.size(1)))

    output = sparse_grid_cluster(pos_tensor, size, data.batch, start)
    cluster = output[0] if isinstance(output, tuple) else output
    batch = output[1] if isinstance(output, tuple) else None

    index, pos = _pool(data.index, data.pos, cluster, weight=None,
                       rm_self_loops=True)

    cluster = Variable(cluster) if not torch.is_tensor(pos) else cluster

    pose_quats = data.pose_vectors

    quaternions = normalize(quaternions)
    #print('quaternions',quaternions.data.min(),quaternions.data.max())

    votes = b_quat_mul(pose_quats[:, None, :, :], quaternions[None, :, :, :])
    #print('votes',votes.data.min(),votes.data.max())
    # Reshape Cluster for scatter operations
    cluster_exp = Variable(cluster.view(-1, 1, 1).
                           expand(-1, num_caps_out,
                                  4))
    cluster_exp2 = Variable(cluster.view(-1, 1, 1).
                            expand(-1, num_caps_out, -1))

    # weights = F.softmax(logits,dim=1)

    prev_act = data.input.view(-1, 1, num_caps_in, 1)

    # Sum in num_caps_in dimension and cluster, weighted by softmax logits
    # and previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, 4)

    weights = prev_act.expand(-1, num_caps_out, -1, -1)
    # Sum in num_caps_in dimension and cluster

    weights_sum = scatter_add(cluster_exp2, weights.sum(2))
    new_quats = scatter_add(cluster_exp, (votes_exp * weights).
                            sum(2)) / weights_sum
    new_quats = normalize(new_quats)
    #print('new_quats',new_quats.data.min(),new_quats.data.max())

    # Compute activations as  variance of votes

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)
    #print(new_quats.size(), cluster_exp.size())
    for it in range(1, iterations):
        new_quats_gath = torch.gather(new_quats, 0, cluster_exp). \
            view(-1, num_caps_out, 1, 4)

        delta_logits = (votes_exp * new_quats_gath).sum(dim=-1)

        #weights = F.sigmoid(beta - alpha*delta_logits.unsqueeze(3)) * prev_act
        #weights = F.sigmoid(delta_logits.unsqueeze(3)) * prev_act
        weights = (delta_logits.unsqueeze(3)/2+0.5) * prev_act
        # Sum in num_caps_in dimension and cluster

        weights_sum = scatter_add(cluster_exp2, weights.sum(2))
        new_quats = scatter_add(cluster_exp, (votes_exp * weights).
                                sum(2)) / weights_sum
        new_quats = normalize(new_quats)
        #print('new_quats',new_quats.data.min(),new_quats.data.max())

    new_quats_gath = torch.gather(new_quats, 0, cluster_exp). \
        view(-1, num_caps_out, 1, 4)

    beta = beta.view(1, -1)
    alpha = alpha.view(1, -1)
    #act_weights = F.sigmoid((votes_exp * new_quats_gath).sum(-1, keepdim=True))
    act_weights = ((votes_exp * new_quats_gath).sum(-1, keepdim=True)/2)+0.5
    activations = scatter_add(cluster_exp2, (prev_act * act_weights).sum(2))

    act_weights_sum = scatter_add(cluster_exp2, act_weights.sum(2))
    activations = (activations / act_weights_sum).squeeze(2)
    #activations = F.sigmoid(beta-alpha*activations)
    data = Data(activations, pos, index, None, data.target, batch)

    if transform is not None:
        data = transform(data)

    data.pose_vectors = new_quats

    if data.pose_vectors.size(0) != data.pos.size(0):
        print('ERROR pool:', data.pose_vectors.size(0), data.input.size(0),
              data.pos.size(0))

    return data, cluster