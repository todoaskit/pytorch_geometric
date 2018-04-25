import torch
from torch.autograd import Variable
import torch.nn.functional as F


def normalize(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    return tensor / (torch.sqrt(squared_norm)+0.0001)

def build_R(quats):
    # Build rotation matrices from quaternions
    pow = (quats ** 2).view(quats.size())
    ri = (quats[:, :, 0] * quats[:, :, 1]).unsqueeze(2)
    rj = (quats[:, :, 0] * quats[:, :, 2]).unsqueeze(2)
    rk = (quats[:, :, 0] * quats[:, :, 3]).unsqueeze(2)
    ij = (quats[:, :, 1] * quats[:, :, 2]).unsqueeze(2)
    ik = (quats[:, :, 1] * quats[:, :, 3]).unsqueeze(2)
    jk = (quats[:, :, 2] * quats[:, :, 3]).unsqueeze(2)

    R = Variable(torch.eye(3).view(1, 1, 3, 3).cuda()) \
        + 2 * torch.cat(
        [(-pow[:, :, 2] - pow[:, :, 3]).unsqueeze(2), ij - rk, ik + rj,
         ij + rk, (-pow[:, :, 1] - pow[:, :, 3]).unsqueeze(2), jk - ri,
         ik - rj, jk + ri, (-pow[:, :, 1] - pow[:, :, 2]).unsqueeze(2)],
        dim=2).view(quats.size(0), quats.size(1), 3, 3)

    return R

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


def quat_capsule_layer(data, quaternions, alpha, beta, spline_convs,
                        num_caps_in, num_caps_out, iterations, dim):
    pose_quats = data.pose_vectors
    input = data.input

    assert pose_quats.size(0) == input.size(0)

    try:
        pose_quats = pose_quats.view(pose_quats.size(0),
                                         num_caps_in,
                                         4)
    except:
        raise ValueError

    # Create votes by quaternion multiplication
    quaternions = normalize(quaternions)
    votes = b_quat_mul(pose_quats[:, None, :, :], quaternions[None, :, :, :])

    prev_act = input.view(-1, 1, num_caps_in, 1)
    weights = prev_act.expand(-1, num_caps_out, -1, -1)

    # Sum in num_caps_in dimension weighted by previous activation
    votes_exp = votes.view(-1, num_caps_out, num_caps_in, 4)
    new_quats = normalize((votes_exp * weights).sum(2)/weights.sum(2))

    beta = beta.view(1, -1, 1)
    alpha = alpha.view(1, -1, 1)

    for it in range(1, iterations):
        delta_logits = (votes_exp * new_quats.view(-1, num_caps_out, 1,
                                                   4)).sum(dim=-1)

        #weights = F.sigmoid(beta - alpha*delta_logits).unsqueeze(3) * prev_act
        #weights = F.sigmoid(delta_logits).unsqueeze(3) * prev_act
        weights = ((delta_logits).unsqueeze(3)/2+0.5) * prev_act
        # Sum in num_caps_in dimension weighted by new weight
        new_quats = normalize((votes_exp * weights).sum(2) / weights.sum(2))


    data.pose_vectors = new_quats.squeeze()
    beta = beta.view(1, -1)
    alpha = alpha.view(1, -1)

    if spline_convs is not None:
        # Perform convolution on quat-rotated inputs
        conv_results = []
        adj = {}
        adj['indices'] = data.adj['indices']
        adj['size'] = data.adj['size']
        row, _ = adj['indices']
        R_mats = build_R(new_quats)
        R_mats = torch.gather(R_mats, 0, Variable(row).view(-1, 1, 1, 1).
                              expand(-1, num_caps_out, 3, 3))
        for out_caps, conv in enumerate(spline_convs):
            adj['values'] = data.adj['values'] - 0.5
            adj['values'] = (adj['values'][:, None, :] @
                             R_mats.data[:, out_caps, :, :]).squeeze(1)
            adj['values'] = adj['values'] + 0.5
            conv_results.append(conv(adj, data.input))

        data.input = F.sigmoid(torch.cat(conv_results, dim=1))
    else:
        # Use agreement as activation
        agreement = (votes_exp * new_quats.view(-1, num_caps_out, 1,
                                                   4)).sum(-1).mean(-1)

        data.input = (agreement/2) + 0.5
        #data.input = F.sigmoid(beta - alpha * agreement)

    return data
