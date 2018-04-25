import torch
from torch.nn import Module, Parameter
from torch.nn import Linear

from .utils.inits import normal
from .utils.repr import repr
from .utils.repeat import repeat_to
from ..functional.pool.pooling_by_agreement import pooling_by_agreement, \
    pooling_by_agreement_unit, pooling_by_agreement_matrix, \
    pooling_by_agreement_quat




class PoolingByAgreement(Module):
    """Spline-based Convolutional Operator :math:`(f \star g)(i) =
    1/\mathcal{N}(i) \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(j)}
    f_l(j) \cdot g_l(u(i, j))`, where :math:`g_l` is a kernel function defined
    over the weighted B-Spline tensor product basis for a single input feature
    map. (Fey et al: SplineCNN: Fast Geometric Deep Learning with Continuous
    B-Spline Kernels, CVPR 2018, https://arxiv.org/abs/1711.08920)

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): Whether to use open or
            closed B-spline bases. (default :obj:`True`)
        degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 num_caps_in,
                 num_caps_out,
                 channels_in,
                 channels_out,
                 iterations,
                 dim):

        super(PoolingByAgreement, self).__init__()

        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.iterations = iterations
        self.dim = dim
        self.W_matrices = Parameter(torch.randn(num_caps_out, num_caps_in,
                                                channels_in, channels_out))
        self.bias = Parameter(torch.randn(num_caps_out))
        self.beta = Parameter(torch.randn(num_caps_out))
        self.alpha = Parameter(torch.randn(num_caps_out))

    def forward(self, data, size, start=None, transform=None):
        return pooling_by_agreement_matrix(data, self.W_matrices, self.bias, self.beta, self.alpha,
                                    size, start,
                                    transform, self.num_caps_in,
                                    self.num_caps_out,
                                    self.channels_in,
                                    self.channels_out,
                                    self.iterations, self.dim)

    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])


class PoolingByQuatAgreement(Module):
    """Spline-based Convolutional Operator :math:`(f \star g)(i) =
    1/\mathcal{N}(i) \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(j)}
    f_l(j) \cdot g_l(u(i, j))`, where :math:`g_l` is a kernel function defined
    over the weighted B-Spline tensor product basis for a single input feature
    map. (Fey et al: SplineCNN: Fast Geometric Deep Learning with Continuous
    B-Spline Kernels, CVPR 2018, https://arxiv.org/abs/1711.08920)

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int or [int]): Size of the convolving kernel.
        is_open_spline (bool or [bool], optional): Whether to use open or
            closed B-spline bases. (default :obj:`True`)
        degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 num_caps_in,
                 num_caps_out,
                 iterations=3,
                 dim=3):

        super(PoolingByQuatAgreement, self).__init__()

        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out
        self.iterations = iterations
        self.dim = dim
        self.quaternions = Parameter(torch.randn(num_caps_out, num_caps_in, 4))
        self.beta = Parameter(torch.randn(num_caps_out))
        self.alpha = Parameter(torch.randn(num_caps_out))

    def forward(self, data, size, start=None, transform=None):
        return pooling_by_agreement_quat(data, self.quaternions, self.alpha,
                                         self.beta, size, start,
                                    transform, self.num_caps_in,
                                    self.num_caps_out,
                                    self.iterations, self.dim)

    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])