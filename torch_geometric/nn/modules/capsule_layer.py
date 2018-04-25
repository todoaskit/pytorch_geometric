import torch
from torch.nn import Module, Parameter
from torch.nn import Linear

from .utils.inits import normal
from .utils.repr import repr
from .utils.repeat import repeat_to
from ..functional.capsule_layer import capsule_layer, capsule_layer_dense, \
    capsule_layer_em, capsule_layer_dense_unit, capsule_layer_unit, \
    capsule_layer_dense_matrix, capsule_layer_matrix, \
    capsule_layer_dense_matrix_quat, capsule_layer_matrix_quat




class CapsLayer(Module):
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
                 dim,
                 locally_dense=False,
                 em_routing=False):

        super(CapsLayer, self).__init__()

        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.iterations = iterations
        self.dim = dim
        self.locally_dense = locally_dense
        self.em_routing = em_routing
        if locally_dense:
            self.W_matrices = Parameter(torch.randn(num_caps_out, num_caps_in,
                                                    channels_in, channels_out))
        else:
            self.W_matrices = Parameter(torch.randn(num_caps_out, num_caps_in,
                                                channels_in, channels_out))
        self.bias = Parameter(torch.randn(num_caps_out))

        if em_routing:
            self.beta_v = Parameter(torch.randn(1))
            self.beta_a = Parameter(torch.randn(1))
        else:
            self.bias = Parameter(torch.randn(num_caps_out))
            self.beta = Parameter(torch.randn(num_caps_out))
            self.alpha = Parameter(torch.randn(num_caps_out))

    def forward(self, data):
        if self.locally_dense:
            if self.em_routing:
                return capsule_layer_em(data, self.W_matrices,
                                        self.beta_v, self.beta_a,
                                     self.num_caps_in,
                                     self.num_caps_out,
                                     self.channels_in,
                                     self.channels_out,
                                     self.iterations, self.dim)
            else:
                return capsule_layer_dense_matrix(data, self.W_matrices, self.bias,
                                 self.beta, self.alpha,
                                 self.num_caps_in,
                                 self.num_caps_out,
                                 self.channels_in,
                                 self.channels_out,
                                 self.iterations, self.dim)
        else:
            return capsule_layer_matrix(data, self.W_matrices, self.bias,
                                 self.beta, self.alpha,
                         self.num_caps_in,
                                self.num_caps_out,
                                self.channels_in,
                                self.channels_out,
                                self.iterations, self.dim)

    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])


class CapsLayerQuat(Module):
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
                 dim,
                 locally_dense=False,
                 em_routing=False):

        super(CapsLayerQuat, self).__init__()

        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.iterations = iterations
        self.dim = dim
        self.locally_dense = locally_dense
        self.em_routing = em_routing
        self.quats = Parameter(torch.randn(num_caps_out, num_caps_in, 4))
        self.scale = Parameter(torch.randn(num_caps_out, num_caps_in, 1))
        self.trans = Parameter(torch.randn(num_caps_out, num_caps_in, 3))
        self.bias = Parameter(torch.randn(num_caps_out))


        self.bias = Parameter(torch.randn(num_caps_out))
        self.beta = Parameter(torch.randn(num_caps_out))
        self.alpha = Parameter(torch.randn(num_caps_out))

    def forward(self, data):
        if self.locally_dense:
            return capsule_layer_dense_matrix_quat(data, self.quats, self.scale, self.trans, self.bias,
                             self.beta, self.alpha,
                             self.num_caps_in,
                             self.num_caps_out,
                             self.channels_in,
                             self.channels_out,
                             self.iterations, self.dim)
        else:
            return capsule_layer_matrix_quat(data, self.quats, self.scale, self.trans, self.bias,
                                 self.beta, self.alpha,
                         self.num_caps_in,
                                self.num_caps_out,
                                self.channels_in,
                                self.channels_out,
                                self.iterations, self.dim)

    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])
