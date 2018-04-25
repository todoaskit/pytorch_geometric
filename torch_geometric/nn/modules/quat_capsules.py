import torch
from torch.nn import Module, Parameter, ModuleList
from torch.nn import Linear

from .utils.inits import normal
from .utils.repr import repr
from .utils.repeat import repeat_to
from ..functional.quat_capsules import quat_capsule_layer
from .spline_conv import SplineConv



class QuatCapsuleLayer(Module):
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
                 dim=3,
                 use_conv=True):

        super(QuatCapsuleLayer, self).__init__()

        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out
        self.iterations = iterations
        self.dim = dim
        self.quaternions = Parameter(torch.randn(num_caps_out, num_caps_in, 4))
        self.beta = Parameter(torch.randn(num_caps_out))
        self.alpha = Parameter(torch.randn(num_caps_out))
        if use_conv:
            self.splineConvs = ModuleList([SplineConv(num_caps_in, 1, dim, 5)
                                for _ in range(num_caps_out)])
        else:
            self.splineConvs = None

    def forward(self, data):
            return quat_capsule_layer(data, self.quaternions, self.alpha,
                                      self.beta, self.splineConvs,
                                 self.num_caps_in,
                                 self.num_caps_out,
                                 self.iterations, self.dim)


    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])



