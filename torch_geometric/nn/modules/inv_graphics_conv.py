import torch
from torch.nn import Module, Parameter
from torch.nn import Linear
from torch.nn import ModuleList

from .utils.inits import uniform
from .utils.repr import repr
from .utils.repeat import repeat_to
from ..functional.inv_graphics_conv import inv_graphics_conv
from torch_geometric.nn.modules import SplineConv
from torchvision.transforms import Compose

from ..functional.spline_conv.spline_conv_gpu \
    import get_weighting_forward_kernel, get_weighting_backward_kernel
from ..functional.spline_conv.spline_conv_gpu \
    import get_basis_kernel, get_basis_backward_kernel


class InvGraphConv(Module):
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
                 in_features,
                 out_features,
                 dim,
                 kernel_size,
                 num_conv=1,
                 is_open_spline=True,
                 degree=1,
                 bias=True):

        super(InvGraphConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features


        self.local_stn1 = SplineConv(1, 64, dim=dim,
                                             kernel_size=kernel_size,
                                             is_open_spline=is_open_spline,
                                             degree=degree, bias=bias)


        self.local_stn2 = Linear(64, 64)

        self.local_stn3 = Linear(64, 64)

        self.local_stn4 = Linear(64, 64)

        self.local_stn5 = Linear(64, 3)

        self.local_stn6 = SplineConv(32, 6, dim=dim,
                                             kernel_size=kernel_size,
                                             is_open_spline=is_open_spline,
                                             degree=degree, bias=bias)

        self.conv_layers = [SplineConv(in_features, out_features, dim=dim,
                                             kernel_size=kernel_size,
                                             is_open_spline=is_open_spline,
                                             degree=degree, bias=bias)]
        for _ in range(1, num_conv):
            self.conv_layers.append(SplineConv(out_features, out_features,
                                               dim=dim, kernel_size=kernel_size,
                                               is_open_spline=is_open_spline,
                                               degree=degree, bias=bias))

        self.conv_layers = ModuleList(self.conv_layers)

    def reset_parameters(self):
        size = self.in_features * (self.K + 1)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, adj, input):
        return inv_graphics_conv(
            adj, input, [self.local_stn1, self.local_stn2, self.local_stn3,
                         self.local_stn4, self.local_stn5],
            self.conv_layers)

    def __repr__(self):
        return repr(self, ['kernel_size', 'is_open_spline', 'degree'])
