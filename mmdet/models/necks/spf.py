# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
#from fairscale.nn.checkpoint import checkpoint_wrapper
from ..builder import NECKS


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@NECKS.register_module()
class SFP(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 use_p2=False,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 use_act_checkpoint=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='bilinear'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SFP, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.use_p2 = use_p2
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if self.use_p2:
            self.p2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels[0], self.in_channels[0] // 2, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.in_channels[0] // 2),
                nn.GELU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels[0] // 2, self.in_channels[0] // 4, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.in_channels[0] // 4),
                nn.GELU(),
                nn.Conv2d(self.in_channels[0] // 4, self.out_channels, kernel_size=1, bias=False),
                LayerNorm(self.out_channels),
                nn.GELU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.out_channels)
            )

        self.p3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels[0], self.in_channels[0] // 2, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.in_channels[0] // 2),
            nn.GELU(),
            nn.Conv2d(self.in_channels[0] // 2, self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.in_channels[0], self.in_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm(self.in_channels[0]),
            nn.GELU(),
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )

        if use_act_checkpoint:
            #self.p3 = checkpoint_wrapper(self.p3)
            #self.p4 = checkpoint_wrapper(self.p4)
            #self.p5 = checkpoint_wrapper(self.p5)
            #self.p6 = checkpoint_wrapper(self.p6)
            #if self.use_p2:
            #    self.p2 = checkpoint_wrapper(self.p2)
            print("checkpoint_warpper is not supported in this version.")

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        x = inputs[0]
        p4 = self.p4(x)
        p3 = self.p3(x)
        p5 = self.p5(x)
        p6 = self.p6(x)
        outs = [p3, p4, p5, p6]
        if self.use_p2:
            outs = [self.p2(x)] + outs
        return tuple(outs)
