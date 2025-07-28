import hetu as ht
import numpy as np
import torch
import math
from typing import Optional, Tuple

from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_multi_ds_parallel_config
from unet_2d_blocks import get_down_block




def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dropout: float = 0.0,
):


    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
        )        
    else:
        raise NotImplementedError(f"{down_block_type} does not exist.")




class DownEncoderBlock2D(ht.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = ht.nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = ht.nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states

class Downsample2D(ht.nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name


        self.conv = ht.nn.Conv2d(
            self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, hidden_states) -> ht.Tensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states



class ResnetBlock2D(ht.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        ds_parallel_configs = None,
    ):
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.ds_parallel_configs = ds_parallel_configs


        # self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)


        self.conv_in = ht.HtParallelConv2d(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernen_size=3,
            stride=1,
            padding=1, 
            bias=False,
            multi_ds_parallel_config = get_multi_ds_parallel_config(self.ds_parallel_configs, 'conv1'),
            name="proj"
        )


        # self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        # self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)



        self.conv2 = ht.HtParallelConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernen_size=3,
            stride=1,
            padding=1, 
            bias=False,
            multi_ds_parallel_config = get_multi_ds_parallel_config(self.ds_parallel_configs, 'conv2'),
            name="proj"
        )


    def forward(self, input_tensor) -> torch.Tensor:

        hidden_states = input_tensor

        # hidden_states = self.norm1(hidden_states)
        hidden_states = ht.silu(hidden_states)


        hidden_states = self.conv1(hidden_states)
        # hidden_states = self.norm2(hidden_states)

        hidden_states = ht.silu(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor
