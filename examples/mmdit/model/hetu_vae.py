import hetu as ht
import numpy as np
import torch
import math
from typing import Optional, Tuple

from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_multi_ds_parallel_config



class DownEncoderBlock3D(ht.nn.Module):

    def __init__(self, num_layers, in_channels, out_channels, dropout, add_downsample, resnet_eps, downsample_padding, ds_parallel_configs, layer_idx):
        super(DownEncoderBlock3D, self).__init__()
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.resnet_eps = resnet_eps
        self.add_downsample = add_downsample
        self.downsample_padding = downsample_padding
    
        self.resnets = []


        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    eps=self.resnet_eps,
                    # groups=self.resnet_groups,
                    dropout=self.dropout,
                    ds_parallel_configs=ds_parallel_configs,
                )
            )
        self.resnets = ht.nn.ModuleList(self.resnets)

        if self.add_downsample:
            self.downsamplers = ht.nn.ModuleList(
                [
                    Downsample2D(
                        self.out_channels, out_channels=self.out_channels, padding=self.downsample_padding, ds_parallel_configs=ds_parallel_configs,name="downsample"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            print("after resnet hidden_states", hidden_states.shape)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                print("after downsampler hidden_states", hidden_states.shape)
        
        return hidden_states



class Downsample2D(ht.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        bias=True,
        ds_parallel_configs = None,
    ):
        super(Downsample2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.padding = padding
        stride = 2
        self.name = name

        self.conv = ht.nn.HtParallelConv3d(in_channels = self.in_channels, 
                                   out_channels = self.out_channels, 
                                   kernel_size = kernel_size, 
                                   stride=stride, 
                                   padding=padding,
                                   bias=False,
                                   multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, 'conv'),
                                   name="conv_in")



    def forward(self, hidden_states) -> ht.Tensor:
        assert hidden_states.shape[1] == self.in_channels
        hidden_states = self.conv(hidden_states)
        return hidden_states



class ResnetBlock2D(ht.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        # groups: int = 32,
        eps: float = 1e-6,
        ds_parallel_configs = None,
    ):
        super(ResnetBlock2D, self).__init__()
        


        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.ds_parallel_configs = ds_parallel_configs


        # self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)


        self.conv1 = ht.nn.HtParallelConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1, 
            bias=False,
            multi_ds_parallel_config = get_multi_ds_parallel_config(self.ds_parallel_configs, 'conv'),
            name="conv1"
        )


        # self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        # self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)



        self.conv2 = ht.nn.HtParallelConv3d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1, 
            bias=False,
            multi_ds_parallel_config = get_multi_ds_parallel_config(self.ds_parallel_configs, 'conv'),
            name="conv2"
        )

        self.use_in_shortcut = self.in_channels != self.out_channels

        if self.use_in_shortcut:
            self.conv_shortcut = ht.nn.HtParallelConv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0, 
                bias=False,
                multi_ds_parallel_config = get_multi_ds_parallel_config(self.ds_parallel_configs, 'conv'),
                name="conv_shortcut"
            )


    def forward(self, input_tensor) -> torch.Tensor:

        hidden_states = input_tensor

        # hidden_states = self.norm1(hidden_states)
        hidden_states = ht.silu(hidden_states)

        assert(hidden_states.shape[1] == self.in_channels)
        hidden_states = self.conv1(hidden_states)
        # hidden_states = self.norm2(hidden_states)

        hidden_states = ht.silu(hidden_states)

        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        assert(hidden_states.shape[1] == self.out_channels)

        if self.use_in_shortcut:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor



class Encoder(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(Encoder, self).__init__()

        self.in_channels = config.in_channels
        self.block_out_channels = config.block_out_channels
        self.layers_per_block = config.layers_per_block
        # self.norm_num_groups = config.norm_num_groups
        # self.mid_block_add_attention = config.mid_block_add_attention
        

        self.conv_in = ht.nn.HtParallelConv3d(
            in_channels=self.in_channels,
            out_channels=self.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1, 
            bias=False,
            multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, 'conv'),
            name="proj"
        )


        self.down_blocks = []
        output_channel = self.block_out_channels[0]
        for i, _ in enumerate(self.block_out_channels):
            input_channel = output_channel
            output_channel = self.block_out_channels[i]
            is_final_block = i == len(self.block_out_channels) - 1


            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=0.0,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                # resnet_groups=self.norm_num_groups,
                downsample_padding=0,
                ds_parallel_configs=ds_parallel_configs,
                layer_idx=i
            )        
            self.down_blocks.append(down_block)
        self.down_blocks = ht.nn.ModuleList(self.down_blocks)

        # self.mid_block = UNetMidBlock2D(
        #     in_channels=self.block_out_channels[-1],
        #     resnet_eps=1e-6,
        #     resnet_act_fn=self.act_fn,
        #     output_scale_factor=1,
        #     resnet_time_scale_shift="default",
        #     attention_head_dim=self.block_out_channels[-1],
        #     resnet_groups=self.norm_num_groups,
        #     temb_channels=None,
        #     add_attention=self.mid_block_add_attention,
        # )
    def forward(self, hidden_states):
        assert(hidden_states.shape[1] == self.in_channels)
        hidden_states = self.conv_in(hidden_states)
        print("hidden_states after conv in", hidden_states.shape)
        for down_block in self.down_blocks:
            print("DownEncoderBlock3D", down_block.layer_idx)
            hidden_states = down_block(hidden_states)
            print("hidden_states", hidden_states.shape)
        # hidden_states = self.mid_block(hidden_states)
        return hidden_states

    

# class UNetMidBlock2D(ht.nn.Module):
#     """
#     A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

#     Args:
#         in_channels (`int`): The number of input channels.
#         temb_channels (`int`): The number of temporal embedding channels.
#         dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
#         num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
#         resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
#         resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
#             The type of normalization to apply to the time embeddings. This can help to improve the performance of the
#             model on tasks with long-range temporal dependencies.
#         resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
#         resnet_groups (`int`, *optional*, defaults to 32):
#             The number of groups to use in the group normalization layers of the resnet blocks.
#         attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
#         resnet_pre_norm (`bool`, *optional*, defaults to `True`):
#             Whether to use pre-normalization for the resnet blocks.
#         add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
#         attention_head_dim (`int`, *optional*, defaults to 1):
#             Dimension of a single attention head. The number of attention heads is determined based on this value and
#             the number of input channels.
#         output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

#     Returns:
#         `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size, in_channels,
#         height, width)`.

#     """

#     def __init__(
#         self,
#         in_channels: int,
#         temb_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",  # default, spatial
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         attn_groups: Optional[int] = None,
#         resnet_pre_norm: bool = True,
#         add_attention: bool = True,
#         attention_head_dim: int = 1,
#         output_scale_factor: float = 1.0,
#     ):
#         super().__init__()
#         resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
#         self.add_attention = add_attention

#         if attn_groups is None:
#             attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

#         resnets = [
#             ResnetBlock2D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 dropout=dropout,
#                 time_embedding_norm=resnet_time_scale_shift,
#                 non_linearity=resnet_act_fn,
#                 output_scale_factor=output_scale_factor,
#                 pre_norm=resnet_pre_norm,
#             )
#         ]
#         attentions = []


#         for _ in range(num_layers):
#             attentions.append(
#                 Attention(
#                     in_channels,
#                     heads=in_channels // attention_head_dim,
#                     dim_head=attention_head_dim,
#                     rescale_output_factor=output_scale_factor,
#                     eps=resnet_eps,
#                     norm_num_groups=attn_groups,
#                     spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
#                     residual_connection=True,
#                     bias=True,
#                     upcast_softmax=True,
#                     _from_deprecated_attn_block=True,
#                 )
#             )

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=in_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                 )
#             )

#         self.attentions = nn.ModuleList(attentions)
#         self.resnets = nn.ModuleList(resnets)

#         self.gradient_checkpointing = False

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.resnets[0](hidden_states)
#         for attn, resnet in zip(self.attentions, self.resnets[1:]):
#             if attn is not None:
#                 hidden_states = attn(hidden_states)
#             hidden_states = resnet(hidden_states)

#         return hidden_states
