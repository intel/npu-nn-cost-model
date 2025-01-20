# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn import VPUNN_lib
import numpy as np
from argparse import Namespace

class Preprocessing:
    def __init__(self):
        self.model = VPUNN_lib.Preprocessing_Interface10_float_t()

    def transform(self, args):
        vector = self.model.transform(generate_model_input(args))
        return np.array(vector)

def str2enum(value):
    return eval(f"VPUNN_lib.{value}")

def generate_model_input(args, wl_type=VPUNN_lib.DPUWorkload):
    if isinstance(args, Namespace):
        wl_ns = args
    else:
        wl_ns = Namespace(**args)

    wl = wl_type()
    if wl_type == VPUNN_lib.DPUWorkload:

        # Convert all input to VPUNN enum
        wl.device = str2enum(wl_ns.device)
        wl.op = str2enum(wl_ns.operation)
        wl.execution_order = str2enum(wl_ns.execution_order)
        wl.input_swizzling = [
            str2enum(wl_ns.input_0_swizzling),
            str2enum(wl_ns.input_1_swizzling),
        ]
        wl.output_swizzling = [str2enum(wl_ns.output_0_swizzling)]
        wl.kernels = [wl_ns.kernel_width, wl_ns.kernel_height]
        wl.strides = [wl_ns.kernel_stride_width, wl_ns.kernel_stride_height]
        wl.padding = [
            wl_ns.kernel_pad_top,
            wl_ns.kernel_pad_bottom,
            wl_ns.kernel_pad_left,
            wl_ns.kernel_pad_right,
        ]

        wl.act_sparsity = wl_ns.input_sparsity_rate
        wl.weight_sparsity = wl_ns.weight_sparsity_rate
        wl.output_write_tiles = wl_ns.output_write_tiles

        wl.inputs = [
            VPUNN_lib.VPUTensor(
                [
                    wl_ns.input_0_width,
                    wl_ns.input_0_height,
                    wl_ns.input_0_channels,
                    1,
                ],
                str2enum(wl_ns.input_0_datatype),
                str2enum(wl_ns.input_0_layout),
                wl_ns.input_sparsity_enabled,
            )
        ]

        wl.weight_sparsity_enabled = wl_ns.weight_sparsity_enabled

        wl.outputs = [
            VPUNN_lib.VPUTensor(
                [
                    wl_ns.output_0_width if 'output_0_width' in args else int(((wl_ns.input_0_width + (wl_ns.kernel_pad_left + wl_ns.kernel_pad_right) - (wl_ns.kernel_width-1)-1)//wl_ns.kernel_stride_width)+1),
                    wl_ns.output_0_height if 'output_0_height' in args else int(((wl_ns.input_0_height + (wl_ns.kernel_pad_top + wl_ns.kernel_pad_bottom) - (wl_ns.kernel_height-1)-1)//wl_ns.kernel_stride_height)+1),
                    wl_ns.output_0_channels,
                    1,
                ],
                str2enum(wl_ns.output_0_datatype),
                str2enum(wl_ns.output_0_layout),
                False,
            )
        ]

        wl.output_write_tiles = wl_ns.output_write_tiles
        wl.activation_function = str2enum(wl_ns.activation_function)
        wl.isi_strategy = str2enum(wl_ns.isi_strategy)

        return wl
    elif wl_type == VPUNN_lib.DMANNWorkload_NPU27:
        dma_wl = wl_type()
        dma_wl.device = str2enum(f"VPUDevice.VPU_2_7")
        dma_wl.length = wl_ns.length
        dma_wl.src_width = wl_ns.src_width
        dma_wl.dst_width = wl_ns.dst_width

        dma_wl.src_stride = wl_ns.src_stride
        dma_wl.dst_stride = wl_ns.dst_stride

        dma_wl.num_planes = wl_ns.num_planes
        dma_wl.src_plane_stride = wl_ns.src_plane_stride
        dma_wl.dst_plane_stride = wl_ns.dst_plane_stride
        dma_wl.transfer_direction = eval(f"VPUNN_lib.MemoryDirection.{wl_ns.direction}")
        return dma_wl

    elif wl_type == VPUNN_lib.DMANNWorkload_NPU40:
        dma_wl = wl_type()

        dma_wl.num_engine = eval(f"VPUNN_lib.Num_DMA_Engine.Num_Engine_{wl_ns.num_engine}")
        dma_wl.transfer_direction = eval(f"VPUNN_lib.MemoryDirection.{wl_ns.direction}")

        dma_wl.src_width  = wl_ns.src_width
        dma_wl.dst_width  = wl_ns.dst_width
        dma_wl.num_dim = wl_ns.num_dim

        ss = VPUNN_lib.DMANNWorkload_NPU40.SizeStride
        size_strides = [ss(),ss(),ss(),ss(),ss()]
        for i in range(1,6):
            for field in vars(wl_ns).keys():
                if field.endswith(f'_{i}'):
                    setattr(size_strides[i-1], field[:-2], getattr(wl_ns, field))
        dma_wl.e_dim = size_strides
        return dma_wl
    else:
        raise Exception("Workload type is not defined")
