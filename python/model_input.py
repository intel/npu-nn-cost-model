# Copyright © 2023 Intel Corporation
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

    wl = wl_type()
    if wl_type == VPUNN_lib.DPUWorkload:
        dpu_wl = Namespace(**args)

        # Convert all input to VPUNN enum
        wl.device = str2enum(dpu_wl.device)
        wl.op = str2enum(dpu_wl.operation)
        wl.execution_order = str2enum(dpu_wl.execution_order)
        wl.input_swizzling = [
            str2enum(dpu_wl.input_0_swizzling),
            str2enum(dpu_wl.input_1_swizzling),
        ]
        wl.output_swizzling = [str2enum(dpu_wl.output_0_swizzling)]
        wl.kernels = [dpu_wl.kernel_width, dpu_wl.kernel_height]
        wl.strides = [dpu_wl.kernel_stride_width, dpu_wl.kernel_stride_height]
        wl.padding = [
            dpu_wl.kernel_pad_top,
            dpu_wl.kernel_pad_bottom,
            dpu_wl.kernel_pad_left,
            dpu_wl.kernel_pad_right,
        ]

        wl.act_sparsity = dpu_wl.input_sparsity_rate
        wl.weight_sparsity = dpu_wl.weight_sparsity_rate
        wl.output_write_tiles = dpu_wl.output_write_tiles

        wl.inputs = [
            VPUNN_lib.VPUTensor(
                [
                    dpu_wl.input_0_width,
                    dpu_wl.input_0_height,
                    dpu_wl.input_0_channels,
                    1,
                ],
                str2enum(dpu_wl.input_0_datatype),
                str2enum(dpu_wl.input_0_layout),
                dpu_wl.input_sparsity_enabled,
            )
        ]

        wl.weight_sparsity_enabled = dpu_wl.weight_sparsity_enabled

        wl.outputs = [
            VPUNN_lib.VPUTensor(
                [
                    dpu_wl.output_0_width if 'output_0_width' in args else int(((dpu_wl.input_0_width + (dpu_wl.kernel_pad_left + dpu_wl.kernel_pad_right) - (dpu_wl.kernel_width-1)-1)//dpu_wl.kernel_stride_width)+1),
                    dpu_wl.output_0_height if 'output_0_height' in args else int(((dpu_wl.input_0_height + (dpu_wl.kernel_pad_top + dpu_wl.kernel_pad_bottom) - (dpu_wl.kernel_height-1)-1)//dpu_wl.kernel_stride_height)+1),
                    dpu_wl.output_0_channels,
                    1,
                ],
                str2enum(dpu_wl.output_0_datatype),
                str2enum(dpu_wl.output_0_layout),
                dpu_wl.output_sparsity_enabled,
            )
        ]

        wl.output_write_tiles = dpu_wl.output_write_tiles
        wl.activation_function = str2enum(dpu_wl.activation_function)
        wl.isi_strategy = str2enum(dpu_wl.isi_strategy)

    elif wl_type == VPUNN_lib.DMAWorkload:
        # Convert all input to VPUNN enum
        wl.device = str2enum(args["device"])
        if "input_dimension" not in args.keys():
            args["input_dimension"] = [
                args.get("input_0_width", args.get("output_0_width")),
                args.get("input_0_height", args.get("output_0_height")),
                args["input_0_channels"],
                1,
            ]
        if "output_dimension" not in args.keys():
            args["output_dimension"] = [
                args["output_0_width"],
                args["output_0_height"],
                args["output_0_channels"],
                1,
            ]

        # This field was renamed in the database, so old models may still use the old name
        input_dtype_key = list(
            filter(
                lambda key: key in args.keys(),
                ["input_0_datatype", "input_datatype", "input_dtype"],
            )
        )[0]

        # Create the tensor
        wl.input = VPUNN_lib.VPUTensor(
            args["input_dimension"], str2enum(args[input_dtype_key])
        )

        # This field was renamed in the database, so old models may still use the old name
        output_dtype_key = list(
            filter(
                lambda key: key in args.keys(),
                ["output_0_datatype", "output_datatype", "output_dtype"],
            )
        )[0]

        wl.output = VPUNN_lib.VPUTensor(
            args["output_dimension"], str2enum(args[output_dtype_key])
        )

        wl.input_location = str2enum(args["input_location"])
        wl.output_location = str2enum(args["output_location"])
    else:
        raise Exception("Workload type is not defined")

    return wl
