# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn import VPUNN
import numpy as np


class Preprocessing:
    def __init__(self):
        self.model = VPUNN.Preprocessing(float)()

    def transform(self, **args):
        vector = self.model.transform(generate_model_input(args))

        return np.array(vector)


def str2enum(value):
    return eval(f"VPUNN.{value}")


def generate_model_input(args, wl_type=VPUNN.DPUWorkload):
    wl = wl_type()
    if wl_type == VPUNN.DPUWorkload:
        # Convert all input to VPUNN enum
        wl.device = str2enum(args["device"])
        wl.op = str2enum(args["operation"])
        wl.activation_function = str2enum(
            args.get("activation_function", "ActivationFunction.NONE")
        )
        wl.execution_order = str2enum(
            args.get("execution_order", "ExecutionMode.MATRIX")
        )
        wl.input_swizzling = [
            str2enum(args.get("input_0_swizzling", "Swizzling.KEY_0")),
            str2enum(args.get("input_1_swizzling", "Swizzling.KEY_0")),
        ]
        wl.output_swizzling = [
            str2enum(args.get("output_0_swizzling", "Swizzling.KEY_0"))
        ]
        wl.kernels = [
            args.get("Kw", args.get("kernel_width", 1)),
            args.get("Kh", args.get("kernel_height", 1)),
        ]
        wl.strides = [
            args.get("Sw", args.get("kernel_stride_width", 1)),
            args.get("Sh", args.get("kernel_stride_height", 1)),
        ]
        wl.padding = [
            args.get("Ph", args.get("kernel_pad_top", args.get("layer_pad_top", 0))),
            args.get(
                "Ph", args.get("kernel_pad_bottom", args.get("layer_pad_bottom", 0))
            ),
            args.get("Pw", args.get("kernel_pad_left", args.get("layer_pad_left", 0))),
            args.get(
                "Pw", args.get("kernel_pad_right", args.get("layer_pad_right", 0))
            ),
        ]

        wl.act_sparsity = args.get(
            "act_sparsity",
            args.get("activation_sparsity_rate", args.get("input_sparsity_rate", 0)),
        )
        wl.weight_sparsity = args.get(
            "param_sparsity", args.get("weight_sparsity_rate", 0)
        )
        wl.output_write_tiles = args.get("output_write_tiles", 1)

        # make sure input_0_dimension, input_1_dimension and output_0 exists_dimension
        # input_dimension might be present, then input_0/1 is not expected

        if "input_dimension" in args.keys():  # just move to input_0/1
            args["input_0_dimension"] = args["input_dimension"]  # intercept it
            args["input_1_dimension"] = [0, 0, 0, 0]  # a non specified/ unknown value
        else:  # use or build input_0 and 1
            if "input_0_dimension" not in args.keys():
                args["input_0_dimension"] = [
                    args["input_0_width"],
                    args["input_0_height"],
                    args["input_0_channels"],
                    args["input_0_batch"],
                ]
            # if "input_1_dimension" not in args.keys():
            #     args["input_1_dimension"] = [
            #         args["input_1_width"],
            #         args["input_1_height"],
            #         args["input_1_channels"],
            #         args["input_1_batch"],
            #     ]

        if "output_dimension" in args.keys():  # just move to output_0
            args["output_0_dimension"] = args["output_dimension"]
        else:  # output_0 is either present or constructed
            if "output_0_dimension" not in args.keys():
                args["output_0_dimension"] = [
                    args["output_0_width"],
                    args["output_0_height"],
                    args["output_0_channels"],
                    args["output_0_batch"],
                ]
        # in-outs dimensions exist here

        # Create the tensor
        wl.inputs = [
            VPUNN.VPUTensor(
                args["input_0_dimension"],
                str2enum(args["input_0_datatype"]),
                args.get("input_sparsity_enabled", 0),
            )
        ]

        # wl.inputs_1 = [
        #     VPUNN.VPUTensor(
        #         args["input_1_dimension"],
        #         str2enum(args.get("input_1_datatype",args["input_0_datatype"])),
        #         args.get("weight_sparsity_enabled", 0),
        #     )
        # ]

        wl.outputs = [
            VPUNN.VPUTensor(
                args["output_0_dimension"],
                str2enum(args.get("output_0_datatype", args.get("output_datatype"))),
                args.get("output_sparsity_enabled", 0),
            )
        ]
    elif wl_type == VPUNN.DMAWorkload:
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
        wl.input = VPUNN.VPUTensor(
            args["input_dimension"], str2enum(args[input_dtype_key])
        )

        # This field was renamed in the database, so old models may still use the old name
        output_dtype_key = list(
            filter(
                lambda key: key in args.keys(),
                ["output_0_datatype", "output_datatype", "output_dtype"],
            )
        )[0]

        wl.output = VPUNN.VPUTensor(
            args["output_dimension"], str2enum(args[output_dtype_key])
        )

        wl.input_location = str2enum(args["input_location"])
        wl.output_location = str2enum(args["output_location"])
    else:
        raise Exception("Workload type is not defined")

    return wl
