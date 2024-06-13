# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn.model_input import str2enum
from vpunn import VPUNN_lib
import argparse
import os


class VPULayerCostModel:
    def __init__(self, filename, profile=False, verbose=False):
        if not os.path.isfile(filename):
            print(f"WARNING: file {filename} does not exists")
        self.model = VPUNN_lib.VPULayerCostModel(filename, profile, 16384, 1)
        if not self.model.nn_initialized():
            print("WARNING: VPUNN model not initialized... using simplistic model")
        self.verbose = verbose

    def describe_input(self, args):
        print("====================== Operation ======================")
        for key, value in args.items():
            print(f"\t{key} = {value}")
        print("=======================================================")

    def Layer(self, nDPU, nTiles, input_ddr, output_ddr, prefetch, **args):
        if self.verbose:
            self.describe_input(args)
        # Get layer cycles
        layer = VPUNN_lib.DPULayer(
            str2enum(args["device"]),
            str2enum(args["operation"]),
            [
                VPUNN_lib.VPUTensor(
                    args["input_dimension"], str2enum(args["input_0_datatype"])
                )
            ],
            [
                VPUNN_lib.VPUTensor(
                    args["output_dimension"], str2enum(args["output_datatype"])
                )
            ],
            [args["Kw"], args["Kh"]],
            [args["Sw"], args["Sh"]],
            [args["Pw"], args["Pw"], args["Ph"], args["Ph"]],
        )
        return self.model.Layer(layer, nDPU, nTiles, input_ddr, output_ddr, prefetch)


def getInputDim(output_dim, kernel, padding, strides):
    def helper_input_dim(o, k, p, s):
        # output dim, kernel size, padding, stride
        i = (o - 1) * s - 2 * p + k
        assert o == (i + 2 * p - k) // s + 1
        return i

    return [
        helper_input_dim(o, k, p, z)
        for o, k, p, z in zip(output_dim, kernel, padding, strides)
    ]


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="VPU layer cost model")

    parser.add_argument("--model", "-m", default="", type=str, help="Model path")
    parser.add_argument("--nDPU", type=int, default=1, help="Number of DPU/tile")
    parser.add_argument("--nTiles", type=int, default=1, help="Number of CMX tiles")

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["VPU_2_0", "VPU_2_1", "VPU_2_7", "VPU_4_0"],
        help="The VPU IP device",
    )
    parser.add_argument(
        "--operation",
        "-op",
        type=str,
        choices=[
            "CONVOLUTION",
            "DW_CONVOLUTION",
            "ELTWISE",
            "MAXPOOL",
            "AVEPOOL",
            "CM_CONVOLUTION",
        ],
        default="CONVOLUTION",
        help="The operation",
    )
    parser.add_argument(
        "--activation",
        "-act",
        type=str,
        choices=["NONE", "RELU", "MULT", "LRELU", "ADD", "SUB"],
        default="NONE",
        help="The operation activation function",
    )
    parser.add_argument("--input-ddr", action="store_true", help="Input in DDR")
    parser.add_argument("--output-ddr", action="store_true", help="Output in DDR")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Consider Prefetch in layer cost calculation",
    )
    parser.add_argument("--width", "-x", type=int, help="Tensor width")
    parser.add_argument("--height", "-y", type=int, help="Tensor height")
    parser.add_argument(
        "--input_channels", "-ic", type=int, help="Tensor input channels"
    )
    parser.add_argument(
        "--output_channels", "-oc", type=int, help="Tensor output channels"
    )
    parser.add_argument("--batch", "-b", type=int, default=1, help="Tensor batch")
    parser.add_argument("--kernel", "-k", type=int, default=1, help="Operation Kernel")
    parser.add_argument(
        "--padding", "-p", type=int, default=0, help="Operation padding"
    )
    parser.add_argument(
        "--strides", "-s", type=int, default=1, help="Operation strides"
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        choices=["UINT8", "INT8", "FLOAT16", "BFLOAT16"],
        default="UINT8",
        help="The input datatype",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        choices=["UINT8", "INT8", "FLOAT16", "BFLOAT16"],
        default="UINT8",
        help="The output datatype",
    )

    parser.add_argument(
        "--act-sparsity", type=float, default=0, help="Input tensor sparsity"
    )
    parser.add_argument(
        "--param-sparsity", type=float, default=0, help="Weight tensor sparsity"
    )

    args = parser.parse_args()

    if args.model == "":
        args.model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"models/{args.device.lower()}.vpunn",
        )

    return args


def main():

    args = define_and_parse_args()

    model = VPULayerCostModel(args.model, verbose=True)
    input_height, input_width = getInputDim(
        [args.height, args.width],
        [args.kernel, args.kernel],
        [args.padding, args.padding],
        [args.strides, args.strides],
    )

    layer_cycles = model.Layer(
        device=f"VPUDevice.{args.device.upper()}",
        operation=f"Operation.{args.operation.upper()}",
        input_dimension=[input_width, input_height, args.input_channels, args.batch],
        output_dimension=[args.width, args.height, args.output_channels, args.batch],
        input_0_datatype=f"DataType.{args.input_dtype}",
        output_datatype=f"DataType.{args.output_dtype}",
        activation_function=f"ActivationFunction.{args.activation.upper()}",
        Kh=args.kernel,
        Kw=args.kernel,
        Sh=args.strides,
        Sw=args.strides,
        Ph=args.padding,
        Pw=args.padding,
        act_sparsity=args.act_sparsity,
        param_sparsity=args.param_sparsity,
        nDPU=args.nDPU,
        nTiles=args.nTiles,
        input_ddr=args.input_ddr,
        output_ddr=args.output_ddr,
        prefetch=args.prefetch,
    )
    print(f"Layer cost: {layer_cycles} cycles")


if __name__ == "__main__":
    # Running the main function
    main()
