# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn.model_input import str2enum, generate_model_input
from vpunn import VPUNN
import argparse
import os


class VPUCostModel:
    def __init__(self, filename, profile=False, verbose=False):
        if not os.path.isfile(filename):
            print(f"WARNING: file {filename} does not exists")
        self.model = VPUNN.VPUCostModel(filename, profile)
        if not self.model.nn_initialized():
            print("WARNING: VPUNN model not initialized... using simplistic model")
        self.verbose = verbose

    def DMA(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DMA cycles
        return self.model.DMA(generate_model_input(args, VPUNN.DMAWorkload))

    def describe_input(self, args):
        print("====================== Operation ======================")
        for key, value in args.items():
            print(f"\t{key} = {value}")
        print("=======================================================")

    def DPU(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DPU cycles
        return self.model.DPU(generate_model_input(args))

    def hw_overhead(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DPU hw utilization
        return self.model.compute_hw_overhead(generate_model_input(args))

    def hw_utilization(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DPU hw utilization
        return self.model.hw_utilization(generate_model_input(args))

    def DPUActivityFactor(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DPU hw utilization
        return self.model.DPUActivityFactor(generate_model_input(args))

    def DPUPower(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DPU hw utilization
        return self.model.DPUPower(generate_model_input(args))

    def DMAPower(self, **args):

        if self.verbose:
            self.describe_input(args)
        # Get DMA hw utilization
        return self.model.DMAPower(generate_model_input(args, VPUNN.DMAWorkload))


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
    parser = argparse.ArgumentParser(description="VPU cost model")

    parser.add_argument("--model", "-m", default="", type=str, help="Model path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["DPU", "DMA", "Utilization"],
        default="DPU",
        help="Profiling mode",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=[
            "cycles",
            "power",
        ],
        default="cycles",
        help="Target type",
    )
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
    parser.add_argument(
        "--mpe_mode",
        type=str,
        choices=["4x4", "16x1", "4x1"],
        default="4x4",
        help="DPU MPE mode",
    )
    parser.add_argument(
        "--nthw-ntk",
        type=str,
        choices=["4x16", "8x8", "16x4"],
        help="DPU nthw-ntk mode",
        default="8x8",
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
    parser.add_argument(
        "--input-swizzling", type=int, default=0, help="Input tensor swizzling"
    )
    parser.add_argument(
        "--param-swizzling", type=int, default=0, help="Weight tensor swizzling"
    )
    parser.add_argument(
        "--output-swizzling", type=int, default=0, help="output tensor swizzling"
    )

    parser.add_argument(
        "--output-write-tiles",
        type=int,
        default=1,
        help="Controls on how many tiles the DPU broadcast (1 = no broadcast)",
    )

    args = parser.parse_args()

    if args.device.upper() in ["VPU_2_7", "VPU_4_0"]:
        if args.nthw_ntk == "4x16":
            args.execution_mode = "CUBOID_4x16"
        elif args.nthw_ntk == "8x8":
            args.execution_mode = "CUBOID_8x16"
        else:
            args.execution_mode = "CUBOID_16x16"
    else:
        if args.input_dtype.upper() in ["FLOAT16", "BFLOAT16"]:
            args.execution_mode = "VECTOR_FP16"
        elif args.mpe_mode == "4x4":
            args.execution_mode = "MATRIX"
        else:
            args.execution_mode = "VECTOR"

    if args.model == "":
        args.model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"models/{args.device.lower()}.vpunn",
        )

    return args


def main():

    args = define_and_parse_args()
    model = VPUCostModel(args.model, verbose=True)
    input_height, input_width = getInputDim(
        [args.height, args.width],
        [args.kernel, args.kernel],
        [args.padding, args.padding],
        [args.strides, args.strides],
    )

    if args.mode == "Utilization":
        call_function = (
            model.hw_utilization if args.target == "cycles" else model.DPUActivityFactor
        )
        hw_utilization = call_function(
            device=f"VPUDevice.{args.device.upper()}",
            operation=f"Operation.{args.operation.upper()}",
            input_dimension=[
                input_width,
                input_height,
                args.input_channels,
                args.batch,
            ],
            output_dimension=[
                args.width,
                args.height,
                args.output_channels,
                args.batch,
            ],
            input_0_datatype=f"DataType.{args.input_dtype}",
            output_datatype=f"DataType.{args.output_dtype}",
            activation_function=f"ActivationFunction.{args.activation.upper()}",
            execution_order=f"ExecutionMode.{args.execution_mode}",
            Kh=args.kernel,
            Kw=args.kernel,
            Sh=args.strides,
            Sw=args.strides,
            Ph=args.padding,
            Pw=args.padding,
            act_sparsity=args.act_sparsity,
            param_sparsity=args.param_sparsity,
            input_0_swizzling=f"Swizzling.KEY_{args.input_swizzling}",
            input_1_swizzling=f"Swizzling.KEY_{args.param_swizzling}",
            output_0_swizzling=f"Swizzling.KEY_{args.output_swizzling}",
            output_write_tiles=args.output_write_tiles,
        )
        print(f"DPU hardware utilization {args.target}: {hw_utilization * 100:.3f}%")
    elif args.mode == "DPU":
        function_ = model.DPU if args.target == "cycles" else model.DPUPower
        cycles = function_(
            device=f"VPUDevice.{args.device.upper()}",
            operation=f"Operation.{args.operation.upper()}",
            input_dimension=[
                input_width,
                input_height,
                args.input_channels,
                args.batch,
            ],
            output_dimension=[
                args.width,
                args.height,
                args.output_channels,
                args.batch,
            ],
            input_0_datatype=f"DataType.{args.input_dtype}",
            output_datatype=f"DataType.{args.output_dtype}",
            activation_function=f"ActivationFunction.{args.activation.upper()}",
            execution_order=f"ExecutionMode.{args.execution_mode}",
            Kh=args.kernel,
            Kw=args.kernel,
            Sh=args.strides,
            Sw=args.strides,
            Ph=args.padding,
            Pw=args.padding,
            act_sparsity=args.act_sparsity,
            param_sparsity=args.param_sparsity,
            input_0_swizzling=f"Swizzling.KEY_{args.input_swizzling}",
            input_1_swizzling=f"Swizzling.KEY_{args.param_swizzling}",
            output_0_swizzling=f"Swizzling.KEY_{args.output_swizzling}",
            output_write_tiles=args.output_write_tiles,
        )
        print(f"DPU execution {args.target}: {cycles}")
    elif args.mode == "DMA":
        function_ = model.DMA if args.target == "cycles" else model.DMAPower
        cycles = function_(
            device=f"VPUDevice.{args.device.upper()}",
            input_dimension=[
                input_width,
                input_height,
                args.input_channels,
                args.batch,
            ],
            output_dimension=[
                args.width,
                args.height,
                args.output_channels,
                args.batch,
            ],
            input_location=f"MemoryLocation.DRAM",
            output_location=f"MemoryLocation.CMX",
            input_dtype=f"DataType.{args.input_dtype}",
            output_dtype=f"DataType.{args.output_dtype}",
        )

        print(f"DMA execution {args.target}: {cycles}")


if __name__ == "__main__":
    # Running the main function
    main()
