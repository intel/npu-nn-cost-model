# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn.model_input import str2enum, generate_model_input
from vpunn import VPUNN_lib
import argparse
import os


class VPUCostModel:
    def __init__(self, filename, profile=False, verbose=False):
        if not os.path.isfile(filename):
            print(f"WARNING: file {filename} does not exists")
        self.model = VPUNN_lib.VPUCostModel(filename, profile)
        if not self.model.nn_initialized():
            print("WARNING: VPUNN model not initialized... using simplistic model")
        self.verbose = verbose

    def DMA(self, **args):
        if self.verbose:
            self.describe_input(args)
        # Get DMA cycles
        return self.model.DMA(generate_model_input(args, VPUNN_lib.DMAWorkload))

    def describe_input(self, args):
        print("====================== Operation ======================")
        for key, value in args.items():
            print(f"\t{key} = {value}")
        print("=======================================================")

    def DPU(self, **args):
        if self.verbose:
            self.describe_input(args)
        wl = generate_model_input(args)
        cycles, error_msg = self.model.DPUMsg(wl)
        if error_msg != '':
            return error_msg
        else:
            return cycles
    
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
        return self.model.DMAPower(generate_model_input(args, VPUNN_lib.DMAWorkload))


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

def sparsity_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0)")
    return x

def padding_type(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError(f"{x} smaller than 0, invalid for padding")
    return x

def stride_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError(f"{x} smaller than 1, invalid for stride")
    return x

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="VPU cost model")

    parser.add_argument("--model", "-m", default="", type=str, help="Model path")
    parser.add_argument('-t','--target', dest='target', type=str, help='The target type', choices=["cycles", "power", "utilization"], default="cycles")
    parser.add_argument('-d','--device', dest='device', type=str, help='The VPU IP device', choices=['VPU_2_0', 'VPU_2_1', 'VPU_2_7'], required=True)

    subparsers = parser.add_subparsers(dest='module')
    subparsers.required = True
    parser_dpu = subparsers.add_parser('DPU')
    parser_dma = subparsers.add_parser('DMA')
    
    parser_dpu.add_argument('-o','--op','--operation', dest='operation', type=str, help='Operation type', required=True, choices=[
                "CONVOLUTION",
                "DW_CONVOLUTION",
                "ELTWISE",
                "MAXPOOL",
                "AVEPOOL",
                "CM_CONVOLUTION",
            ])
    parser_dpu.add_argument('--inch','--input-channels','--input-0-channels', dest='input_0_channels', type=int, help='Number of input channels', required=True)
    parser_dpu.add_argument('--outch','--output-channels','--output-0-channels', dest='output_0_channels', type=int, help='Number of output channels')
    parser_dpu.add_argument('--height','--input-height','--input-0-height', dest='input_0_height', type=int, help='Input activation height', required=True)
    parser_dpu.add_argument('--width','--input-width','--input-0-width', dest='input_0_width', type=int, help='Input activation width', required=True)
    # parser_dpu.add_argument('--input-sparsity-enabled', help='The flag to enable input sparsity', action='store_true')
    parser_dpu.add_argument('--weight-sparsity-enabled', help='The flag to enable weight sparsity', action='store_true')
    # parser_dpu.add_argument('--input-sparsity-rate', type=sparsity_float, help='The rate of input sparsity (only valid when enabling input sparsity)', default=0.0)
    parser_dpu.add_argument('--weight-sparsity-rate', type=sparsity_float, help='The rate of weight sparsity (only valid when enabling weight sparsity)', default=0.0)
    parser_dpu.add_argument('--mpe-mode','--execution-order','--execution-mode', dest='execution_order', type=str, help='For KMB device set the MPE mode (VECTOR_FP16, VECTOR, MATRIX), for later devices it sets the Execution Order (nthw)', required=True, choices=[
                'VECTOR_FP16',
                'VECTOR',
                'MATRIX',
                'CUBOID_4x16',
                'CUBOID_8x16',
                'CUBOID_16x16'
            ])
    parser_dpu.add_argument('--kh','--kernel-height', dest='kernel_height', type=int, help='The kernel height', required=True)
    parser_dpu.add_argument('--kw','--kernel-width', dest='kernel_width', type=int, help='The kernel width', required=True)
    parser_dpu.add_argument('--pb','--pad-bottom', dest='kernel_pad_bottom', type=padding_type, help='The bottom padding', default=0)
    parser_dpu.add_argument('--pl','--pad-left', dest='kernel_pad_left', type=padding_type, help='The left padding', default=0)
    parser_dpu.add_argument('--pr','--pad-right', dest='kernel_pad_right', type=padding_type, help='The right padding', default=0)
    parser_dpu.add_argument('--pt','--pad-top', dest='kernel_pad_top', type=padding_type, help='The top padding', default=0)
    parser_dpu.add_argument('--sh','--stride-height', dest='kernel_stride_height', type=stride_type, help='The stride height', default=1)
    parser_dpu.add_argument('--sw','--stride-width', dest='kernel_stride_width', type=stride_type, help='The stride width ', default=1)
    parser_dpu.add_argument('--indt','--input-datatype','--input_datatype', dest='input_0_datatype', type=str, help='The input datatype', choices=["UINT8", "INT8", "FLOAT16", "BFLOAT16"], required=True)
    parser_dpu.add_argument('--outdt','--output-datatype','--output_datatype', dest='output_0_datatype', type=str, help='The output datatype', choices=["UINT8", "INT8", "FLOAT16", "BFLOAT16"], required=True)
    parser_dpu.add_argument('--isi','--isi-strategy', dest='isi_strategy', type=str, help='The ISI Strategy', default='CLUSTERING', choices=[
                            'CLUSTERING',
                            'SOH',
                            'SOK'
                        ])
    parser_dpu.add_argument('--owt','--output-write-tiles', dest='output_write_tiles', type=int, help='Controls on how many tiles the DPU broadcast (1 = no broadcast)', default=1)
    # parser_dpu.add_argument('--output-sparsity-enabled', help='The flag to enable output sparsity', action='store_true')

    parser_dma.add_argument('--height', type=int, required=True)
    parser_dma.add_argument('--width', type=int, required=True)
    parser_dma.add_argument('--kernel', type=int, required=True)
    parser_dma.add_argument('--padding', type=int, required=True)
    parser_dma.add_argument('--strides', type=int, required=True)
    parser_dma.add_argument('--device', type=str, required=True)
    parser_dma.add_argument('--input_channels', type=int, required=True)
    parser_dma.add_argument('--output_channels', type=int, required=True)
    parser_dma.add_argument('--input_dtype', type=int, required=True)
    parser_dma.add_argument('--output_dtype', type=int, required=True)
    
    args = parser.parse_args()

    if args.output_0_channels is None:
        if args.operation in ['cm_convolution', 'convolution']:
            raise argparse.ArgumentTypeError(f"The number of output channels must be specified when operation is set to cm_convolution or convolution")
        else:
            args.output_0_channels = args.input_0_channels
        
    args.input_1_datatype = args.input_0_datatype
    
    if args.model == "":
        args.model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"models/{args.device.lower()}.vpunn",
        )
    return args

def main():

    args = define_and_parse_args()
    model = VPUCostModel(args.model, verbose=True)
    
    isi_dict = {'CLUSTERING':'CLUSTERING','SOH':'SPLIT_OVER_H','SOK':'SPLIT_OVER_K'}

    if args.module == "DPU":
        if args.target == "cycles":
            function_ = model.DPU
        elif args.target == "power":
            function_ = model.DPUActivityFactor
        else:
            function_ = model.hw_utilization
        result = function_(
            device = f"VPUDevice.{args.device}",
            operation = f"Operation.{args.operation}",
            input_0_channels = args.input_0_channels,
            output_0_channels = args.output_0_channels,
            input_0_height = args.input_0_height,
            input_0_width = args.input_0_width,
            input_sparsity_enabled = False,#args.input_sparsity_enabled,
            weight_sparsity_enabled = args.weight_sparsity_enabled,
            input_sparsity_rate = 0.0,#args.input_sparsity_rate,
            weight_sparsity_rate = args.weight_sparsity_rate,
            execution_order = f"ExecutionMode.{args.execution_order}",
            activation_function = f"ActivationFunction.NONE",
            kernel_height = args.kernel_height,
            kernel_width = args.kernel_width,
            kernel_pad_bottom = args.kernel_pad_bottom,
            kernel_pad_left = args.kernel_pad_left,
            kernel_pad_right = args.kernel_pad_right,
            kernel_pad_top = args.kernel_pad_top,
            kernel_stride_height = args.kernel_stride_height,
            kernel_stride_width = args.kernel_stride_width,
            input_0_datatype = f"DataType.{args.input_0_datatype}",
            input_1_datatype = f"DataType.{args.input_1_datatype}",
            output_0_datatype = f"DataType.{args.output_0_datatype}",
            input_0_layout = f"Layout.ZXY",
            input_1_layout = f"Layout.ZXY",
            output_0_layout = f"Layout.ZXY",
            input_0_swizzling = f"Swizzling.KEY_0",
            input_1_swizzling = f"Swizzling.KEY_0",
            output_0_swizzling = f"Swizzling.KEY_0",
            isi_strategy = f"ISIStrategy.{isi_dict[args.isi_strategy]}",
            output_write_tiles = 2 if args.isi_strategy=='SOK' else args.output_write_tiles,
            output_sparsity_enabled = False#args.output_sparsity_enabled,
        )
        print(f"DPU execution {args.target}: {result}")
    elif args.mode == "DMA":
        input_height, input_width = getInputDim(
            [args.height, args.width],
            [args.kernel, args.kernel],
            [args.padding, args.padding],
            [args.strides, args.strides],
        )
        if args.target == "cycles" :
            function_ = model.DMA 
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
