import os
import argparse

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

def add_parser_dpu(subparsers_module):
    parser_dpu = subparsers_module.add_parser('DPU')
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
    parser_dpu.add_argument('--input-sparsity-enabled', help='The flag to enable input sparsity', action='store_true')
    parser_dpu.add_argument('--weight-sparsity-enabled', help='The flag to enable weight sparsity', action='store_true')
    parser_dpu.add_argument('--input-sparsity-rate', type=sparsity_float, help='The rate of input sparsity (only valid when enabling input sparsity)', default=0.0)
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

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="VPU cost model")

    parser.add_argument("--model", "-m", default="", type=str, help="Model path")

    subparsers_device = parser.add_subparsers(dest='device')
    subparsers_device.required = True
    parser_27 = subparsers_device.add_parser('VPU_2_7')
    subparsers_module = parser_27.add_subparsers(dest='module')
    subparsers_module.required = True
    add_parser_dpu(subparsers_module)

    parser_dma = subparsers_module.add_parser('DMA')

    parser_dma.add_argument('-l', '--length', dest='length', type=int, help='Length of the DMA transaction (in bytes)', required=True)
    parser_dma.add_argument('--sw', '--src_width', dest='src_width', type=int, help='Source width of the DMA transfer (in bytes)', required=True)
    parser_dma.add_argument('--dw', '--dst_width', dest='dst_width', type=int, help='Destination width of the DMA transfer (in bytes)', required=True)
    parser_dma.add_argument('--ss', '--src_stride', dest='src_stride', type=int, default=0, help='Source stride of the DMA transfer')
    parser_dma.add_argument('--ds', '--dst_stride', dest='dst_stride', type=int, default=0, help='Destination stride of the DMA transfer')
    parser_dma.add_argument('-p', '--num_planes', dest='num_planes', type=int, choices=[*range(0,8)], default=0, help='Number of DMA planes')
    parser_dma.add_argument('--sps', '--src_plane_stride', dest='src_plane_stride', type=int, help='Source plane stride of the DMA transfer', default=0)
    parser_dma.add_argument('--dps', '--dst_plane_stride', dest='dst_plane_stride', type=int, help='Destination plane stride of the DMA transfer', default=0)
    parser_dma.add_argument('-d', '--direction', dest='direction', type=str, help='Direction of the DMA transfer', required=True,
                            choices=['DDR2CMX', 'CMX2CMX', 'CMX2DDR', 'DDR2DDR'])

    parser_40 = subparsers_device.add_parser('VPU_4_0')
    subparsers_module = parser_40.add_subparsers(dest='module')
    subparsers_module.required = True
    parser_dpu = subparsers_module.add_parser('DPU')
    add_parser_dpu(subparsers_module)

    parser_dma = subparsers_module.add_parser('DMA')

    parser_dma.add_argument('--sw', '--src_width', dest='src_width', type=int, help='Source width of the DMA transfer (in bytes)', required=True)
    parser_dma.add_argument('--dw', '--dst_width', dest='dst_width', type=int, help='Destination width of the DMA transfer (in bytes)', required=True)
    parser_dma.add_argument('--parser_dma', '--num_dim', dest='num_dim', choices=[*range(6)], default=0, help='Number of dimensions [0-5]')

    parser_dma.add_argument('--ss1', '--src_stride_1', dest='src_stride_1', type=int, help='Source stride of the DMA transfer', default=0)
    parser_dma.add_argument('--ds1', '--dst_stride_1', dest='dst_stride_1', type=int, help='Destination stride of the DMA transfer', default=0)
    parser_dma.add_argument('--sds1', '--src_dim_size_1', dest='src_dim_size_1', type=int, help='Source Dimension size of the DMA transfer', default=0)
    parser_dma.add_argument('--dds1', '--dst_dim_size_1', dest='dst_dim_size_1', type=int, help='Destination Dimension size of the DMA transfer', default=0)

    parser_dma.add_argument('--ss2', '--src_stride_2', dest='src_stride_2', type=int, help='Source stride of the DMA transfer', default=0)
    parser_dma.add_argument('--ds2', '--dst_stride_2', dest='dst_stride_2', type=int, help='Destination stride of the DMA transfer', default=0)
    parser_dma.add_argument('--sds2', '--src_dim_size_2', dest='src_dim_size_2', type=int, help='Source Dimension size of the DMA transfer', default=0)
    parser_dma.add_argument('--dds2', '--dst_dim_size_2', dest='dst_dim_size_2', type=int, help='Destination Dimension size of the DMA transfer', default=0)

    parser_dma.add_argument('--ss3', '--src_stride_3', dest='src_stride_3', type=int, help='Source stride of the DMA transfer', default=0)
    parser_dma.add_argument('--ds3', '--dst_stride_3', dest='dst_stride_3', type=int, help='Destination stride of the DMA transfer', default=0)
    parser_dma.add_argument('--sds3', '--src_dim_size_3', dest='src_dim_size_3', type=int, help='Source Dimension size of the DMA transfer', default=0)
    parser_dma.add_argument('--dds3', '--dst_dim_size_3', dest='dst_dim_size_3', type=int, help='Destination Dimension size of the DMA transfer', default=0)

    parser_dma.add_argument('--ss4', '--src_stride_4', dest='src_stride_4', type=int, help='Source stride of the DMA transfer', default=0)
    parser_dma.add_argument('--ds4', '--dst_stride_4', dest='dst_stride_4', type=int, help='Destination stride of the DMA transfer', default=0)
    parser_dma.add_argument('--sds4', '--src_dim_size_4', dest='src_dim_size_4', type=int, help='Source Dimension size of the DMA transfer', default=0)
    parser_dma.add_argument('--dds4', '--dst_dim_size_4', dest='dst_dim_size_4', type=int, help='Destination Dimension size of the DMA transfer', default=0)

    parser_dma.add_argument('--ss5', '--src_stride_5', dest='src_stride_5', type=int, help='Source stride of the DMA transfer', default=0)
    parser_dma.add_argument('--ds5', '--dst_stride_5', dest='dst_stride_5', type=int, help='Destination stride of the DMA transfer', default=0)
    parser_dma.add_argument('--sds5', '--src_dim_size_5', dest='src_dim_size_5', type=int, help='Source Dimension size of the DMA transfer', default=0)
    parser_dma.add_argument('--dds5', '--dst_dim_size_5', dest='dst_dim_size_5', type=int, help='Destination Dimension size of the DMA transfer', default=0)

    parser_dma.add_argument('-p', '--num_engine', dest='num_engine', type=int, help='Number of DMA engines', default=1)
    parser_dma.add_argument('-d', '--direction', dest='direction', type=str, help='Direction of the DMA transfer', required=True,
                            choices=['DDR2CMX', 'CMX2CMX', 'CMX2DDR', 'DDR2DDR'])

    args = parser.parse_args()

    args.device = f"VPUDevice.{args.device}"

    if args.module == 'DPU':
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

        args.operation = f"Operation.{args.operation}"
        args.execution_order = f"ExecutionMode.{args.execution_order}"
        args.activation_function = f"ActivationFunction.NONE"
        args.input_0_datatype = f"DataType.{args.input_0_datatype}"
        args.input_1_datatype = f"DataType.{args.input_1_datatype}"
        args.output_0_datatype = f"DataType.{args.output_0_datatype}"
        args.input_0_layout = f"Layout.ZXY"
        args.input_1_layout = f"Layout.ZXY"
        args.output_0_layout = f"Layout.ZXY"
        args.input_0_swizzling = f"Swizzling.KEY_0"
        args.input_1_swizzling = f"Swizzling.KEY_0"
        args.output_0_swizzling = f"Swizzling.KEY_0"
        isi_dict = {'CLUSTERING':'CLUSTERING','SOH':'SPLIT_OVER_H','SOK':'SPLIT_OVER_K'}
        args.isi_strategy = f"ISIStrategy.{isi_dict[args.isi_strategy]}"
        args.output_write_tiles = 2 if args.isi_strategy=='SOK' else args.output_write_tiles
    return args
