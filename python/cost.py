# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import os
import argparse
import numpy as np
from vpunn.parse_args import define_and_parse_args
from vpunn.model_input import str2enum, generate_model_input
from vpunn import VPUNN_lib

def main():

    args = define_and_parse_args()

    if args.device=='VPUDevice.VPU_2_7':
        if args.module == "DPU":
            if not os.path.exists(args.model):
                args.model = f'{os.path.dirname(__file__)}/models/vpu_2_7.vpunn'
            model = VPUNN_lib.VPUCostModel(args.model)

            workload = generate_model_input(args)
            print(workload)

            result = model.DPUMsg(workload)
            print(f"DPU execution cycles: {result}")
        elif args.module == "DMA":
            if not os.path.exists(args.model):
                args.model = f'{os.path.dirname(__file__)}/models/dma_2_7.vpunn'
            model = VPUNN_lib.DMACostModel_VPUNN_DMANNWorkload_NPU27_t(args.model)

            workload = generate_model_input(args, VPUNN_lib.DMANNWorkload_NPU27)
            print(workload)
            print(model.getDmaDescriptor(workload))

            size_div_cycle_perc = model.computeBandwidthMsg(workload)[0]
            size_div_cycle = (size_div_cycle_perc*32)
            size = (args.num_planes+1) * args.length
            cycles = size/size_div_cycle

            print(f"DMA cycles report:\n\tLatency (VPU clock cycles): {int(cycles):,} cc\n\tSize: {int(size):,} Bytes\n\tSize/Latency: {size_div_cycle:.3} B/cc\n\tEfficency (vs Ideal): {size_div_cycle_perc*100:.3}%")
    elif args.device=='VPUDevice.VPU_4_0':
        if args.module == "DPU":
            if not os.path.exists(args.model):
                args.model = f'{os.path.dirname(__file__)}/models/vpu_4_0.vpunn'
            model = VPUNN_lib.VPUCostModel(args.model)

            workload = generate_model_input(args)
            print(workload)

            result = model.DPUMsg(workload)
            print(f"DPU execution cycles: {result}")
        else:
            if not os.path.exists(args.model):
                args.model = f'{os.path.dirname(__file__)}/models/dma_4_0.vpunn'

            model = VPUNN_lib.DMACostModel_VPUNN_DMANNWorkload_NPU40_t(args.model)

            workload = generate_model_input(args, VPUNN_lib.DMANNWorkload_NPU40)
            print(workload)

            bw, error_msg = model.computeBandwidthMsg(workload)
            cycles, error_msg = model.computeCyclesMsg(workload)
            if error_msg != '':
                print(error_msg)
            elif cycles > 4e9:
                print("Generic Error")
            else:
                print(f"DMA report:\n\tLatency (DPU clock cycles): {int(cycles):,} cc\n\tBandwidth: {bw*100:.3}%")
    else:
        raise NotImplementedError

if __name__ == "__main__":
    # Running the main function
    main()
