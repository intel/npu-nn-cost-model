# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn import VPUNN
from vpunn.network import VPUNetwork, VPUNetworkCostModel
import os


def generate_linear_network(n_layers, tiling_strategy=VPUNN.VPUTilingStrategy.SOH):
    network = VPUNetwork(nDPUs=1, nSHVs=2, nTiles=2)

    for idx in range(0, n_layers):
        if idx % 2:
            layer = VPUNN.DPULayer(
                VPUNN.VPUDevice.VPU_2_7,
                VPUNN.Operation.CONVOLUTION,  # Device and op type
                [
                    VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8)
                ],  # Input tensor
                [
                    VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8)
                ],  # Output tensor
                [1, 1],
                [1, 1],
                [0, 0, 0, 0],  # Kernel, strides, padding
            )
        else:
            layer = VPUNN.SHVSigmoid(
                VPUNN.VPUDevice.VPU_2_7,
                VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8),  # Input tensor
                VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8),  # Output tensor
            )

        network.add_layer(f"layer{idx}", layer)
        network.set_tiling_strategy(f"layer{idx}", tiling_strategy)

    for idx in range(0, n_layers - 1):
        network.add_edge(f"layer{idx}", f"layer{idx + 1}")

    return network


def compute_cycles():
    network = generate_linear_network(10)

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../models/vpu_2_7.vpunn",
    )

    return VPUNetworkCostModel(model_path).cost(network)


if __name__ == "__main__":

    cycles = compute_cycles()

    print(f"Network cost: {cycles} cycles")
