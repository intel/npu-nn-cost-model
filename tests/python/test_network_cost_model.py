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
import pytest


model = VPUNetworkCostModel("model_path")


def generate_dpu_layer():
    return VPUNN.DPULayer(
        VPUNN.VPUDevice.VPU_2_7,
        VPUNN.Operation.CONVOLUTION,  # Device and op type
        [VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8)],  # Input tensor
        # [VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8)],  # Input_1 tensor ??
        [VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8)],  # Output tensor
        [1, 1],
        [1, 1],
        [0, 0, 0, 0],  # Kernel, strides, padding
    )


def generate_shv_layer():
    return VPUNN.SHVSigmoid(
        VPUNN.VPUDevice.VPU_2_7,
        VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8),  # Input tensor
        VPUNN.VPUTensor([56, 56, 64, 1], VPUNN.DataType.UINT8),  # Output tensor
    )


def generate_network(n_layers, n_tiles, node_type, edge_pattern, tiling_strategy):
    network = VPUNetwork(nTiles=n_tiles)

    for idx in range(0, n_layers):
        layer = generate_dpu_layer() if node_type == "dpu" else generate_shv_layer()
        network.add_layer(f"layer{idx}", layer)

    for idx in range(0, n_layers - 1):
        network.add_edge(f"layer{idx}", f"layer{idx + 1}")
        network.set_tiling_strategy(f"layer{idx}", tiling_strategy)

    if edge_pattern == "residual":
        for idx in range(0, n_layers - 2):
            network.add_edge(f"layer{idx}", f"layer{idx + 2}")

    return network


@pytest.mark.parametrize("nodes", [1, 10, 20, 50, 100])
@pytest.mark.parametrize("n_tiles", [1, 2])
@pytest.mark.parametrize("node_type", ["dpu", "shv"])
@pytest.mark.parametrize("edge_pattern", ["linear", "residual"])
@pytest.mark.parametrize(
    "tiling_strategy",
    [
        VPUNN.VPUTilingStrategy.NONE,
        VPUNN.VPUTilingStrategy.SOH,
        VPUNN.VPUTilingStrategy.SOK,
    ],
)
def test_compute_cycles(nodes, n_tiles, node_type, edge_pattern, tiling_strategy):
    network = generate_network(nodes, n_tiles, node_type, edge_pattern, tiling_strategy)

    assert model.cost(network) > 0
