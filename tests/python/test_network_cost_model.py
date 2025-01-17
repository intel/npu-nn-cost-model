# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import os
from vpunn import VPUNN_lib
from vpunn.network import VPUNetwork, VPUNetworkCostModel
import pytest


model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../models/vpu_2_7.vpunn"
)
model = VPUNetworkCostModel(model_path)
config = {
    "1": {"channels": 32, "activation": 128},
    "2": {"channels": 1024, "activation": 7},
}


def generate_dpu_layer(activation=56, channels=64):
    return VPUNN_lib.DPULayer(
        VPUNN_lib.VPUDevice.VPU_2_7,
        VPUNN_lib.Operation.CONVOLUTION,
        [
            VPUNN_lib.VPUTensor(
                [activation, activation, channels, 1], VPUNN_lib.DataType.UINT8
            )
        ],
        [
            VPUNN_lib.VPUTensor(
                [activation, activation, channels, 1], VPUNN_lib.DataType.UINT8
            )
        ],
        [1, 1],
        [1, 1],
        [0, 0, 0, 0],
    )


def generate_shv_layer(activation=56, channels=64):
    return VPUNN_lib.SHVSigmoid(
        VPUNN_lib.VPUDevice.VPU_2_7,
        VPUNN_lib.VPUTensor(
            [activation, activation, channels, 1], VPUNN_lib.DataType.UINT8
        ),
        VPUNN_lib.VPUTensor(
            [activation, activation, channels, 1], VPUNN_lib.DataType.UINT8
        ),
    )


def generate_network(n_layers, n_tiles, node_type, edge_pattern, tiling_strategy):
    network = VPUNetwork(nTiles=n_tiles)

    for idx in range(n_layers):
        layer = generate_dpu_layer() if node_type == "dpu" else generate_shv_layer()
        network.add_layer(f"layer{idx}", layer)

    for idx in range(n_layers - 1):
        network.add_edge(f"layer{idx}", f"layer{idx + 1}")
        network.set_tiling_strategy(f"layer{idx}", tiling_strategy)

    if edge_pattern == "residual":
        for idx in range(n_layers - 2):
            network.add_edge(f"layer{idx}", f"layer{idx + 2}")

    return network


def generate_simple_network(n_layers, edge_pattern, tiling_strategy, config_index):
    config = {
        "1": {"channels": 32, "activation": 128},
        "2": {"channels": 1024, "activation": 7},
    }
    channels = config[str(config_index)]["channels"]
    activation = config[str(config_index)]["activation"]

    network = VPUNetwork(nDPUs=1, nSHVs=2, nTiles=2)

    for idx in range(n_layers):
        layer = generate_dpu_layer(activation, channels)
        network.add_layer(f"layer{idx}", layer)
        network.set_tiling_strategy(f"layer{idx}", tiling_strategy)

    for idx in range(n_layers - 1):
        network.add_edge(f"layer{idx}", f"layer{idx + 1}")

    if edge_pattern == "residual":
        for idx in range(n_layers - 2):
            network.add_edge(f"layer{idx}", f"layer{idx + 2}")

    return network


@pytest.mark.parametrize("nodes", [1, 10, 20, 50, 100])
@pytest.mark.parametrize("n_tiles", [1, 2])
@pytest.mark.parametrize("node_type", ["dpu", "shv"])
@pytest.mark.parametrize("edge_pattern", ["linear", "residual"])
@pytest.mark.parametrize(
    "tiling_strategy",
    [
        VPUNN_lib.VPUTilingStrategy.NONE,
        VPUNN_lib.VPUTilingStrategy.SOH_Overlapped,
        VPUNN_lib.VPUTilingStrategy.SOK,
    ],
)
def test_compute_cycles(nodes, n_tiles, node_type, edge_pattern, tiling_strategy):
    network = generate_network(nodes, n_tiles, node_type, edge_pattern, tiling_strategy)

    assert model.cost(network) > 0


# @pytest.mark.parametrize("nodes", [3, 5])
# @pytest.mark.parametrize("edge_pattern", ["linear", "residual"])
# def test_cost_comparsion(nodes, edge_pattern):
#     network_soh_1 = generate_simple_network(
#         nodes, edge_pattern, VPUNN_lib.VPUTilingStrategy.SOH, 1
#     )
#     network_sok_1 = generate_simple_network(
#         nodes, edge_pattern, VPUNN_lib.VPUTilingStrategy.SOK, 1
#     )

#     network_soh_2 = generate_simple_network(
#         nodes, edge_pattern, VPUNN_lib.VPUTilingStrategy.SOH, 2
#     )
#     network_sok_2 = generate_simple_network(
#         nodes, edge_pattern, VPUNN_lib.VPUTilingStrategy.SOK, 2
#     )
    
#     assert model.cost(network_soh_1) < model.cost(network_sok_1)
#     assert model.cost(network_soh_2) > model.cost(network_sok_2)
