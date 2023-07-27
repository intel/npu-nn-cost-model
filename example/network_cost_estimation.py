# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import os
import argparse
from vpunn import VPUNN_lib
from vpunn.network import VPUNetwork, VPUNetworkCostModel

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../models/vpu_2_7.vpunn"
)
model = VPUNetworkCostModel(model_path)
config = {
    "1": {"channels": 32, "activation": 128},
    "2": {"channels": 1024, "activation": 7},
}


def generate_shv_layer(in_channels, out_channels, activation):
    return VPUNN_lib.SHVSigmoid(
        VPUNN_lib.VPUDevice.VPU_2_7,
        VPUNN_lib.VPUTensor(
            [in_channels, in_channels, activation, 1], VPUNN_lib.DataType.UINT8
        ),
        VPUNN_lib.VPUTensor(
            [out_channels, out_channels, activation, 1], VPUNN_lib.DataType.UINT8
        ),
    )


def generate_dpu_layer(op_type, in_channels, out_channels, activation, kernel_size):
    kernel = [kernel_size] * 2
    strides = [1, 1]
    padding = [0, 0, 0, 0]
    if "convolution" in op_type:
        operation = VPUNN_lib.Operation.CONVOLUTION
    elif "elt" in op_type:
        operation = VPUNN_lib.Operation.ELTWISE
    input_tensor = [
        VPUNN_lib.VPUTensor(
            [activation, activation, in_channels, 1], VPUNN_lib.DataType.UINT8
        )
    ]
    output_tensor = [
        VPUNN_lib.VPUTensor(
            [activation, activation, out_channels, 1], VPUNN_lib.DataType.UINT8
        )
    ]

    return VPUNN_lib.DPULayer(
        VPUNN_lib.VPUDevice.VPU_2_7,
        operation,
        input_tensor,
        output_tensor,
        kernel,
        strides,
        padding,
    )


def generate_linear_network(n_layers, tiling_strategy):
    network = VPUNetwork(nDPUs=1, nSHVs=2, nTiles=2)

    for idx in range(n_layers):
        if idx % 2:
            layer = generate_dpu_layer("convolution", 56, 56, 64, 1)
        else:
            layer = generate_shv_layer(56, 56, 64)
        network.add_layer(f"layer{idx}", layer)
        network.set_tiling_strategy(f"layer{idx}", tiling_strategy)

    for idx in range(n_layers - 1):
        network.add_edge(f"layer{idx}", f"layer{idx + 1}")

    return network


def generate_simple_network(n_layers, tiling_strategy, config_index):
    network = VPUNetwork(nDPUs=1, nSHVs=2, nTiles=2)

    in_channels = config[str(config_index)]["channels"]
    out_channels = config[str(config_index)]["channels"]
    activation = config[str(config_index)]["activation"]

    for idx in range(n_layers):
        layer = generate_dpu_layer(
            "convolution", in_channels, out_channels, activation, 1
        )
        network.add_layer(f"layer{idx}", layer)
        network.set_tiling_strategy(f"layer{idx}", tiling_strategy)

    for idx in range(n_layers - 1):
        network.add_edge(f"layer{idx}", f"layer{idx + 1}")

    return network


def generate_bottlenek_network(tiling_strategy, config_index):
    network = VPUNetwork(nDPUs=1, nSHVs=2, nTiles=2)

    activation = config[str(config_index)]["activation"]
    in_channels = config[str(config_index)]["channels"]
    out_channels = config[str(config_index)]["channels"]

    # input elt
    layer = generate_dpu_layer("elt_wise", in_channels, out_channels, activation, 1)
    network.add_layer("input", layer)
    network.set_tiling_strategy("input", tiling_strategy)

    # layer 1
    layer = generate_dpu_layer(
        "convolution", in_channels, int(out_channels / 4), activation, 1
    )
    network.add_layer("layer_1", layer)
    network.set_tiling_strategy("layer_1", tiling_strategy)

    # layer 2
    layer = generate_dpu_layer(
        "convolution", int(in_channels / 4), int(out_channels / 4), activation, 3
    )
    network.add_layer("layer_2", layer)
    network.set_tiling_strategy("layer_2", tiling_strategy)

    # layer 3
    layer = generate_dpu_layer(
        "convolution", int(in_channels / 4), out_channels, activation, 1
    )
    network.add_layer("layer_3", layer)
    network.set_tiling_strategy("layer_3", tiling_strategy)

    # output elt
    layer = generate_dpu_layer("elt_wise", in_channels, out_channels, activation, 1)
    network.add_layer("output", layer)
    network.set_tiling_strategy("output", tiling_strategy)

    # edges
    network.add_edge("input", "layer_1")
    network.add_edge("layer_1", "layer_2")
    network.add_edge("layer_2", "layer_3")
    network.add_edge("layer_3", "output")
    network.add_edge("input", "output")

    return network


def generate_eltwise_network(tiling_strategy, config_index):
    network = VPUNetwork(nDPUs=1, nSHVs=2, nTiles=2)

    activation = config[str(config_index)]["activation"]
    in_channels = config[str(config_index)]["channels"]
    out_channels = config[str(config_index)]["channels"]

    # input elt
    layer = generate_dpu_layer("elt_wise", in_channels, out_channels, activation, 1)
    network.add_layer("input", layer)
    network.set_tiling_strategy("input", tiling_strategy)

    # layer 1
    layer = generate_dpu_layer("convolution", in_channels, out_channels, activation, 1)
    network.add_layer("layer_1", layer)
    network.set_tiling_strategy("layer_1", tiling_strategy)

    # layer 2
    layer = generate_dpu_layer("convolution", in_channels, out_channels, activation, 1)
    network.add_layer("layer_2", layer)
    network.set_tiling_strategy("layer_2", tiling_strategy)

    # elt
    layer = generate_dpu_layer("elt_wise", in_channels, out_channels, activation, 1)
    network.add_layer("elt", layer)
    network.set_tiling_strategy("elt", tiling_strategy)

    # layer 3
    layer = generate_dpu_layer("convolution", in_channels, out_channels, activation, 1)
    network.add_layer("layer_3", layer)
    network.set_tiling_strategy("layer_3", tiling_strategy)

    # layer 4
    layer = generate_dpu_layer("convolution", in_channels, out_channels, activation, 1)
    network.add_layer("layer_4", layer)
    network.set_tiling_strategy("layer_4", tiling_strategy)

    # output elt
    layer = generate_dpu_layer("elt_wise", in_channels, out_channels, activation, 1)
    network.add_layer("output", layer)
    network.set_tiling_strategy("output", tiling_strategy)

    # edges
    network.add_edge("input", "layer_1")
    network.add_edge("input", "layer_2")
    network.add_edge("layer_1", "elt")
    network.add_edge("layer_2", "elt")
    network.add_edge("elt", "layer_3")
    network.add_edge("elt", "layer_4")
    network.add_edge("layer_3", "output")
    network.add_edge("layer_4", "output")

    return network


def to_vpunn_strategy_name(strategy_name):
    if strategy_name in ["SplitOverH", "SoH", "SOH"]:
        return VPUNN_lib.VPUTilingStrategy.SOH
    elif strategy_name in ["SplitOverK", "SoK", "SOK"]:
        return VPUNN_lib.VPUTilingStrategy.SOK
    elif strategy_name in ["Clustering", "Clustering".upper(), "Clustering".lower()]:
        return VPUNN_lib.VPUTilingStrategy.clustering
    else:
        return VPUNN_lib.VPUTilingStrategy.NONE


def compute_cycles(network_type, strategy, config_index):
    if network_type == "simple":
        network = generate_simple_network(3, strategy, config_index)
    elif network_type == "linear":
        network = generate_linear_network(3, strategy)
    elif network_type == "bottleneck":
        network = generate_bottlenek_network(strategy, config_index)
    elif network_type == "eltwise":
        network = generate_eltwise_network(strategy, config_index)
    return model.cost(network)


def main(args):
    strategy = to_vpunn_strategy_name(args.strategy)
    cycles = compute_cycles(args.network_type, strategy, args.config_index)
    print(f"Network cost: {cycles} cycles for {args.strategy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="network cost script")
    parser.add_argument(
        "--network_type", type=str, default="linear", help="network_type"
    )
    parser.add_argument("--strategy", type=str, default="SOH", help="strategy")
    parser.add_argument("--config_index", type=int, default=1, help="config_index")

    args = parser.parse_args()
    main(args)
