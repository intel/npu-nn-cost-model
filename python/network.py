# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from vpunn import VPUNN_lib
import networkx as nx
import os


class VPUNetwork:
    def __init__(self, nDPUs=1, nSHVs=1, nTiles=1):
        self.__g = nx.DiGraph()
        self.nDPUs = nDPUs
        self.nSHVs = nSHVs
        self.nTiles = nTiles

    def add_layer(self, name, layer):
        strategy = VPUNN_lib.VPULayerStrategy()
        strategy.nDPUs = self.nDPUs
        strategy.nSHVs = self.nSHVs
        strategy.nTiles = self.nTiles

        self.__g.add_node(
            name,
            type=type(layer),
            ref=VPUNN_lib.VPUComputeNode(layer),
            strategy=strategy,
        )

    def add_edge(self, source, sink):
        if source not in self.__g.nodes:
            raise RuntimeError(f"layer {source} not in the graph")

        if sink not in self.__g.nodes:
            raise RuntimeError(f"layer {sink} not in the graph")

        self.__g.add_edge(source, sink)

    def set_tiling_strategy(self, name, strategy):
        self.__g.nodes[name]["strategy"].tiling_strategy = strategy

    def build(self):
        dag = VPUNN_lib.VPUComputationDAG()
        strategy = VPUNN_lib.VPUNetworkStrategy()

        for name in self.__g.nodes:
            dag.addNode(self.__g.nodes[name]["ref"])
            strategy.set(self.__g.nodes[name]["ref"], self.__g.nodes[name]["strategy"])

        for source, sink in self.__g.edges:
            dag.addEdge(self.__g.nodes[source]["ref"], self.__g.nodes[sink]["ref"])

        return dag, strategy


class VPUNetworkCostModel:
    def __init__(self, filename, profile=False, verbose=False):
        if not os.path.isfile(filename):
            print(f"WARNING: file {filename} does not exists")
        self.model = VPUNN_lib.VPUNetworkCostModel(filename, profile, 16384, 1)
        if not self.model.nn_initialized():
            print("WARNING: VPUNN model not initialized... using simplistic model")
        self.verbose = verbose

    def cost(self, network):
        dag, strategy = network.build()
        return self.model.Network(dag, strategy)
