// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "vpu_network_cost_model.h"

#ifndef VPU_2_7_MODEL_PATH
#define VPU_2_7_MODEL_PATH "../../../models/vpu_2_7.vpunn"
#endif

#ifndef VPU_2_0_MODEL_PATH
#define VPU_2_0_MODEL_PATH "../../../models/vpu_2_0.vpunn"
#endif

static auto model = VPUNN::VPUNetworkCostModel();
static auto model_2_7 = VPUNN::VPUNetworkCostModel(VPU_2_7_MODEL_PATH);
static auto model_2_0 = VPUNN::VPUNetworkCostModel(VPU_2_0_MODEL_PATH);

TEST(NetworkLoadModels, BasicAssertions) {
    EXPECT_EQ(model_2_7.nn_initialized(), true);
    EXPECT_EQ(model_2_0.nn_initialized(), true);
}

std::shared_ptr<VPUNN::SWOperation> generate_helper_shv_layer(const unsigned int dim, const unsigned int channels) {
    return std::make_shared<VPUNN::SHVSigmoid>(VPUNN::VPUDevice::VPU_2_0,
                                               VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16),
                                               VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16));
}

std::shared_ptr<VPUNN::DPULayer> generate_helper_dpu_layer(const unsigned int dim, const unsigned int channels) {
    auto inputs = std::array<VPUNN::VPUTensor, 1>({VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)});
    // auto inputs_1 =
    //         std::array<VPUNN::VPUTensor, 1>({VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)});
    auto outputs = std::array<VPUNN::VPUTensor, 1>({VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)});
    auto kernels = std::array<unsigned int, 2>({1, 1});
    auto strides = std::array<unsigned int, 2>({1, 1});
    auto padding = std::array<unsigned int, 4>({0, 0, 0, 0});
    return std::make_shared<VPUNN::DPULayer>(VPUNN::VPUDevice::VPU_2_0, VPUNN::Operation::CONVOLUTION,
                                             inputs, /*inputs_1,*/
                                             outputs, kernels, strides, padding);
}

TEST(SmokeTestVPUComputeNode, BasicAssertions) {
    auto dpu_node = VPUNN::VPUComputeNode(generate_helper_dpu_layer(32, 64));
    EXPECT_EQ(dpu_node.type, VPUNN::VPUComputeNode::OpType::DPU_COMPUTE_NODE);

    auto shv_node = VPUNN::VPUComputeNode(generate_helper_shv_layer(32, 64));
    EXPECT_EQ(shv_node.type, VPUNN::VPUComputeNode::OpType::SHV_COMPUTE_NODE);
}

TEST(SmokeTestVPUComputationDAG, BasicAssertions) {
    // A list of Layers
    std::vector<VPUNN::VPUComputeNodePtr> layers = {
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64)),
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64)),
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64)),
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64))};

    auto dag = VPUNN::VPUComputationDAG();
    for (auto node : layers) {
        dag.addNode(node);
    }

    for (unsigned int idx = 0; idx < layers.size() - 1; idx++) {
        dag.addEdge(layers[idx], layers[idx + 1]);
    }

    EXPECT_EQ(dag.nodes(), layers.size());
    EXPECT_EQ(dag.edges(), layers.size() - 1);
    EXPECT_EQ(dag.sources().size(), 1);
}

VPUNN::VPUComputationDAG generate_helper_dag() {
    std::vector<VPUNN::VPUComputeNodePtr> layers = {
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64)),
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64)),
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64)),
            std::make_shared<VPUNN::VPUComputeNode>(generate_helper_shv_layer(32, 64))};

    auto dag = VPUNN::VPUComputationDAG();
    for (auto& node : layers) {
        dag.addNode(node);
    }

    for (unsigned int idx = 0; idx < layers.size() - 1; idx++) {
        dag.addEdge(layers[idx], layers[idx + 1]);
    }

    return dag;
}

TEST(SmokeTestNetworkCostModelFailure, BasicAssertions) {
    // Generate a random DAG
    auto dag = generate_helper_dag();
    VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy> strategy = {};

    EXPECT_THROW(
            {
                try {
                    model.Network(dag, strategy);
                } catch (const std::runtime_error& e) {
                    // and this tests that it has the correct message
                    EXPECT_STREQ("Impossible to find a strategy for a layer", e.what());
                    throw;
                }
            },
            std::runtime_error);
}

TEST(SmokeTestNetworkCostModel, BasicAssertions) {
    // Generate a random DAG
    auto dag = generate_helper_dag();
    const VPUNN::VPULayerStrategy basic_strategy{1, 1, 1, VPUNN::VPUTilingStrategy::NONE, false, false};

    VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy> strategy = {};
    for (auto layer : dag) {
        strategy[layer] = basic_strategy;
    }
    const unsigned long int cost = model.Network(dag, strategy);
    // Naive cost (no spilling, or fancy strategy)
    const unsigned long int naive_cost =
            static_cast<unsigned int>(dag.nodes()) * model.SHAVE(*generate_helper_shv_layer(32, 64));

    EXPECT_GT(cost, 0u);
    // The cost of this simple dag is the same as the
    EXPECT_EQ(cost, naive_cost);

    {  // all layers are spilling => slower
        const VPUNN::VPULayerStrategy extra_strategy{1, 1, 1, VPUNN::VPUTilingStrategy::NONE, true, true};
        VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy> complex_strategy = {};
        for (auto layer : dag) {
            complex_strategy[layer] = extra_strategy;
        }
        unsigned long int complex_cost = model.Network(dag, complex_strategy);
        EXPECT_GE(complex_cost, naive_cost);
    }
    {  // no spilling , but 2 shave/dpu=> faster
        const VPUNN::VPULayerStrategy extra_strategy{1, 2, 1, VPUNN::VPUTilingStrategy::NONE, false, false};
        VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy> complex_strategy = {};
        for (auto layer : dag) {
            complex_strategy[layer] = extra_strategy;
        }
        unsigned long int complex_cost = model.Network(dag, complex_strategy);
        EXPECT_LE(complex_cost, naive_cost);
    }
}

TEST(StressTestNetworkCostModel, BasicAssertions) {
    unsigned long int old_cost = 0;
    for (int idx = 0; idx < 100; idx++) {
        // Generate a random DAG
        auto dag = generate_helper_dag();
        VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy> strategy = {};
        for (auto layer : dag) {
            strategy[layer] = {1, 1, 1, VPUNN::VPUTilingStrategy::NONE, false, false};
        }
        unsigned long int cost = model.Network(dag, strategy);
        if (idx > 0) {
            EXPECT_EQ(cost, old_cost);
        }
        old_cost = cost;
    }
}
