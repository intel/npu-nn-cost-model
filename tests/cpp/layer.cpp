// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "vpu_layer_cost_model.h"

#ifndef VPU_2_7_MODEL_PATH
#define VPU_2_7_MODEL_PATH "../../../models/vpu_2_7.vpunn"
#endif

#ifndef VPU_2_0_MODEL_PATH
#define VPU_2_0_MODEL_PATH "../../../models/vpu_2_0.vpunn"
#endif

static auto model = VPUNN::VPULayerCostModel();
static auto model_2_7 = VPUNN::VPULayerCostModel(VPU_2_7_MODEL_PATH);
static auto model_2_0 = VPUNN::VPULayerCostModel(VPU_2_0_MODEL_PATH);

TEST(LayerLoadModels, BasicAssertions) {
    EXPECT_EQ(model_2_7.nn_initialized(), true);
    EXPECT_EQ(model_2_0.nn_initialized(), true);
}

VPUNN::DPULayer generate_helper_layer(const unsigned int dim, const unsigned int channels) {
    return VPUNN::DPULayer(
            VPUNN::VPUDevice::VPU_2_0,                                            // VPU device
            VPUNN::Operation::CONVOLUTION,                                        // Operation
            {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            //{VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // input_1 dimensions ??
            {VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                               // kernels
            {1, 1},                                                               // strides
            {1, 1, 1, 1}                                                          // padding
    );
}

VPUNN::SHVSigmoid generate_helper_sw_layer(const unsigned int dim, const unsigned int channels) {
    return VPUNN::SHVSigmoid(VPUNN::VPUDevice::VPU_2_0,                                          // VPU device
                             VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16),  // Input tensor
                             VPUNN::VPUTensor(dim, dim, channels, 1, VPUNN::DataType::FLOAT16)   // Output tensor
    );
}

TEST(SplitAcrossTileSOH, BasicAssertions) {
    auto wl = generate_helper_layer(16, 64);

    auto SOH_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 1);
    auto SOH_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 2);
    auto SOH_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOH, 4);

    // Basic expectations
    EXPECT_EQ(SOH_single_tile.size(), 1);
    EXPECT_EQ(SOH_two_tile.size(), 2);
    EXPECT_EQ(SOH_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOH_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOH_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOH_two_tile[0].outputs[0].get_shape()[1] * 2, wl.outputs[0].get_shape()[1]);

    EXPECT_EQ(SOH_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOH_four_tile[0].outputs[0].get_shape()[1] * 4, wl.outputs[0].get_shape()[1]);
}

TEST(SplitAcrossTileSOK, BasicAssertions) {
    auto wl = generate_helper_layer(16, 64);

    auto SOK_single_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 1);
    auto SOK_two_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 2);
    auto SOK_four_tile = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 4);

    // Basic expectations
    EXPECT_EQ(SOK_single_tile.size(), 1);
    EXPECT_EQ(SOK_two_tile.size(), 2);
    EXPECT_EQ(SOK_four_tile.size(), 4);

    // The shape of the single split must be equal to the origial layer
    EXPECT_EQ(SOK_single_tile[0].outputs[0].get_shape(), wl.outputs[0].get_shape());

    EXPECT_EQ(SOK_two_tile[0].outputs[0].size(), wl.outputs[0].size() / 2);
    EXPECT_EQ(SOK_two_tile[0].outputs[0].get_shape()[2] * 2, wl.outputs[0].get_shape()[2]);

    EXPECT_EQ(SOK_four_tile[0].outputs[0].size(), wl.outputs[0].size() / 4);
    EXPECT_EQ(SOK_four_tile[0].outputs[0].get_shape()[2] * 4, wl.outputs[0].get_shape()[2]);
}

TEST(SplitAcrossTileSOKAsymmetric, BasicAssertions) {
    auto wl = generate_helper_layer(16, 64);

    auto SOK_asymmetric = wl.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 5);

    // Basic expectations
    EXPECT_EQ(SOK_asymmetric.size(), 4);

    for (unsigned int idx = 0; idx < SOK_asymmetric.size(); idx++) {
        EXPECT_EQ(SOK_asymmetric[idx].outputs[0].get_shape()[2], 16u);
    }

    auto wl_2 = generate_helper_layer(16, 48);

    auto SOK_asymmetric_2 = wl_2.splitAcrossTiles(VPUNN::VPUTilingStrategy::SOK, 5);

    // Basic expectations
    EXPECT_EQ(SOK_asymmetric_2.size(), 3);

    for (unsigned int idx = 0; idx < SOK_asymmetric_2.size(); idx++) {
        EXPECT_EQ(SOK_asymmetric_2[idx].outputs[0].get_shape()[2], 16u);
    }
}

TEST(LayerCostModelVPU_2_0, BasicAssertions) {
    auto layer = generate_helper_layer(16, 64);
    auto layer_cost = model_2_0.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(layer_cost, 0u);
}

TEST(LayerCostModelVPU_2_7_shv, BasicAssertions) {
    auto layer = generate_helper_sw_layer(16, 64);
    auto layer_cost = model_2_7.Layer(layer, 5, 4);

    // Basic expectations
    EXPECT_GT(layer_cost, 0u);
}
