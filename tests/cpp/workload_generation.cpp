// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include "vpu/optimization/workload_optimization.h"

#ifndef VPU_2_7_MODEL_PATH
#define VPU_2_7_MODEL_PATH "../../../models/vpu_2_7.vpunn"
#endif

#ifndef VPU_2_0_MODEL_PATH
#define VPU_2_0_MODEL_PATH "../../../models/vpu_2_0.vpunn"
#endif

static auto model_theoretical = std::make_shared<VPUNN::VPUCostModel>();
static auto model_2_7 = std::make_shared<VPUNN::VPUCostModel>(VPU_2_7_MODEL_PATH);
static auto model_2_0 = std::make_shared<VPUNN::VPUCostModel>(VPU_2_0_MODEL_PATH);

VPUNN::DPULayer generate_helper_layer(const VPUNN::VPUDevice device, const unsigned int dim,
                                      const unsigned int channels, const unsigned int kernel = 1,
                                      const VPUNN::DataType dtype = VPUNN::DataType::FLOAT16) {
    return VPUNN::DPULayer(device,                                            // VPU device
                           VPUNN::Operation::CONVOLUTION,                     // Operation
                           {VPUNN::VPUTensor(dim, dim, channels, 1, dtype)},  // input dimensions
                           //{VPUNN::VPUTensor(dim, dim, channels, 1, dtype)},  // input_1 dimensions ???
                           {VPUNN::VPUTensor(dim, dim, channels, 1, dtype)},  // output dimensions
                           {kernel, kernel},                                  // kernels
                           {1, 1},                                            // strides
                           {kernel / 2, kernel / 2, kernel / 2, kernel / 2}   // padding
    );
}

// std::vector<std::tuple<maxWorkloads, maxLatencyUs, nDPU, runtimeOverheat, VPUNN::VPUOptimizationTarget,
// VPUNN::VPUSplitStrategy, VPUNN::VPUSplitStrategy>>
typedef std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, VPUNN::VPUOptimizationTarget,
                               VPUNN::VPUSplitStrategy, VPUNN::VPUSplitStrategy>>
        list;

list createCombinations() {
    list tuple_list;

    for (const VPUNN::VPUOptimizationTarget target :
         {VPUNN::VPUOptimizationTarget::POWER, VPUNN::VPUOptimizationTarget::LATENCY,
          VPUNN::VPUOptimizationTarget::EFFICIENCY}) {
        for (const VPUNN::VPUSplitStrategy strategy :
             {VPUNN::VPUSplitStrategy::HW_TILING, VPUNN::VPUSplitStrategy::Z_TILING, VPUNN::VPUSplitStrategy::H_TILING,
              VPUNN::VPUSplitStrategy::W_TILING}) {
            tuple_list.push_back(std::make_tuple(1, 1000, 100, 0, target, strategy, strategy));
            tuple_list.push_back(std::make_tuple(64, 1000, 4, 10, target, strategy, strategy));
        }
    }

    return tuple_list;
}

void validate_wl(VPUNN::DPULayer& layer, VPUNN::DPUWorkloads& workloads) {
    // Empty workloads are not valid
    if (workloads.size() == 0) {
        FAIL() << "empty workloads list";
    }

    // Same volume of the original layer
    auto layer_dim = 1;
    auto sum_wl = 0;

    for (auto idx = 0; idx < 4; idx++) {
        layer_dim *= layer.outputs[0].get_shape()[idx];
    }

    for (auto& wl : workloads) {
        auto wl_dim = 1;
        for (auto idx = 0; idx < 4; idx++) {
            wl_dim *= wl.outputs[0].get_shape()[idx];
        }
        sum_wl += wl_dim;
    }
    if (layer_dim != sum_wl) {
        FAIL() << "expect same dimension";
    }

    // Same start and end position

    auto min_x = workloads[0].outputs[0].get_shape()[0];
    auto min_y = workloads[0].outputs[0].get_shape()[1];
    auto min_z = workloads[0].outputs[0].get_shape()[2];

    auto max_x = min_x + workloads[0].offsets[0];
    auto max_y = min_y + workloads[0].offsets[1];
    auto max_z = min_z + workloads[0].offsets[2];

    for (auto& wl : workloads) {
        min_x = std::min(min_x, wl.outputs[0].get_shape()[0]);
        min_y = std::min(min_y, wl.outputs[0].get_shape()[1]);
        min_z = std::min(min_z, wl.outputs[0].get_shape()[2]);

        max_x = std::max(max_x, wl.outputs[0].get_shape()[0] + wl.offsets[0]);
        max_y = std::max(max_y, wl.outputs[0].get_shape()[1] + wl.offsets[1]);
        max_z = std::max(max_z, wl.outputs[0].get_shape()[2] + wl.offsets[2]);
    }

    if ((min_x != 0) && (min_y != 0) && (min_z != 0) &&
        (std::tie(max_x, max_y, max_z) !=
         std::tie(layer.outputs[0].get_shape()[0], layer.outputs[0].get_shape()[1], layer.outputs[0].get_shape()[2]))) {
        FAIL() << "expect same start and end position";
    }

    // No intersection between workloads
    for (long unsigned int idx = 0; idx < workloads.size(); idx++) {
        for (long unsigned int jdx = 0; jdx < workloads.size(); jdx++) {
            if (idx != jdx && workloads[idx].offsets == workloads[jdx].offsets)
                FAIL() << "expect no intersection between workloads";
        }
    }

    // Workloads z dimension must be multiple of 16
    for (auto& wl : workloads) {
        if (wl.outputs[0].get_shape()[2] % 16 != 0)
            FAIL() << "expect multiple of 16";
    }

    // All workloads volume > 0
    for (auto& wl : workloads) {
        auto dim_wl = 1;
        for (auto idx = 0; idx < 4; idx++) {
            dim_wl *= wl.outputs[0].get_shape()[idx];
        }
        if (dim_wl <= 0)
            FAIL() << "expect all workloads volume > 0";
    }
}

TEST(WorkloadGenerationLoadModels, BasicAssertions) {
    EXPECT_EQ(model_theoretical->nn_initialized(), false);
    EXPECT_EQ(model_2_7->nn_initialized(), true);
    EXPECT_EQ(model_2_0->nn_initialized(), true);
}

TEST(WorkloadGenerationCreateTilerMultipleWLs, BasicAssertions) {
    list tuple_list = createCombinations();

    for (const auto& i : tuple_list) {
        VPUNN::SplitOptions options;
        options.nDPU = std::get<2>(i);
        // TODO: remove this once multiple wl split is supported
        options.maxLatencyUs = std::get<1>(i);
        options.maxWorkloads = std::get<0>(i);
        options.runtimeOverhead = std::get<3>(i);
        options.target = std::get<4>(i);
        options.availableStrategies = {std::get<5>(i), std::get<6>(i)};

        for (auto model : {model_theoretical, model_2_0, model_2_7}) {
            // Get the tiler
            auto layer = generate_helper_layer(VPUNN::VPUDevice::VPU_2_0, 56, 64);
            std::unique_ptr<VPUNN::DPUTiler> tiler = VPUNN::getDPUTiler(model);

            if (options.target == VPUNN::VPUOptimizationTarget::POWER)
                EXPECT_THROW(
                        {
                            try {
                                tiler->intraTileSplit(layer, options);
                            } catch (std::runtime_error const& err) {
                                // and this tests that it has the correct message
                                EXPECT_STREQ("intraTileSplit: no valid workload generated", err.what());
                                throw;
                            } catch (...) {
                                FAIL() << "Expected std::runtime_error";
                            }
                        },
                        std::runtime_error);
            else {
                // Split the layer into multiple workloads
                auto workloads = tiler->intraTileSplit(layer, options);
                // Validate workloads
                validate_wl(layer, workloads);
            }
        }
    }
}
