// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the ?Software Package?)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the ?third-party-programs.txt? or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "core/logger.h"
#include "vpu/power.h"
#include "vpu/types.h"
#include "vpu_cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
class ActivityFactor : public testing::Test {
public:
protected:
    void SetUp() override {
    }

    VPUNN::VPUCostModel model = VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH);

    VPUNN::DPUWorkload generate_helper_layer(const unsigned int dim, const unsigned int ic, const unsigned int oc,
                                             VPUNN::Operation operation, VPUNN::VPUDevice device,
                                             VPUNN::DataType dtype) {
        VPUNN::DPUWorkload wl;
        wl.device = device;
        wl.op = operation;
        wl.execution_order = VPUNN::ExecutionMode::CUBOID_8x16;   // execution mode
        wl.inputs = {VPUNN::VPUTensor(dim, dim, ic, 1, dtype)};   // input dimensions
        wl.outputs = {VPUNN::VPUTensor(dim, dim, oc, 1, dtype)};  // output dimensions
        wl.kernels = {1, 1};                                      // kernels
        wl.strides = {1, 1};                                      // strides
        wl.padding = {0, 0, 0, 0};                                // padding
        wl.input_swizzling[0] = VPUNN::Swizzling::KEY_5;
        wl.input_swizzling[1] = VPUNN::Swizzling::KEY_5;
        wl.output_swizzling[0] = VPUNN::Swizzling::KEY_5;
        return wl;
    }
};

TEST_F(ActivityFactor, TestPowerActivityFactorU8Conv) {
    // Workloads with HxW 16x16 were profiled for a small sweep of input channels
    std::list<unsigned int> ic = {16, 128, 256, 512, 1024, 2048};
    // Expected power activity factors based on implementation of LUT against profiled workloads
    std::list<float> expected = {0.3749f, 0.7148f, 0.7076f, 0.7110f, 0.7090f, 0.7091f};

    auto i = ic.begin();
    auto j = expected.begin();

    for (; i != ic.end() && j != expected.end(); ++i, ++j) {
        auto wl = generate_helper_layer(16, *i, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::UINT8);
        EXPECT_NEAR(model.DPUActivityFactor(wl), *j, 0.05f);
    }

    // Following sweep of input channels (HxW 16x16 again) were profiled but are not explicit
    // entries on the LUT, the expect values are check with relaxed tolerance. This is a
    // weak test for the LUT entry interpolation functionality
    ic = {32, 64};
    expected = {0.4271f, 0.6356f};

    i = ic.begin();
    j = expected.begin();

    for (; i != ic.end() && j != expected.end(); ++i, ++j) {
        auto wl = generate_helper_layer(16, *i, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::UINT8);
        EXPECT_NEAR(model.DPUActivityFactor(wl), *j, 0.25f);  // Relaxing tolerance
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorFPConv) {
    // Workloads with HxW 16x16 were profiled for a tiny sweep of input channels
    std::list<unsigned int> ic = {64, 256};
    // Expected power activity factors based on implementation of LUT against profiled workloads
    std::list<float> expected = {0.8362f, 0.9222f};

    auto i = ic.begin();
    auto j = expected.begin();

    for (; i != ic.end() && j != expected.end(); ++i, ++j) {
        auto wl = generate_helper_layer(16, *i, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
        EXPECT_NEAR(model.DPUActivityFactor(wl), *j, 0.05f);
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorConv576) {
    // IC = 576 not in LUT, results interpolated, expected values derived empirically
    std::list<VPUNN::DataType> dtypes = {VPUNN::DataType::FLOAT16, VPUNN::DataType::UINT8};
    std::list<float> expected = {0.98f, 0.75f};
    auto i = dtypes.begin();
    auto j = expected.begin();

    for (; i != dtypes.end() && j != expected.end(); ++i, ++j) {
        auto wl = generate_helper_layer(16, 576, 576, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7, *i);
        EXPECT_NEAR(model.DPUActivityFactor(wl), *j, 0.05f);
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorConvMisc) {
    // Sweep over some tensor dimensions which aren't in the profiled set
    std::list<unsigned int> dims = {4, 8, 16};
    std::list<unsigned int> ic = {112, 288, 400};
    std::list<VPUNN::DataType> dtype = {VPUNN::DataType::UINT8, VPUNN::DataType::FLOAT16};

    auto i = dims.begin();
    auto j = ic.begin();
    auto k = dtype.begin();

    for (; i != dims.end(); ++i) {
        for (; j != ic.end(); ++j) {
            for (; k != dtype.end(); ++k) {
                auto wl = generate_helper_layer(*i, *j, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                                *k);
                float af = model.DPUActivityFactor(wl);
                // Just range check
                EXPECT_GT(af, 0.2f);
                EXPECT_LT(af, 0.9f);
            }
        }
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorDwConv) {
    // Only one DW workload profiled 64x28x28x64 3x3
    VPUNN::DPUWorkload wl;

    wl = generate_helper_layer(28, 64, 64, VPUNN::Operation::DW_CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                               VPUNN::DataType::UINT8);
    wl.kernels = {3, 3};
    wl.padding = {1, 1, 1, 1};
    EXPECT_NEAR(model.DPUActivityFactor(wl), 0.4902, 0.005f);

    // Only one AVEPOOL (which is implemented as DW) workload profiled 2048x7x7x2048 7x7
    wl = generate_helper_layer(7, 64, 64,  // Max workload size is 64
                               VPUNN::Operation::AVEPOOL, VPUNN::VPUDevice::VPU_2_7, VPUNN::DataType::UINT8);
    // Override specific fields to correctly configure AVEPOOL workload
    wl.kernels = {7, 7};
    wl.outputs = {VPUNN::VPUTensor(1, 1, 64, 1, VPUNN::DataType::UINT8)};
    EXPECT_NEAR(model.DPUActivityFactor(wl), 0.1683, 0.005f);
}

TEST_F(ActivityFactor, TestPowerActivityFactorEltWise) {
    // Only one ELTWISE workload profiled 256x56x56x256
    VPUNN::DPUWorkload wl;

    wl = generate_helper_layer(56, 256, 256, VPUNN::Operation::ELTWISE, VPUNN::VPUDevice::VPU_2_7,
                               VPUNN::DataType::UINT8);

    EXPECT_NEAR(model.DPUActivityFactor(wl), 0.0760, 0.005f);
}

TEST_F(ActivityFactor, TestPowerActivityFactorMaxPool) {
    // Only one MAXPOOL workload profiled 64x28x28x64 3x3
    VPUNN::DPUWorkload wl;

    wl = generate_helper_layer(28, 64, 64, VPUNN::Operation::MAXPOOL, VPUNN::VPUDevice::VPU_2_7,
                               VPUNN::DataType::UINT8);

    // Override specific fields to correctly configure AVEPOOL workload
    wl.kernels = {3, 3};
    wl.padding = {1, 1, 1, 1};
    wl.strides = {2, 2};
    wl.outputs = {VPUNN::VPUTensor(14, 14, 64, 1, VPUNN::DataType::UINT8)};
    EXPECT_NEAR(model.DPUActivityFactor(wl), 0.4258, 0.05f);
}
}  // namespace VPUNN_unit_tests
