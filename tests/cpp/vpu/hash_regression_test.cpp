// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>

#include <functional>

#include "vpu/dpu_workload.h"
#include "vpu/validation/data_dpu_operation.h"
#include "vpu/validation/serializable_dpu.h"

namespace VPUNN_unit_tests {

using namespace VPUNN;

struct HashBaselineCase {
    const char* name;
    DPUWorkload workload;
    uint32_t expected_workload_hash;
    size_t expected_operation_hash;
    size_t expected_serializable_hash;
    std::function<void(DPUOperation&)> configure_operation_for_baseline;
};

class HashRegressionTest : public ::testing::Test {
protected:
    // IMPORTANT: These values are intentionally hardcoded baselines.
    //
    // What this test does:
    // - Verifies that DPUWorkload::hash(), SerializableDPU::hash() and DPUOperation::hash() remain byte-for-byte
    //   compatible with captured reference baselines.
    //
    // - Captured outputs used here:
    //   conv_npu5_sparse
    //     DPUWorkload::hash = 3047471188 (0xB5A4B854)
    //     DPUOperation::hash = 1443614963 (0x560BD0F3)
    //     SerializableDPU::hash = 2348771807 (0x8C01543F)
    //   dwconv_NPU_RESERVED_halo_sep
    //     DPUWorkload::hash = 3693615666 (0xDC281A32)
    //     DPUOperation::hash = 3883484373 (0xE77944D5)
    //     SerializableDPU::hash = 1224966957 (0x4902382D)
    //   eltwise_npu5_inplace
    //     DPUWorkload::hash = 2160729845 (0x80CA1EF5)
    //     DPUOperation::hash = 1198316386 (0x476CDB62)
    //     SerializableDPU::hash = 3125163774 (0xBA45CF9E)
    //
    //  When changing these constants, keep in mind that:
    // - Changing them invalidates compatibility guarantees and can silently break cache/UID
    //   continuity with historical data.
    // - Only update when there is an explicit, reviewed decision to change hash contract.
    //   If updated, re-capture on the intended target toolchain and document rationale.
    std::vector<HashBaselineCase> make_baseline_cases() {
        std::vector<HashBaselineCase> cases;

        {
            DPUWorkload wl{VPUDevice::NPU_5_0,
                           Operation::CONVOLUTION,
                           {VPUTensor(16, 16, 64, 1, DataType::UINT8, Layout::ZXY)},
                           {VPUTensor(16, 16, 64, 1, DataType::UINT8, Layout::ZXY)},
                           {3, 3},
                           {1, 1},
                           {1, 1, 1, 1},
                           ExecutionMode::CUBOID_16x16};
            wl.activation_function = ActivationFunction::RELU;
            wl.weight_sparsity_enabled = true;
            wl.weight_sparsity = 0.25F;
            wl.act_sparsity = 0.1F;
            wl.weight_type = DataType::INT8;
            wl.mpe_engine = MPEEngine::SCL;
            wl.reduce_minmax_op = false;

            cases.push_back({"conv_npu5_sparse", wl, static_cast<uint32_t>(3047471188U),
                             static_cast<size_t>(1443614963ULL), static_cast<size_t>(2348771807ULL),
                             [](DPUOperation& op) {
                                 op.input_1.batch = 1;
                                 op.input_1.channels = 64;
                                 op.input_1.height = 3;
                                 op.input_1.width = 3;
                                 op.input_1.layout = Layout::ZXY;
                                 op.input_1.sparsity_enabled = true;
                             }});
        }

        {
            DPUWorkload wl{VPUDevice::NPU_RESERVED,
                           Operation::DW_CONVOLUTION,
                           {VPUTensor(28, 28, 64, 1, DataType::INT8, Layout::ZXY)},
                           {VPUTensor(28, 28, 64, 1, DataType::UINT8, Layout::ZXY)},
                           {5, 5},
                           {2, 2},
                           {2, 2, 1, 1},
                           ExecutionMode::CUBOID_8x16};
            wl.halo.input_0_halo.top = 1;
            wl.halo.output_0_halo.left = 2;
            wl.sep_activators.sep_activators = true;
            wl.sep_activators.no_sparse_map = true;
            wl.input_swizzling = {Swizzling::KEY_1, Swizzling::KEY_2};
            wl.output_swizzling = {Swizzling::KEY_3};
            wl.output_write_tiles = 4;
            wl.isi_strategy = ISIStrategy::SPLIT_OVER_K;
            wl.input_autopad = true;
            wl.output_autopad = true;
            wl.mpe_engine = MPEEngine::SCL;
            wl.reduce_minmax_op = true;

            cases.push_back({"dwconv_NPU_RESERVED_halo_sep", wl, static_cast<uint32_t>(3693615666U),
                             static_cast<size_t>(3883484373ULL), static_cast<size_t>(1224966957ULL),
                             [](DPUOperation& op) {
                                 op.input_1.batch = 1;
                                 op.input_1.channels = 64;
                                 op.input_1.height = 5;
                                 op.input_1.width = 5;
                                 op.input_1.layout = Layout::ZXY;
                             }});
        }

        {
            DPUWorkload wl{VPUDevice::NPU_5_0,
                           Operation::ELTWISE,
                           {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                           {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY)},
                           {1, 1},
                           {1, 1},
                           {0, 0, 0, 0},
                           ExecutionMode::CUBOID_16x16};
            wl.set_inplace_input1(true);
            wl.set_inplace_output_memory(true);
            wl.set_superdense(true);
            wl.input_autopad = false;
            wl.output_autopad = true;
            wl.mpe_engine = MPEEngine::SCL;

            cases.push_back({"eltwise_npu5_inplace", wl, static_cast<uint32_t>(2160729845U),
                             static_cast<size_t>(1198316386ULL), static_cast<size_t>(3125163774ULL),
                             [](DPUOperation& op) {
                                 op.input_1.batch = 1;
                                 op.input_1.channels = 64;
                                 op.input_1.height = 1;
                                 op.input_1.width = 1;
                                 op.input_1.layout = Layout::ZXY;
                             }});
        }

        return cases;
    }
};

TEST_F(HashRegressionTest, DPUWorkload_StoredBaselines) {
    const auto test_cases = make_baseline_cases();
    for (const auto& test_case : test_cases) {
        EXPECT_EQ(test_case.workload.hash(), test_case.expected_workload_hash) << "Case: " << test_case.name;
    }
}

TEST_F(HashRegressionTest, SerializableDPU_StoredBaselines) {
    const auto test_cases = make_baseline_cases();
    for (const auto& test_case : test_cases) {
        DPUOperation op{test_case.workload};
        test_case.configure_operation_for_baseline(op);
        const SerializableDPU serializable{op};

        EXPECT_EQ(serializable.hash(), test_case.expected_serializable_hash) << "Case: " << test_case.name;
    }
}

TEST_F(HashRegressionTest, DPUOperation_StoredBaselines) {
    const auto test_cases = make_baseline_cases();
    for (const auto& test_case : test_cases) {
        DPUOperation op{test_case.workload};
        test_case.configure_operation_for_baseline(op);

        EXPECT_EQ(op.hash(), test_case.expected_operation_hash) << "Case: " << test_case.name;
    }
}

}  // namespace VPUNN_unit_tests
