// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/layer_sanitizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "common_helpers.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;
class LayersValidationTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

TEST_F(LayersValidationTest, basicLayerValidatorTest) {
    VPUNN::LayersValidation dut;
    VPUNN::VPUDevice device_req{VPUNN::VPUDevice::VPU_2_7};

    {
        VPUNN::DPULayer wl(VPUNN::DPUWorkload{
                device_req,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
                {1, 1},                                                     // kernels
                {1, 1},                                                     // strides
                {0, 0, 0, 0},                                               // padding
                VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
        });

        {
            VPUNN::SanityReport sane;
            dut.check_completeLayer_consistency(wl, sane, VPUNN::ISIStrategy::CLUSTERING, 1);

            EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                    << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                    << wl;
        }

        {
            VPUNN::SanityReport sane;
            dut.check_splitLayer_consistency(wl, sane);

            EXPECT_EQ(sane.value(), V(VPUNN::Cycles::NO_ERROR))
                    << sane.info << "\n error is : " << VPUNN::Cycles::toErrorText(sane.value()) << "\n"
                    << wl;
        }
    }
}

TEST_F(LayersValidationTest, ExecutionModes_layer_level_test) {
    VPUNN::LayersValidation dut;

    auto mkWl = [](VPUDevice dev, Operation op, unsigned int in_ch = 16, unsigned int out_ch = 16,
                   Layout layout = Layout::ZXY) {
        DPUWorkload _wl{
                dev,
                op,
                {VPUTensor(43, 3, in_ch, 1, DataType::FLOAT16, layout)},   // input dimensions
                {VPUTensor(43, 3, out_ch, 1, DataType::FLOAT16, layout)},  // output dimensions
                {1, 1},                                                    // kernels
                {1, 1},                                                    // strides
                {0, 0, 0, 0},                                              // padding
                ExecutionMode::CUBOID_16x16,  // execution mode --> execution mode doesn't matter here, we will iterate
                                              // trough a list of all execution modes
                ActivationFunction::NONE,     // activation
                0.0F,                         // act_sparsity
                0.0F,                         // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                            // input_swizzling
                {Swizzling::KEY_0},                                              // output_swizzling
                1,                                                               // output_write_tiles
                {0, 0, 0, 0},                                                    // offsets
                ISIStrategy::CLUSTERING,                                         // isi_strategy
                false,                                                           // weight_sparsity_enabled
                {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
                {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
                std::optional<DataType>{},  // DataType::UINT8,                 // input1 data type (opt)
                "",                         // layer_info
                std::optional<bool>{},      // false,                       // weightless_operation (opt)
                std::optional<bool>{},      // false,                      // in_place_output_memory (opt)
                std::optional<bool>{},      // false                       // superdense_memory (opt)
        };
        DPULayer layer(_wl);
        return layer;
    };

    struct TestInput {
        DPULayer layer;
        // there is a list with all execution modes
        std::vector<ExecutionMode> all_exec_modes{ExecutionMode::VECTOR_FP16, ExecutionMode::VECTOR,
                                                  ExecutionMode::MATRIX,      ExecutionMode::CUBOID_4x16,
                                                  ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_16x16};
    };

    struct TestExpectation {
        CyclesInterfaceType err_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto check_err_splitLayer = [&dut](TestsVector& tests) {
        SanityReport sane;
        std::cout << "---------------- check_splitLayer_consistency ---------------- \n";
        for (auto& t : tests) {
            std::cout << "-------- Device:" << VPUDevice_ToText.at(static_cast<int>(t.t_in.layer.device))
                      << " Operation:" << Operation_ToText.at(static_cast<int>(t.t_in.layer.op)) << " ---------\n";

            for (auto exec : t.t_in.all_exec_modes) {
                t.t_in.layer.execution_order = exec;
                dut.check_splitLayer_consistency(t.t_in.layer, sane);

                EXPECT_EQ(sane.value(), (t.t_exp.err_expected))
                        << sane.info << "\n error is : " << Cycles::toErrorText(sane.value())
                        << " Exec order:" << ExecutionMode_ToText.at(static_cast<int>(t.t_in.layer.execution_order))
                        << "\n";
            }
            std::cout << "-------------------------------------------------------\n\n";
        }
        std::cout << "----------------------------------------------------------------------\n\n";
    };

    auto check_err_completeLayer = [&dut](TestsVector& tests) {
        SanityReport sane;
        std::cout << "---------------- check_completeLayer_consistency ---------------- \n";
        for (auto& t : tests) {
            std::cout << "-------- Device:" << VPUDevice_ToText.at(static_cast<int>(t.t_in.layer.device))
                      << " Operation:" << Operation_ToText.at(static_cast<int>(t.t_in.layer.op)) << " --------\n";

            for (auto exec : t.t_in.all_exec_modes) {
                t.t_in.layer.execution_order = exec;
                dut.check_completeLayer_consistency(t.t_in.layer, sane, VPUNN::ISIStrategy::CLUSTERING, 1);

                EXPECT_EQ(sane.value(), (t.t_exp.err_expected))
                        << sane.info << "\n error is : " << Cycles::toErrorText(sane.value())
                        << " Exec order:" << ExecutionMode_ToText.at(static_cast<int>(t.t_in.layer.execution_order))
                        << "\n";
            }
            std::cout << "-------------------------------------------------------\n\n";
        }
        std::cout << "----------------------------------------------------------------------\n\n";
    };

    {
        VPUDevice device = VPUDevice::VPU_2_0;
        TestsVector tests = {
                // clang-format off
        {{mkWl(device, Operation::CONVOLUTION, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::DW_CONVOLUTION, 16, 16, Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::CM_CONVOLUTION, 16, 16,  Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::ELTWISE, 16, 16,  Layout::ZMAJOR)},{Cycles::NO_ERROR}},
        {{mkWl(device, Operation::MAXPOOL, 16, 16,  Layout::ZMAJOR)},{Cycles::NO_ERROR}},

                // clang-format on
        };

        check_err_splitLayer(tests);
        check_err_completeLayer(tests);
    }

    {
        VPUDevice device = VPUDevice::VPU_2_7;

        TestsVector tests = {
                {{mkWl(device, Operation::CONVOLUTION)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::DW_CONVOLUTION)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::CM_CONVOLUTION, 15)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::ELTWISE)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::MAXPOOL)}, {Cycles::NO_ERROR}},
        };

        check_err_splitLayer(tests);
        check_err_completeLayer(tests);
    }

    {
        VPUDevice device = VPUDevice::VPU_4_0;

        TestsVector tests = {
                {{mkWl(device, Operation::CONVOLUTION)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::DW_CONVOLUTION)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::CM_CONVOLUTION, 15)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::ELTWISE)}, {Cycles::NO_ERROR}},
                {{mkWl(device, Operation::MAXPOOL)}, {Cycles::NO_ERROR}},
        };

        check_err_splitLayer(tests);
        check_err_completeLayer(tests);
    }
}

// here we want to see the behavior of check_layer_consistency() function when layer have bigger weight, height or/and
// channels than we normally accept now we should accept W,H bigger than we normally do at high/layer level because we
// will handle possible problems regarding these situations at lower levels (split layers and workloads)
TEST_F(LayersValidationTest, Check_layer_with_big_shape) {
    auto generate_wl = [](unsigned int w, unsigned int h, unsigned int c) {
        DPUWorkload wl{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(w, h, c, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(w, h, c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                 // kernels
                {1, 1},                                                 // strides
                {0, 0, 0, 0},                                           // padding
                ExecutionMode::CUBOID_16x16,                            // execution mode
        };
        DPULayer wl_(wl);
        return wl_;
    };

    LayersValidation dut;

    struct TestInput {
        const DPULayer wl;
        ISIStrategy strategy;
        unsigned int nTiles;
        VPUTilingStrategy t_str;
    };

    struct TestExpectation {
        CyclesInterfaceType err_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        const std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // big weight
    auto lambda = [&dut](TestsVector& tests) {
        SanityReport sane;

        for (auto& t : tests) {
            std::cout << t.test_case << " " << VPUTilingStrategy_ToText.at(static_cast<int>(t.t_in.t_str)) << "\n";

            dut.check_completeLayer_consistency(t.t_in.wl, sane, t.t_in.strategy, t.t_in.nTiles, t.t_in.t_str);

            EXPECT_EQ(sane.value(), V(t.t_exp.err_expected))
                    << sane.info << "\n error is : " << Cycles::toErrorText(sane.value()) << "\n";
        }
    };

    TestsVector tests = {
            // clang-format off
            {{generate_wl(16000, 20, 16), ISIStrategy::CLUSTERING, 1U, VPUTilingStrategy::SOW},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W, one tile"},
            {{generate_wl(16000, 20, 16), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOW},{Cycles::NO_ERROR}, "Big W, 2 tile"},

            {{generate_wl(16000, 20, 16), ISIStrategy::CLUSTERING, 1U, VPUTilingStrategy::SOK},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W, one tile"},
            {{generate_wl(16000, 20, 16), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOK},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W, 2 tiles"},


            {{generate_wl(20, 16000, 16), ISIStrategy::CLUSTERING, 1U, VPUTilingStrategy::SOH_Overlapped},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big H, one tile"},
            {{generate_wl(20, 16000, 16), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOH_Overlapped},{Cycles::NO_ERROR}, "Big H, 2 tiles"},

            {{generate_wl(20, 16000, 16), ISIStrategy::CLUSTERING, 1U, VPUTilingStrategy::SOH_HaloRead},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big H, one tile"},
            {{generate_wl(20, 16000, 16), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOHK},{Cycles::NO_ERROR}, "Big H, 2 tiles"},

            {{generate_wl(20,  16, 16000), ISIStrategy::CLUSTERING, 1U, VPUTilingStrategy::SOK},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big C, one tile"},
            {{generate_wl(20,  16, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOK},{Cycles::NO_ERROR}, "Big C, 2 tiles"},

            {{generate_wl(20,  16, 16000), ISIStrategy::CLUSTERING, 1U, VPUTilingStrategy::SOH_Overlapped},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big C, one tile"},
            {{generate_wl(20,  16, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOW},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big C, 2 tiles"},

            {{generate_wl(16000, 16000, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOHK},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W,H,C, 2 tiles"},
            {{generate_wl(16000, 16000, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOHW},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W,H,C, 2 tiles"},
            {{generate_wl(16000, 16000, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOH_Overlapped},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W,H,C, 2 tiles"},
            {{generate_wl(16000, 16000, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOK_NO_BROADCAST},{Cycles::ERROR_INVALID_LAYER_CONFIGURATION}, "Big W,H,C, 2 tiles"},
                                                                         
            {{generate_wl(16, 16000, 16000), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOHK},{Cycles::NO_ERROR}, "Big H,C, 2 tiles"},
            {{generate_wl(16000, 16000, 32), ISIStrategy::CLUSTERING, 2U, VPUTilingStrategy::SOHW},{Cycles::NO_ERROR}, "Big W, H 2 tiles"},
            // clang-format on
    };

    lambda(tests);
}
}  // namespace VPUNN_unit_tests
