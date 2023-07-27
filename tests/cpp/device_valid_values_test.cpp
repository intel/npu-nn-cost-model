// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/validation/device_valid_values.h"

#include <gtest/gtest.h>

#include "vpu/validation/dpu_operations_validator.h"  //for mock

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DeviceValidValuesTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

    class DeviceValidValuesMock : public VPUNN::IDeviceValidValues {
    public:
        const VPUNN::OperationsBehaviour specific_behaviours{};  ///< known behaviors

        /// @brief non public constructor for initializing the reference
        DeviceValidValuesMock(): VPUNN::IDeviceValidValues(specific_behaviours) {
        }

        const VPUNN::Channels& get_output_channels_range(const VPUNN::DPUOperation&) const override {
            return channels_1;
        };

        const VPUNN::Channels& get_input_channels_range(const VPUNN::DPUOperation&) const override {
            return channels_1;
        };

        VPUNN::Layout adapt_device_comaptible_tensor_layout(VPUNN::Layout layout) const override {
            return layout;
        };

        VPUNN::Swizzling adapt_device_comaptible_swizzling(VPUNN::Swizzling swizz) const override {
            return swizz;
        };

        const VPUNN::Channels channels_1{1};
    };

private:
};

/// Test make list method
TEST_F(DeviceValidValuesTest, makeListTest) {
    DeviceValidValuesMock dut;

    {
        const auto res{dut.makeList(1, 10)};
        VPUNN::Values<int> exp{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        ASSERT_EQ(res.size(), exp.size());

        for (unsigned int i = 0; i < exp.size(); ++i) {
            EXPECT_EQ(res[i], exp[i]) << "i: " << i;
        }
    }
    {
        const auto res{dut.makeList(1, 10, 100)};
        VPUNN::Values<int> exp{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

        ASSERT_EQ(res.size(), exp.size());

        for (unsigned int i = 0; i < exp.size(); ++i) {
            EXPECT_EQ(res[i], exp[i]) << "i: " << i;
        }
    }

    {
        const auto res{dut.makeList(25, 25, 100)};
        VPUNN::Values<int> exp{2500};

        ASSERT_EQ(res.size(), exp.size());

        for (unsigned int i = 0; i < exp.size(); ++i) {
            EXPECT_EQ(res[i], exp[i]) << "i: " << i;
        }
    }
}

/// Test get_padMax method
TEST_F(DeviceValidValuesTest, padMaxTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.get_padMax(7), 3);
    ASSERT_EQ(dut.get_padMax(8), 4);

    ASSERT_EQ(dut.get_padMax(1), 0);
    ASSERT_EQ(dut.get_padMax(2), 1);
    ASSERT_EQ(dut.get_padMax(0), 0);

    ASSERT_EQ(dut.get_padMax(3), 1);
    ASSERT_EQ(dut.get_padMax(4), 2);

    ASSERT_EQ(dut.get_padMax(5), 2);
    ASSERT_EQ(dut.get_padMax(6), 3);
}

/// Test compute_output_dim method
TEST_F(DeviceValidValuesTest, outputDimTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 3, 1), 5);
    ASSERT_EQ(dut.compute_output_dim(7, 1, 1, 3, 1), 7);
    ASSERT_EQ(dut.compute_output_dim(7, 1, 0, 3, 1), 6);
    ASSERT_EQ(dut.compute_output_dim(7, 0, 1, 3, 1), 6);

    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 3, 2), 3);

    ASSERT_EQ(dut.compute_output_dim(6, 0, 0, 3, 2), 2);

    ASSERT_EQ(dut.compute_output_dim(3, 0, 0, 3, 1), 1);
    ASSERT_EQ(dut.compute_output_dim(3, 0, 0, 3, 2), 1);
    ASSERT_EQ(dut.compute_output_dim(3, 1, 1, 3, 1), 3);

    ASSERT_EQ(dut.compute_output_dim(10, 2, 2, 5, 2), 5);
    ASSERT_EQ(dut.compute_output_dim(10, 1, 1, 5, 2), 4);
    ASSERT_EQ(dut.compute_output_dim(10, 1, 2, 5, 2), 5);

    ASSERT_EQ(dut.compute_output_dim(10, 3, 3, 9, 5), 2);
    ASSERT_EQ(dut.compute_output_dim(10, 3, 3, 9, 4), 2);

    ASSERT_EQ(dut.compute_output_dim(10, 2, 2, 9, 4), 2);
    ASSERT_EQ(dut.compute_output_dim(10, 2, 3, 9, 4), 2);

    ASSERT_EQ(dut.compute_output_dim(10, 3, 3, 7, 4), 3);
    ASSERT_EQ(dut.compute_output_dim(10, 2, 3, 7, 4), 3);
    ASSERT_EQ(dut.compute_output_dim(10, 2, 2, 7, 4), 2);

    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 1, 1), 7);
    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 1, 2), 4);
    ASSERT_EQ(dut.compute_output_dim(7, 0, 0, 1, 3), 3);

    ASSERT_EQ(dut.compute_output_dim(7, 1, 0, 1, 3), 3);

    // errors examples
    ASSERT_EQ(dut.compute_output_dim(73, 1, 1, 7, 6), 12);

    EXPECT_EQ(dut.compute_output_dim(13, 1, 1, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(15, 1, 1, 7, 6), 2);

    EXPECT_EQ(dut.compute_output_dim(12, 1, 1, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(12, 0, 1, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(12, 1, 0, 7, 6), 2);
    EXPECT_EQ(dut.compute_output_dim(12, 0, 0, 7, 6), 1);
}

/// Test pad_to_next_multiple method
TEST_F(DeviceValidValuesTest, pad_to_next_multipleTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.align_to(7, 16), 16);
    ASSERT_EQ(dut.align_to(7, 20), 20);
    ASSERT_EQ(dut.align_to(0, 16), 0);

    ASSERT_EQ(dut.align_to(17, 16), 32);
    ASSERT_EQ(dut.align_to(33, 16), 48);
}

TEST_F(DeviceValidValuesTest, check_trailing_padding_dimTest) {
    DeviceValidValuesMock dut;

    ASSERT_EQ(dut.check_trailing_padding(7, 5, 0, 3, 1), 0);
    ASSERT_EQ(dut.check_trailing_padding(7, 7, 1, 3, 1), 1);

    ASSERT_EQ(dut.check_trailing_padding(10, 5, 2, 5, 2), 1);  // can be

    ASSERT_EQ(dut.check_trailing_padding(7, 7, 1, 3, 1), 1);  // huge padding

    ASSERT_EQ(dut.check_trailing_padding(7, 3, 0, 1, 3), 0);  // k=1

    ASSERT_EQ(dut.check_trailing_padding(7, 7, 0, 1, 1), 0);  // k=1

    // errors examples
    EXPECT_EQ(dut.check_trailing_padding(73, 12, 1, 7, 6), 0);

    EXPECT_EQ(dut.check_trailing_padding(13, 2, 1, 7, 6), 0);
    EXPECT_EQ(dut.check_trailing_padding(14, 2, 1, 7, 6), 0);
    EXPECT_EQ(dut.check_trailing_padding(15, 2, 1, 7, 6), 0);

    EXPECT_EQ(dut.check_trailing_padding(12, 2, 1, 7, 6), 0);
    EXPECT_EQ(dut.check_trailing_padding(12, 2, 0, 7, 6), 1);
    EXPECT_EQ(dut.check_trailing_padding(12, 2, 1, 7, 6), 0);
}
using namespace VPUNN;

// checks what is the minim input height that can be used with this kernel & padding, and STrategy if unsplit
TEST_F(DeviceValidValuesTest, get_input_height_range) {
    DeviceValidValuesMock dut;
    const DPUWorkload tst_refH16P1K3x3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K5x5{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 5},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K2x2{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 2},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    auto check_it = [&dut](DPUOperation& wl, std::array<int, 2> pad, int minExp, std::string h = "") {
        wl.kernel.pad_top = pad[0];
        wl.kernel.pad_bottom = pad[1];
        auto r = dut.get_input_height_range(wl);
        EXPECT_EQ(r[0], minExp) << h << "paddings: " << pad[0] << " , " << pad[1] << "\n" << wl;
    };

    {  // 3x3
        std::string info{" x3 "};
        DPUOperation wl{tst_refH16P1K3x3};
        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 2, info);
        check_it(wl, {0, 1}, 2, info);
        check_it(wl, {0, 0}, 3, info);
    }
    {  // x5
        std::string info{" x5 "};
        DPUOperation wl{tst_refH16P1K5x5};
        check_it(wl, {2, 2}, 1, info);
        check_it(wl, {1, 0}, 4, info);
        check_it(wl, {1, 1}, 3, info);
        check_it(wl, {0, 0}, 5, info);

        check_it(wl, {10, 10}, 1, info + "pathologic pad! ");
    }
    {  // x2
        std::string info{" x2 "};
        DPUOperation wl{tst_refH16P1K2x2};

        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 1, info);
        check_it(wl, {0, 1}, 1, info);
        check_it(wl, {0, 0}, 2, info);
    }

    {  // 3x3 and SOH special:
        DPUOperation wl{tst_refH16P1K3x3};
        {
            std::string info{" x3SOH1x "};
            wl.isi_strategy = ISIStrategy::SPLIT_OVER_H;
            EXPECT_EQ(dut.input_heigth_start_factor_SOH, 1);

            check_it(wl, {1, 1}, 1, info);
            check_it(wl, {1, 0}, 2, info);
            check_it(wl, {0, 1}, 2, info);
            check_it(wl, {0, 0}, 3, info);
        }
        {
            // 3x3 and SOH special :
            std::string info{" x3SOH2x "};
            dut.input_heigth_start_factor_SOH = 2;  // now we need t least 2 rows in output to be guaranteed
            EXPECT_EQ(dut.input_heigth_start_factor_SOH, 2);

            check_it(wl, {1, 1}, 2, info);
            check_it(wl, {1, 0}, 3, info);
            check_it(wl, {0, 1}, 3, info);
            check_it(wl, {0, 0}, 4, info);
        }
    }
}

// checks what is the minim input height that can be used with this kernel & padding
TEST_F(DeviceValidValuesTest, get_input_width_range) {
    DeviceValidValuesMock dut;
    const DPUWorkload tst_refH16P1K3x3{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 1},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K5x5{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 1},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH16P1K2x2{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {2, 1},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    auto check_it = [&dut](DPUOperation& wl, std::array<int, 2> pad, int minExp, std::string h = "") {
        wl.kernel.pad_left = pad[0];
        wl.kernel.pad_right = pad[1];
        auto r = dut.get_input_width_range(wl);
        EXPECT_EQ(r[0], minExp) << h << "paddings: " << pad[0] << " , " << pad[1] << "\n" << wl;
    };

    {  // 3x3
        std::string info{" x3 "};
        DPUOperation wl{tst_refH16P1K3x3};
        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 2, info);
        check_it(wl, {0, 1}, 2, info);
        check_it(wl, {0, 0}, 3, info);
    }
    {  // x5
        std::string info{" x5 "};
        DPUOperation wl{tst_refH16P1K5x5};
        check_it(wl, {2, 2}, 1, info);
        check_it(wl, {1, 0}, 4, info);
        check_it(wl, {1, 1}, 3, info);
        check_it(wl, {0, 0}, 5, info);

        check_it(wl, {10, 10}, 1, info + "pathologic pad! ");
    }
    {  // x2
        std::string info{" x2 "};
        DPUOperation wl{tst_refH16P1K2x2};

        check_it(wl, {1, 1}, 1, info);
        check_it(wl, {1, 0}, 1, info);
        check_it(wl, {0, 1}, 1, info);
        check_it(wl, {0, 0}, 2, info);
    }

    {  // 3x3 and SOH special:
        DPUOperation wl{tst_refH16P1K3x3};
        {
            std::string info{" x3SOH1x "};
            wl.isi_strategy = ISIStrategy::SPLIT_OVER_H;
            EXPECT_EQ(dut.input_heigth_start_factor_SOH, 1);

            check_it(wl, {1, 1}, 1, info);
            check_it(wl, {1, 0}, 2, info);
            check_it(wl, {0, 1}, 2, info);
            check_it(wl, {0, 0}, 3, info);
        }
        {
            // 3x3 and SOH special :
            std::string info{" x3SOH2x "};
            dut.input_heigth_start_factor_SOH = 2;  // NO change /influence
            EXPECT_EQ(dut.input_heigth_start_factor_SOH, 2);

            check_it(wl, {1, 1}, 1, info);
            check_it(wl, {1, 0}, 2, info);
            check_it(wl, {0, 1}, 2, info);
            check_it(wl, {0, 0}, 3, info);
        }
    }
}

class DPUOperationTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl_ref1{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},          // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    const VPUNN::DPUWorkload wl_ref2{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::INT8, VPUNN::Layout::YXZ, true)},  // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::INT8, VPUNN::Layout::YZX)},          // output dimensions
            {3, 3},                                                                               // kernels
            {2, 2},                                                                               // strides
            {1, 0, 1, 0},                                                                         // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                                   // execution mode
            VPUNN::ActivationFunction::NONE,                                                      // activation
            0.8F,                                                                                 // act_sparsity
            0.2F,                                                                                 // weight_sparsity
            {VPUNN::Swizzling::KEY_1, VPUNN::Swizzling::KEY_2},                                   // input_swizzling
            {VPUNN::Swizzling::KEY_4},                                                            // output_swizzling
            1,                                                                                    // output_write_tiles
            {0, 0, 0, 0},                                                                         // offsets
            VPUNN::ISIStrategy::SPLIT_OVER_H,                                                     // isi_strategy
            true,  // weight_sparsity_enabled
    };
    const float EPSILON{0.00001f};
    bool is_equal(float a, float b) {
        return (std::fabs(a - b) < EPSILON);  // very simple since vals around zero
    };

    void testEqual(const DPUOperation& dpu, const DPUWorkload& wl) {
        EXPECT_EQ(dpu.device, wl.device);
        EXPECT_EQ(dpu.operation, wl.op);

        // input 0/ act
        EXPECT_EQ(dpu.input_0.height, wl.inputs[0].height());
        EXPECT_EQ(dpu.input_0.width, wl.inputs[0].width());
        EXPECT_EQ(dpu.input_0.channels, wl.inputs[0].channels());
        EXPECT_EQ(dpu.input_0.batch, wl.inputs[0].batches());
        EXPECT_EQ(dpu.input_0.datatype, wl.inputs[0].get_dtype());
        EXPECT_EQ(dpu.input_0.layout, wl.inputs[0].get_layout());

        EXPECT_NEAR(dpu.input_0.sparsity, wl.act_sparsity, EPSILON);

        EXPECT_EQ(dpu.input_0.sparsity_enabled, wl.inputs[0].get_sparsity());
        EXPECT_EQ(dpu.input_0.swizzling, wl.input_swizzling[0]);

        // inout 1 /weights
        EXPECT_EQ(dpu.input_1.height, 0);
        EXPECT_EQ(dpu.input_1.width, 0);
        EXPECT_EQ(dpu.input_1.channels, 0);
        EXPECT_EQ(dpu.input_1.batch, 1);
        EXPECT_EQ(dpu.input_1.datatype, wl.inputs[0].get_dtype());
        EXPECT_EQ(dpu.input_1.layout, Layout::ZXY);

        EXPECT_NEAR(dpu.input_1.sparsity, wl.weight_sparsity, EPSILON);

        EXPECT_EQ(dpu.input_1.sparsity_enabled, wl.weight_sparsity_enabled);
        EXPECT_EQ(dpu.input_1.swizzling, wl.input_swizzling[1]);

        // output
        EXPECT_EQ(dpu.output_0.height, wl.outputs[0].height());
        EXPECT_EQ(dpu.output_0.width, wl.outputs[0].width());
        EXPECT_EQ(dpu.output_0.channels, wl.outputs[0].channels());
        EXPECT_EQ(dpu.output_0.batch, wl.outputs[0].batches());
        EXPECT_EQ(dpu.output_0.datatype, wl.outputs[0].get_dtype());
        EXPECT_EQ(dpu.output_0.layout, wl.outputs[0].get_layout());

        EXPECT_NEAR(dpu.output_0.sparsity, 0.0f, EPSILON);

        EXPECT_EQ(dpu.output_0.sparsity_enabled, wl.outputs[0].get_sparsity());
        EXPECT_EQ(dpu.output_0.swizzling, wl.output_swizzling[0]);

        EXPECT_EQ(dpu.execution_order, wl.execution_order);

        EXPECT_EQ(dpu.kernel.height, wl.kernels[Dim::Grid::H]);
        EXPECT_EQ(dpu.kernel.width, wl.kernels[Dim::Grid::W]);
        EXPECT_EQ(dpu.kernel.pad_bottom, wl.padding[Dim::Padding::BOTTOM]);
        EXPECT_EQ(dpu.kernel.pad_left, wl.padding[Dim::Padding::LEFT]);
        EXPECT_EQ(dpu.kernel.pad_right, wl.padding[Dim::Padding::RIGHT]);
        EXPECT_EQ(dpu.kernel.pad_top, wl.padding[Dim::Padding::TOP]);
        EXPECT_EQ(dpu.kernel.stride_height, wl.strides[Dim::Grid::H]);
        EXPECT_EQ(dpu.kernel.stride_width, wl.strides[Dim::Grid::W]);

        EXPECT_EQ(dpu.output_write_tiles, wl.output_write_tiles);
        EXPECT_EQ(dpu.isi_strategy, wl.isi_strategy);
    }
};

TEST_F(DPUOperationTest, constructor) {
    EXPECT_FALSE(wl_ref1 == wl_ref2);

    {
        DPUOperation dpu(wl_ref2);
        testEqual(dpu, wl_ref2);
    }
    {
        DPUOperation dpu(wl_ref1);
        testEqual(dpu, wl_ref1);
    }
}

TEST_F(DPUOperationTest, toDPUWorkload) {
    {
        DPUOperation dpu(wl_ref2);
        DPUWorkload w{dpu.clone_as_DPUWorkload()};

        EXPECT_EQ(w, wl_ref2);
        EXPECT_FALSE(w == wl_ref1);
    }
    {
        DPUOperation dpu(wl_ref1);
        DPUWorkload w{dpu.clone_as_DPUWorkload()};

        EXPECT_EQ(w, wl_ref1);
        EXPECT_FALSE(w == wl_ref2);
    }
}

}  // namespace VPUNN_unit_tests
