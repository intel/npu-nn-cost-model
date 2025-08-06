// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "inference/preprocessing.h"
#include <gtest/gtest.h>
#include <array>
#include "vpu/compatibility/types01.h"
#include "vpu/compatibility/types11.h"
#include "vpu/compatibility/types12.h"
#include "vpu/compatibility/types13.h"

#include "vpu_cost_model.h"

#include <algorithm>
#include "common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

// class TestPreprocessingLatest : public ::testing::Test {
// public:
// protected:
//     void SetUp() override {
//     }
//     const DPUWorkload wl = {
//             VPUDevice::VPU_2_7,
//             Operation::CONVOLUTION,
//             {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
//             {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
//             {3, 3},                                       // kernels
//             {1, 1},                                       // strides
//             {1, 1},                                       // padding
//             ExecutionMode::CUBOID_16x16,
//             ActivationFunction::NONE,  //
//             0.2F,                      // act sparsity
//             0.8F,                      // weight_sparsity
//             {swz_def, swz_def},        // input_swizzling
//             {swz_def},                 // output_swizzling
//             1,                         // owtiles
//             {0, 0, 0, 0},              // offsets,
//             ISIStrategy::CLUSTERING,   //
//             false,                     // weight_sparsity_enabled
//             {
//                     {1, 2, 3, 4},              // input_0_halo
//                     {5, 6, 7, 8},              // output_0_halo
//                     {9, 10, 11, 12},           // output_0_halo_broadcast_cnt
//                     {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
//             }                                  // halo
//     };
//
// private:
// };
//
//// Demonstrate some basic assertions.
// TEST_F(TestPreprocessingLatest, SingleWLTestPreprocessing) {
//     // Instantiate the preprocessing class
//     auto pp = PreprocessingLatest<float>();
//     // Transform a single workload
//     std::vector<float> result = pp.transform(wl);
// }
//
//// Demonstrate some basic assertions.
// TEST_F(TestPreprocessingLatest, MultipleWLTestPreprocessing) {
//     // Instantiate the preprocessing class
//     auto pp = PreprocessingLatest<float>();
//     // Transform a batch of them
//     const std::vector<DPUWorkload> wl_lst = {wl, wl, wl};
//     std::vector<float> result = pp.transform(wl_lst);
//
//     for (unsigned int batch_idx = 0; batch_idx < 3; batch_idx++) {
//         std::vector<float> batch_result = pp.transform(wl_lst[batch_idx]);
//
//         EXPECT_EQ(batch_result.size() * 3, result.size());
//         for (unsigned int idx = 0; idx < batch_result.size(); idx++) {
//             EXPECT_EQ(batch_result[idx], result[idx + batch_idx * pp.output_size()]);
//         }
//     }
// }
//
//// Demonstrate basic creation and size
// TEST_F(TestPreprocessingLatest, CreationAndSize) {
//     const unsigned int expected_descriptor_size =
//             102 + 2  // NPU device (5.0 & 5.0W)
//             + 2      // new 2 operations
//             + ((2 + 6) /*new 2 data types*/ * 3 /*tensors*/) +
//             (3 * 6 /*3*6 halo info*/ - 9 /* rem Device and isi*/ - 15 /*swizz is 1 now*/);
//
//     EXPECT_EQ(expected_descriptor_size, 124);
//
//     auto pp{PreprocessingLatest<float>()};
//     EXPECT_EQ(pp.output_size(), expected_descriptor_size);
//
//     size_t data_written = 0;
//     std::vector<float> result = pp.generate_descriptor(wl, data_written);
//     EXPECT_EQ(data_written, expected_descriptor_size);
// }
//
//// The latest fast implementation might have also a stable version implementation and their outputs should be the same
// TEST_F(TestPreprocessingLatest, TestLatestIsEqualToSomeVersion) {
//     // Instantiate the preprocessing class
//     auto pp = PreprocessingLatest<float>();
//
//     auto pp_equiv = Preprocessing_Interface12<float>();
//     // auto pp_equiv = PreprocessingLatest<float>();  // not equal to any
//
//     EXPECT_EQ(pp.interface_version(), (int)NNVersions::VERSION_00_LATEST_NONE);
//     EXPECT_EQ(pp.implements_also_interface(), pp_equiv.interface_version());
//
//     {
//         size_t filled = 0;
//         std::vector<float> resultL = pp.generate_descriptor(wl, filled);
//         size_t filled_equiv = 0;
//         std::vector<float> result_equiv = pp_equiv.generate_descriptor(wl, filled_equiv);
//
//         ASSERT_EQ(resultL.size(), result_equiv.size());
//         EXPECT_EQ(filled, filled_equiv) << "Actual filled data must match";
//
//         std::cout << "\nLatest    ";
//         for (const auto x : resultL) {
//             std::cout << " " << x << "";
//         }
//         std::cout << "\nEquivalent";
//         for (const auto x : result_equiv) {
//             std::cout << " " << x;
//         }
//
//         for (size_t i = 0; i < resultL.size(); ++i) {
//             EXPECT_EQ(resultL[i], result_equiv[i]) << "!= at elem : " << i;
//         }
//     }
//     // todo:  test should be done on multiple workloads
//     {
//         const DPUWorkload tst_wl = {
//                 VPUDevice::VPU_2_7,
//                 Operation::CONVOLUTION,
//                 {VPUTensor(56, 57, 16, 1, DataType::UINT8, Layout::XYZ, true)},   // input dimensions
//                 {VPUTensor(56, 57, 16, 1, DataType::UINT8, Layout::XZY, false)},  // output dimensions
//                 {1, 2},                                                           // kernels
//                 {3, 4},                                                           // strides
//                 {5, 6, 7, 8},                                                     // padding
//                 ExecutionMode::CUBOID_16x16,                                      //
//                 ActivationFunction::NONE,                                         //
//                 0.7F,                                                             // act sparsity
//                 0.3F,                                                             // weight_sparsity
//                 {Swizzling::KEY_0, Swizzling::KEY_1},                             // swiz in
//                 {Swizzling::KEY_5},                                               // swiz out
//                 101,                                                              // owtiles
//                 {0, 0, 0, 0},                                                     // offsets,
//                 ISIStrategy::SPLIT_OVER_H,                                        //
//                 false,                                                            // weight_sparsity_enabled
//                 {
//                         {1, 2, 3, 4},              // input_0_halo
//                         {5, 6, 7, 8},              // output_0_halo
//                         {9, 10, 11, 12},           // output_0_halo_broadcast_cnt
//                         {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
//                 }                                  // halo
//         };
//
//         size_t filled = 0;
//         std::vector<float> resultL = pp.generate_descriptor(tst_wl, filled);
//         size_t filled_equiv = 0;
//         std::vector<float> result_equiv = pp_equiv.generate_descriptor(tst_wl, filled_equiv);
//
//         ASSERT_EQ(resultL.size(), result_equiv.size());
//         EXPECT_EQ(filled, filled_equiv) << "Actual filled data must match";
//
//         std::cout << "\nLatest    ";
//         for (const auto x : resultL) {
//             std::cout << " " << x << "";
//         }
//         std::cout << "\nEquivalent";
//         for (const auto x : result_equiv) {
//             std::cout << " " << x;
//         }
//
//         for (size_t i = 0; i < resultL.size(); ++i) {
//             EXPECT_EQ(resultL[i], result_equiv[i]) << "!= at elem : " << i;
//         }
//     }
// }

class TestPreprocessing_Interface01 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const DPUWorkload wl = {VPUDevice::VPU_2_7,
                            Operation::CONVOLUTION,
                            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                            {3, 3},                                       // kernels
                            {1, 1},                                       // strides
                            {1, 1},                                       // padding
                            ExecutionMode::CUBOID_16x16};

private:
};
/// Test cases covering the creation of the Preprocesing mode
TEST_F(TestPreprocessing_Interface01, CreationTest) {
    auto pp = Preprocessing_Interface01<float>();

    ASSERT_EQ(pp.output_size(), 71);

    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    ASSERT_EQ(data_written, 67);
}

TEST_F(TestPreprocessing_Interface01, TransformBad) {
    auto pp = Preprocessing_Interface01<float>();
    const DPUWorkload wl_b = {VPUDevice::VPU_2_7,
                              Operation::CONVOLUTION,
                              {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                              {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                              {3, 3},                                       // kernels
                              {1, 1},                                       // strides
                              {1, 1},                                       // padding
                              ExecutionMode::CUBOID_16x16};

    ASSERT_EQ(pp.output_size(), 71);

    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl_b, data_written);
    EXPECT_EQ(data_written, 67);
    EXPECT_FLOAT_EQ(result[0], 0);  // device
    EXPECT_FLOAT_EQ(result[1], 0);
    EXPECT_FLOAT_EQ(result[2], 1);
    EXPECT_FLOAT_EQ(result[3], 0);

    EXPECT_FLOAT_EQ(result[4 + 0], 1);  // operation
    EXPECT_FLOAT_EQ(result[4 + 1], 0);
    EXPECT_FLOAT_EQ(result[4 + 2], 0);
    EXPECT_FLOAT_EQ(result[4 + 3], 0);
    EXPECT_FLOAT_EQ(result[4 + 4], 0);
    EXPECT_FLOAT_EQ(result[4 + 5], 0);
}

class TestPreprocessing_Interface10 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const DPUWorkload wl = {VPUDevice::VPU_2_7,
                            Operation::CONVOLUTION,
                            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                            {3, 3},                                       // kernels
                            {1, 1},                                       // strides
                            {1, 1},                                       // padding
                            ExecutionMode::CUBOID_16x16};

private:
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface10, SingleWLTestPreprocessing) {
    // Instantiate the preprocessing class
    auto pp = Preprocessing_Interface10<float>();
    // Transform a single workload
    std::vector<float> result = pp.transformSingle(wl);
}

// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface10, CreationAndSize) {
    auto pp = Preprocessing_Interface10<float>();
    EXPECT_EQ(pp.output_size(), 67);

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)NNVersions::VERSION_10_ENUMS_SAME);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, 67);
}

class TestPreprocessing_Interface11 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const DPUWorkload wl = {
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // output dimensions
            {3, 3},                                                          // kernels
            {1, 1},                                                          // strides
            {1, 1},                                                          // padding
            ExecutionMode::CUBOID_16x16,                                     //
            ActivationFunction::NONE,                                        //
            0.7F,                                                            // act sparsity
            0.3F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // owtiles
            {0, 0, 0, 0},                                                    // offsets,
            ISIStrategy::SPLIT_OVER_H,                                       //
    };

    // const int descriptor_expected_size{93};
    const int tensorDescriptorSize =
            4 + (int)intf_11::DataType::__size + (int)intf_11::Layout::__size + 1;  // 4 + 12 +9+1 =26
    const int tens_dType_idx{4};
    const int tens_Layout_idx{tens_dType_idx + (int)intf_11::DataType::__size};
    const int tens_spars_idx{tens_Layout_idx + (int)intf_11::Layout::__size};

    const int dev_idx = 0;
    const int op_idx = dev_idx + (int)intf_11::VPUDevice::__size;  //+4
    const int in_0_idx{op_idx + (int)intf_11::Operation::__size};  //
    const int in_1_idx{in_0_idx + tensorDescriptorSize};
    const int out_0_idx{in_1_idx + tensorDescriptorSize};

    const int ksp_idx = out_0_idx + tensorDescriptorSize;

    const int swizz_idx =
            ksp_idx + 8 /*kernel:2 +strides:2+padding:4*/ + (int)intf_11::ExecutionMode::__size + 2 /*sparsities:2*/;
    const int swizz_size{(int)intf_11::Swizzling::__size};

    const int owt_idx = swizz_idx + (swizz_size * 3);

private:
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface11, SingleWLTestPreprocessing) {
    // Instantiate the preprocessing class
    auto pp = Preprocessing_Interface11<float>();
    // Transform a single workload
    std::vector<float> result = pp.transformSingle(wl);
    EXPECT_EQ(result.size(), 93);
}
// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface11, CreationAndSize) {
    auto pp = Preprocessing_Interface11<float>();
    EXPECT_EQ(pp.output_size(), 93);  // less , should be 91!!!

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)NNVersions::VERSION_11_VPU27_BETA);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, 93);  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
}
TEST_F(TestPreprocessing_Interface11, DescriptorContentTest) {
    const DPUWorkload wl2 = {
            VPUDevice::VPU_2_7,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {1, 1},                                                             // kernels
            {1, 1},                                                             // strides
            {0, 0, 0, 0},                                                       // padding
            ExecutionMode::CUBOID_4x16,                                         //
            ActivationFunction::NONE,                                           //
            0.F,                                                                // act sparsity
            0.F,                                                                // weight_sparsity
            {swz_def, swz_def},                                                 // input_swizzling
            {swz_def},                                                          // output_swizzling
            1,                                                                  // owtiles
            {0, 0, 0, 0},                                                       // offsets,
            ISIStrategy::CLUSTERING,                                            //
    };

    auto pp = Preprocessing_Interface11<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << std::lround(x) << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[dev_idx + (int)wl2.device]), 1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    // const int in_0_idx = 10;
    // const int in_1_idx = in_0_idx + 16;
    // const int out_0_idx = in_1_idx + 16;
    EXPECT_EQ(std::lround(result[in_0_idx]), wl2.inputs[0].width());
    EXPECT_EQ(std::lround(result[in_0_idx + 1]), wl2.inputs[0].height());
    EXPECT_EQ(std::lround(result[in_0_idx + 2]), wl2.inputs[0].z());
    EXPECT_EQ(std::lround(result[in_0_idx + 3]), wl2.inputs[0].b());

    EXPECT_EQ(std::lround(result[out_0_idx + 0]), wl2.outputs[0].width());
    EXPECT_EQ(std::lround(result[out_0_idx + 1]), wl2.outputs[0].height());
    EXPECT_EQ(std::lround(result[out_0_idx + 2]), wl2.outputs[0].z());
    EXPECT_EQ(std::lround(result[out_0_idx + 3]), wl2.outputs[0].b());

    EXPECT_EQ(std::lround(result[in_1_idx + 0]), std::lround(result[in_0_idx + 1]));
    EXPECT_EQ(std::lround(result[in_1_idx + 1]), std::lround(result[in_0_idx + 2]));
    EXPECT_EQ(std::lround(result[in_1_idx + 2]), std::lround(result[in_0_idx + 0]));
    EXPECT_EQ(std::lround(result[in_1_idx + 3]), std::lround(result[in_0_idx + 3]));
}

TEST_F(TestPreprocessing_Interface11, Test_spatial_memory_tensor_input_2_7) {
    DPUWorkload wl_SOH{
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(65, 12, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(65, 4, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_H,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo info TBLRFB
    };

    const int width_idx = in_0_idx;
    const int height_idx = in_0_idx + 1;
    const int channels_idx = in_0_idx + 2;
    const int batch_idx = in_0_idx + 3;

    /// @param wl_ref
    /// @param in_halo: input halo
    /// @param in_w: expected value for input0 tensor width
    /// @param in_h: expected value for input0 tensor height
    /// @param in_c: expected value for input0 tensor channels
    /// @param in_b: expected value for input0 tensor batch
    auto verify_input_WH = [&width_idx, &height_idx, &channels_idx, &batch_idx](
                                   DPUWorkload& wl_ref, const HaloWorkload::HaloInfoHWC in_halo,
                                   const unsigned int in_w, const unsigned int in_h, const unsigned int in_c,
                                   const unsigned int in_b) {
        wl_ref.halo.input_0_halo = in_halo;

        auto pp = Preprocessing_Interface11<float>();
        size_t data_written = 0;
        std::vector<float> result = pp.generate_descriptor(wl_ref, data_written);
        EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

        EXPECT_EQ(result.size(), data_written);
        std::cout << "\n descriptor\n";

        for (auto x : result) {
            std::cout << " " << std::lround(x) << " ";
        }
        std::cout << "\n" << wl_ref << "\n";

        EXPECT_EQ(std::lround(result[width_idx]), in_w);
        EXPECT_EQ(std::lround(result[height_idx]), in_h);
        EXPECT_EQ(std::lround(result[channels_idx]), in_c);
        EXPECT_EQ(std::lround(result[batch_idx]), in_b);
    };
    // clang-format off
    // input0 tensor dimensions: W=65 H=12 C=64 B=1
    
    // ISIStrategy SOH
    //************* base wl  |      in halo     |     W      |     H     | C | B | 
    verify_input_WH(wl_SOH, {0, 4, 0, 6, 0, 0}, 59 /*65-6*/, 8 /*12-4*/, 64, 1);
    verify_input_WH(wl_SOH, {4, 4, 4, 4, 3, 2}, 57 /*68-4-4*/, 4 /*12-4-4*/, 64, 1);
    verify_input_WH(wl_SOH, {-2, 4, 3, -6, 0, 1}, 62 /*65-3-0*/, 8 /*12-4-0*/, 64, 1); //negative halo is considered equal to 0
    verify_input_WH(wl_SOH, {-1, -1, -1, -1, -7, 0}, 65 /*65-0-0*/, 12 /*12-0-0*/, 64, 1);//negative halo is considered equal to 0
    verify_input_WH(wl_SOH, {3, 4, 0, 0, 0, 0}, 65 /*65*/, 5 /*12-4-3*/, 64, 1);
    verify_input_WH(wl_SOH, {0, 0, 3, 4, 0, 0}, 58 /*65-3-4*/, 12 /*12*/, 64, 1);

    //ISIStrategy CLUSTERING
    DPUWorkload wl_CLU{std::move(wl_SOH)};
    wl_CLU.isi_strategy=ISIStrategy::CLUSTERING;

     //************* base wl |    in halo    |  W  |  H | C | B | 
    verify_input_WH(wl_CLU, {0, 4, 0, 6, 0, 0}, 65 , 12 , 64, 1);
    verify_input_WH(wl_CLU, {3, -4, 0, 0, 0, 0}, 65 , 12 , 64, 1);
    verify_input_WH(wl_CLU, {0, 0, 5, -6, 0, 0}, 65 , 12 , 64, 1);
    verify_input_WH(wl_CLU, {1, 2, 3, 6, 1, 2}, 65 , 12 , 64, 1);
    // clang-format on
}

TEST_F(TestPreprocessing_Interface11, Test_spatial_memory_tensor_input_descriptors_comparing) {
    DPUWorkload wl_SOH{
            VPUDevice::VPU_2_7,
            Operation::DW_CONVOLUTION,
            {VPUTensor(65, 12, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(65, 4, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::SPLIT_OVER_H,                      // isi_strategy
            false,                                          // weight_sparsity_enabled
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo info TBLRFB
    };

    const int width_idx = in_0_idx;
    const int height_idx = in_0_idx + 1;

    /// here we test if 2 descriptors are equal, except for the elements input0 height and input0 width, witch change
    /// due to the presence of halo ans SPLIT_OVER_H
    /// @param wl_SOH DPUWorkload with ISIStrategy SPLIT_OVER_H
    /// @param wl_CLU DPUWorkload with ISIStrategy CLUSTERING
    /// @param in_halo: input halo
    auto verify_input_WH = [&width_idx, &height_idx](DPUWorkload& wl_SOH, DPUWorkload& wl_CLU,
                                                     const HaloWorkload::HaloInfoHWC in_halo) {
        wl_SOH.halo.input_0_halo = in_halo;
        wl_CLU.halo.input_0_halo = in_halo;

        auto pp = Preprocessing_Interface11<float>();
        size_t data_written_wl_SOH = 0;
        std::vector<float> result_wl_SOH = pp.generate_descriptor(wl_SOH, data_written_wl_SOH);
        EXPECT_EQ(data_written_wl_SOH, pp.output_size());  // this is the actual filled dimension

        EXPECT_EQ(result_wl_SOH.size(), data_written_wl_SOH);

        size_t data_written_wl_CLU = 0;
        std::vector<float> result_wl_CLU = pp.generate_descriptor(wl_SOH, data_written_wl_CLU);
        EXPECT_EQ(data_written_wl_CLU, pp.output_size());  // this is the actual filled dimension

        EXPECT_EQ(result_wl_CLU.size(), data_written_wl_CLU);

        // both descriptors should be equal, except for the elements input0 width and input0 height
        // if input halo is zero or negative values are equal else W and H for wl_SOH should be lower
        EXPECT_LE(std::lround(result_wl_SOH[width_idx]), std::lround(result_wl_CLU[width_idx]));
        EXPECT_LE(std::lround(result_wl_SOH[height_idx]), std::lround(result_wl_CLU[height_idx]));

        // here we check that the rest of elements in both descriptors are equals
        for (size_t i = 0; i < result_wl_CLU.size(); i++) {
            if (i != static_cast<long unsigned int>(width_idx) || i != static_cast<long unsigned int>(height_idx)) {
                EXPECT_EQ(std::lround(result_wl_SOH[i]), std::lround(result_wl_CLU[i]));
            }
        }
    };
    DPUWorkload wl_CLU{wl_SOH};
    wl_CLU.isi_strategy = ISIStrategy::CLUSTERING;
    verify_input_WH(wl_SOH, wl_CLU, {1, 2, 3, 4, 5, 6});
    verify_input_WH(wl_SOH, wl_CLU, {-2, -3, 0, -4, 0, 0});
}

// for more info you can see the function @avoid_untrained_space()
TEST_F(TestPreprocessing_Interface11, Test_limit_owt_to_2_mechanism_2_7) {
    DPUWorkload wl_ref{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(65, 12, 64, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(65, 4, 64, 1, DataType::FLOAT16)},   // output dimensions
            {9, 9},                                         // kernels
            {1, 1},                                         // strides
            {0, 0, 4, 4},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            5,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::CLUSTERING,                        // isi_strategy
            false,                                          // weight_sparsity_enabled

    };

    auto generate_des = [](DPUWorkload& wl_ref) -> std::vector<float> {
        auto pp = Preprocessing_Interface11<float>();
        size_t data_written = 0;
        std::vector<float> result = pp.generate_descriptor(wl_ref, data_written);
        return result;
    };

    // case when owt is changed
    std::vector<float> result_owt_change = generate_des(wl_ref);
    std::cout << "\n descriptor when owt change\n";
    for (auto x : result_owt_change) {
        std::cout << " " << std::lround(x) << " ";
    }
    EXPECT_EQ(std::lround(result_owt_change[owt_idx]), 2);

    // case when owt is not changed
    wl_ref.output_write_tiles = 1;
    std::vector<float> result_owt_not_change = generate_des(wl_ref);
    std::cout << "\n descriptor when owt does not change\n";
    for (auto x : result_owt_not_change) {
        std::cout << " " << std::lround(x) << " ";
    }
    EXPECT_EQ(std::lround(result_owt_not_change[owt_idx]), 1);
}
TEST_F(TestPreprocessing_Interface11, SwizzlingContentTest) {
    const DPUWorkload wl2 = {
            VPUDevice::VPU_2_7,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {1, 2},                                                             // kernels
            {3, 4},                                                             // strides
            {5, 6, 7, 8},                                                       // padding
            ExecutionMode::CUBOID_4x16,                                         //
            ActivationFunction::NONE,                                           //
            0.11F,                                                              // act sparsity
            0.22F,                                                              // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                               // swiz in
            {Swizzling::KEY_0},                                                 // swiz out
    };
    const DPUWorkload wl3 = {
            VPUDevice::VPU_2_7,
            Operation::MAXPOOL,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {1, 2},                                                             // kernels
            {3, 4},                                                             // strides
            {5, 6, 7, 8},                                                       // padding
            ExecutionMode::CUBOID_4x16,                                         //
            ActivationFunction::NONE,                                           //
            0.11F,                                                              // act sparsity
            0.22F,                                                              // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                               // swiz in
            {Swizzling::KEY_0},                                                 // swiz out
    };

    auto pp = Preprocessing_Interface11<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(swizz_size, 6);

    struct TestExpected {
        Swizzling swizIn0;
        Swizzling swizIn1;
        Swizzling swizOut0;
    };

    auto executeTest = [&pp, this](const DPUWorkload& tin, const TestExpected& texp) {
        size_t data_written = 0;
        std::vector<float> result = pp.generate_descriptor(tin, data_written);

        // std::cout << "\n descriptor\n";

        // for (auto x : result) {
        //    std::cout << " " << std::lround(x) << " ";
        //}

        const int swzi0_idx{swizz_idx + 0 * swizz_size};
        const int swzi1_idx{swizz_idx + 1 * swizz_size};
        const int swzo0_idx{swizz_idx + 2 * swizz_size};

        const int swzPosIn0{(int)texp.swizIn0};
        const int swzPosIn1{(int)texp.swizIn1};
        const int swzPosOut0{(int)texp.swizOut0};

        EXPECT_EQ(std::lround(result[swzi0_idx + swzPosIn0]), 1) << swzi0_idx << std::endl << tin;
        EXPECT_EQ(std::lround(result[swzi1_idx + swzPosIn1]), 1) << swzi1_idx << std::endl << tin;
        EXPECT_EQ(std::lround(result[swzo0_idx + swzPosOut0]), 1) << swzo0_idx << std::endl << tin;

        // EXPECT_EQ(std::lround(result[swizz_idx + 1 * swizz_size]), texp.swizIn1) << swizz_idx << std::endl <<
        // tin;
        //  EXPECT_EQ(std::lround(result[swizz_idx + 1 * swizz_size + 1]), 0);
        // EXPECT_EQ(std::lround(result[swizz_idx + 2 * swizz_size]), texp.swizOut0) << swizz_idx << std::endl <<
        // tin;
        //  EXPECT_EQ(std::lround(result[swizz_idx + 2 * swizz_size + 1]), 0);
    };

    auto gen = [](Swizzling in0, Swizzling in1, Swizzling out0, const DPUWorkload& prototype) {
        DPUWorkload wl{prototype};
        wl.input_swizzling[0] = in0;
        wl.input_swizzling[1] = in1;
        wl.output_swizzling[0] = out0;
        return wl;
    };

    const DPUWorkload elm_proto{std::move(wl2)};
    const DPUWorkload conv_proto{wl};
    const DPUWorkload maxp_proto{std::move(wl3)};

    struct Test {
        const DPUWorkload tin;
        const TestExpected texp;
    };
    const TestExpected allZero{Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0};
    const TestExpected allFive{Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5};
    const TestExpected five_0_5{Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5};

    std::vector<Test> tests{
            {gen(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0, elm_proto), allZero},
            {gen(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_1, elm_proto), allFive},
            {gen(Swizzling::KEY_0, Swizzling::KEY_1, Swizzling::KEY_0, elm_proto), allFive},
            {gen(Swizzling::KEY_1, Swizzling::KEY_0, Swizzling::KEY_0, elm_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0, elm_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_1, elm_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5, elm_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5, elm_proto), allFive},

            {gen(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0, conv_proto), allFive},
            {gen(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_1, conv_proto), allFive},
            {gen(Swizzling::KEY_0, Swizzling::KEY_1, Swizzling::KEY_0, conv_proto), allFive},
            {gen(Swizzling::KEY_1, Swizzling::KEY_0, Swizzling::KEY_0, conv_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0, conv_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_1, conv_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5, conv_proto), allFive},
            {gen(Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5, conv_proto), allFive},

            {gen(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_1, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_0, Swizzling::KEY_1, Swizzling::KEY_0, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_1, Swizzling::KEY_0, Swizzling::KEY_0, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_1, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5, maxp_proto), five_0_5},
            {gen(Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5, maxp_proto), five_0_5},
    };

    for (auto& t : tests) {
        executeTest(t.tin, t.texp);
    }
}

// check the INVALID wts layout
TEST_F(TestPreprocessing_Interface11, MAXPOOLContentTest) {
    const DPUWorkload wl2 = {
            VPUDevice::VPU_2_7,
            Operation::MAXPOOL,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1},                                                    // padding
            ExecutionMode::CUBOID_16x16,                               //
            ActivationFunction::NONE,                                  //
            0.0F,                                                      // act sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // owtiles
            {0, 0, 0, 0},                                              // offsets,
            ISIStrategy::SPLIT_OVER_H,                                 // isi
            true,                                                      // wt sparsity ON/OFF
    };

    auto pp = Preprocessing_Interface11<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << std::lround(x) << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[dev_idx + (int)wl2.device]),
              1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    {  // input act
        const auto idx{in_0_idx};
        EXPECT_EQ(std::lround(result[idx + 0]), wl2.inputs[0].width());
        EXPECT_EQ(std::lround(result[idx + 1]), wl2.inputs[0].height());
        EXPECT_EQ(std::lround(result[idx + 2]), wl2.inputs[0].z());
        EXPECT_EQ(std::lround(result[idx + 3]), wl2.inputs[0].b());
        // datatyoe
        EXPECT_EQ(std::lround(result[idx + tens_dType_idx + 0]), 1.0f);
        EXPECT_EQ(std::lround(result[idx + tens_dType_idx + 1]), 0.0f);
        // layout XYZ, XZY, YXZ, YZX, ZXY, ZYX, INVALID
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 0]), 0.0f);  // XYZ,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 1]), 0.0f);  // XZY,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 2]), 0.0f);  // YXZ,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 3]), 0.0f);  // YZX,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 4]), 1.0f);  // ZXY,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 5]), 0.0f);  // ZYX,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 6]), 0.0f);  // INVALID
        /// sparsity
        EXPECT_EQ(std::lround(result[in_0_idx + tens_spars_idx + 0]), 0.0f);  // sparsity ON/OFF
    }

    {  // output
        const auto idx{out_0_idx};
        EXPECT_EQ(std::lround(result[idx + 0]), wl2.outputs[0].width());
        EXPECT_EQ(std::lround(result[idx + 1]), wl2.outputs[0].height());
        EXPECT_EQ(std::lround(result[idx + 2]), wl2.outputs[0].z());
        EXPECT_EQ(std::lround(result[idx + 3]), wl2.outputs[0].b());
        // datatype
        EXPECT_EQ(std::lround(result[idx + tens_dType_idx + 0]), 1.0f);
        EXPECT_EQ(std::lround(result[idx + tens_dType_idx + 1]), 0.0f);
        // layout XYZ, XZY, YXZ, YZX, ZXY, ZYX, INVALID
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 0]), 1.0f);  // XYZ,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 1]), 0.0f);  // XZY,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 2]), 0.0f);  // YXZ,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 3]), 0.0f);  // YZX,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 4]), 0.0f);  // ZXY,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 5]), 0.0f);  // ZYX,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 6]), 0.0f);  // INVALID
        /// sparsity
        EXPECT_EQ(std::lround(result[idx + tens_spars_idx + 0]), 0.0f);  // sparsity ON/OFF
    }
    {  // input 1 Wts
        const auto idx{in_1_idx};
        EXPECT_EQ(std::lround(result[idx + 0]), std::lround(0));
        EXPECT_EQ(std::lround(result[idx + 1]), std::lround(0));
        EXPECT_EQ(std::lround(result[idx + 2]), std::lround(0));
        EXPECT_EQ(std::lround(result[idx + 3]), std::lround(0));
        // datatyoe
        EXPECT_EQ(std::lround(result[idx + tens_dType_idx + 0]), 1.0f);
        EXPECT_EQ(std::lround(result[idx + tens_dType_idx + 1]), 0.0f);
        // layout XYZ, XZY, YXZ, YZX, ZXY, ZYX, INVALID
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 0]), 0.0f);  // XYZ,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 1]), 0.0f);  // XZY,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 2]), 0.0f);  // YXZ,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 3]), 0.0f);  // YZX,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 4]), 0.0f);  // ZXY,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 5]), 0.0f);  // ZYX,
        EXPECT_EQ(std::lround(result[idx + tens_Layout_idx + 6]), 1.0f);  // INVALID
        /// sparsity
        EXPECT_EQ(std::lround(result[idx + tens_spars_idx + 0]), 1.0f);  // sparsity ON/OFF
    }
}

TEST_F(TestPreprocessing_Interface11, DescriptorContentTest_FLOAT_INT) {
    class Builder {
    public:
        static DPUWorkload makeSameWl(DataType Tin, DataType Tout) {
            return DPUWorkload{
                    VPUDevice::VPU_2_7,
                    Operation::ELTWISE,
                    {VPUTensor(15, 1972, 16, 1, Tin)},   // input dimensions
                    {VPUTensor(15, 1972, 16, 1, Tout)},  // output dimensions
                    {1, 1},                              // kernels
                    {1, 1},                              // strides
                    {0, 0, 0, 0},                        // padding
                    ExecutionMode::CUBOID_4x16,          //
                    ActivationFunction::NONE,            //
                    0.F,                                 // act sparsity
                    0.F,                                 // weight_sparsity
                    {swz_def, swz_def},                  // input_swizzling
                    {swz_def},                           // output_swizzling
                    1,                                   // owtiles
                    {0, 0, 0, 0},                        // offsets,
                    ISIStrategy::CLUSTERING,             //
            };
        }
    };

    const DPUWorkload wl_int_int{Builder::makeSameWl(DataType::UINT8, DataType::UINT8)};
    const DPUWorkload wl_float_float{Builder::makeSameWl(DataType::FLOAT16, DataType::FLOAT16)};
    const DPUWorkload wl_float_int{Builder::makeSameWl(DataType::FLOAT16, DataType::UINT8)};
    const DPUWorkload wl_int_float{Builder::makeSameWl(DataType::UINT8, DataType::FLOAT16)};

    auto show_desc = [](const std::vector<float>& result, std::string prefix) {
        std::cout << prefix;
        for (auto x : result) {
            std::cout << " " << std::lround(x) << " ";
        }
    };

    auto pp = Preprocessing_Interface11<float>();
    size_t data_written = 0;
    std::vector<float> result_i_i = pp.generate_descriptor(wl_int_int, data_written);
    std::vector<float> result_f_i = pp.generate_descriptor(wl_float_int, data_written);
    std::vector<float> result_i_f = pp.generate_descriptor(wl_int_float, data_written);
    std::vector<float> result_f_f = pp.generate_descriptor(wl_float_float, data_written);

    std::cout << "\n descriptors\n";

    show_desc(result_i_i, "\nii:");
    show_desc(result_f_i, "\nfi:");
    show_desc(result_i_f, "\nif:");
    show_desc(result_f_f, "\nff:");
    std::cout << std::endl;

    // EXPECT_EQ(result_i_i, result_i_f);

    using datatypeContent = std::array<int, (int)intf_11::DataType::__size>;
    const datatypeContent uint8_content{1, 0, 0, 0};
    const datatypeContent f16_content{0, 0, 1, 0};
    const int cntToCheck{(int)std::tuple_size<decltype(uint8_content)>{}};

    const int dtype_idx = 4;
    // const int in_0_idx = 10;
    // const int in_1_idx = in_0_idx + 16;
    // const int out_0_idx = in_1_idx + 16;

    {
        auto& r = result_i_i;
        auto& exp_in_0{uint8_content};
        auto& exp_in_1{uint8_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + i]), exp_in_1[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i_f;
        auto& exp_in_0{uint8_content};
        auto& exp_in_1{uint8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + i]), exp_in_1[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_i;
        auto& exp_in_0{f16_content};
        auto& exp_in_1{f16_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + i]), exp_in_1[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_f;
        auto& exp_in_0{f16_content};
        auto& exp_in_1{f16_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + i]), exp_in_1[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
}

/// interface 12

class TestPreprocessing_Interface12 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const DPUWorkload wl = {
            VPUDevice::NPU_RESERVED,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // output dimensions
            {3, 3},                                                          // kernels
            {1, 1},                                                          // strides
            {1, 1},                                                          // padding
            ExecutionMode::CUBOID_16x16,                                     //
            ActivationFunction::NONE,                                        //
            0.7F,                                                            // act sparsity
            0.3F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // owtiles
            {0, 0, 0, 0},                                                    // offsets,
            ISIStrategy::SPLIT_OVER_H,                                       //
            false,                                                           // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 41, 42},      // input_0_halo
                    {5, 6, 7, 8, 81, 82},      // output_0_halo
                    {9, 10, 11, 12, 91, 92},   // output_0_halo_broadcast_cnt
                    {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
            }                                  // halo
    };

    const int descriptor_expected_size{44};
    const int tensorDescriptorSize = 4 + (int)intf_12::DataType::__size;

    const int op_idx = 0;
    const int in_0_idx{op_idx + (int)intf_12::Operation::__size};
    const int in_1_dtype_idx{in_0_idx + tensorDescriptorSize};
    const int out_0_idx{in_1_dtype_idx + (int)intf_12::DataType::__size};
    const int ksp_idx = out_0_idx + tensorDescriptorSize;
    const int execution_mode_idx = ksp_idx + 8; /*kernel:2 +strides:2+padding:4*/
    const int sparsities_idx = execution_mode_idx + (int)intf_12::ExecutionMode::__size;
    const int owt_idx = sparsities_idx + 2;  /*sparsities:2*/
    const int odu_permute_idx = owt_idx + 1; /*owt:1*/
    const int first_after_end_idx = odu_permute_idx + (int)intf_12::Layout::__size;
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface12, SingleWLTestPreprocessing) {
    EXPECT_EQ(first_after_end_idx, descriptor_expected_size);
    EXPECT_EQ(descriptor_expected_size, 44);
    // Instantiate the preprocessing class
    auto pp = Preprocessing_Interface12<float>();
    // Transform a single workload
    std::vector<float> result = pp.transformSingle(wl);
    EXPECT_EQ(result.size(), descriptor_expected_size);
}
// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface12, CreationAndSize) {
    auto pp = Preprocessing_Interface12<float>();
    EXPECT_EQ(pp.output_size(), descriptor_expected_size);  //

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)NNVersions::VERSION_12_NPU_RESERVED);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, descriptor_expected_size);  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
}

TEST_F(TestPreprocessing_Interface12, DescriptorContentTest) {
    const DPUWorkload wl2 = {
            VPUDevice::NPU_RESERVED,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {1, 2},                                                             // kernels
            {3, 4},                                                             // strides
            {5, 6, 7, 8},                                                       // padding
            ExecutionMode::CUBOID_4x16,                                         //
            ActivationFunction::NONE,                                           //
            0.11F,                                                              // act sparsity
            0.22F,                                                              // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                               // swiz in
            {Swizzling::KEY_0},                                                 // swiz out
            100,                                                                // owtiles
            {0, 0, 0, 0},                                                       // offsets,
            ISIStrategy::SPLIT_OVER_K,                                          //
            false,                                                              // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 5, 6},        // input_0_halo
                    {7, 8, 9, 10, 11, 12},     // output_0_halo
                    {13, 14, 15, 16, 17, 18},  // output_0_halo_broadcast_cnt
                    {19, 20, 21, 22, 23, 24},  // output_0_inbound_halo
            }                                  // halo
    };

    auto pp = Preprocessing_Interface12<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << x /*std::lround(x) */ << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[in_0_idx]), wl2.inputs[0].width());
    EXPECT_EQ(std::lround(result[in_0_idx + 1]), wl2.inputs[0].height());
    EXPECT_EQ(std::lround(result[in_0_idx + 2]), wl2.inputs[0].z());
    EXPECT_EQ(std::lround(result[in_0_idx + 3]), wl2.inputs[0].b());

    EXPECT_EQ(std::lround(result[out_0_idx + 0]), wl2.outputs[0].width());
    EXPECT_EQ(std::lround(result[out_0_idx + 1]), wl2.outputs[0].height());
    EXPECT_EQ(std::lround(result[out_0_idx + 2]), wl2.outputs[0].z());
    EXPECT_EQ(std::lround(result[out_0_idx + 3]), wl2.outputs[0].b());

    EXPECT_EQ(std::lround(result[ksp_idx + 0]), 1);
    EXPECT_EQ(std::lround(result[ksp_idx + 1]), 2);
    EXPECT_EQ(std::lround(result[ksp_idx + 2]), 3);
    EXPECT_EQ(std::lround(result[ksp_idx + 3]), 4);
    EXPECT_EQ(std::lround(result[ksp_idx + 4]), 5);
    EXPECT_EQ(std::lround(result[ksp_idx + 5]), 6);
    EXPECT_EQ(std::lround(result[ksp_idx + 6]), 7);
    EXPECT_EQ(std::lround(result[ksp_idx + 7]), 8);

    EXPECT_EQ(result[sparsities_idx + 0], wl2.act_sparsity);
    EXPECT_EQ(result[sparsities_idx + 1], wl2.weight_sparsity);

    EXPECT_EQ(std::lround(result[owt_idx + 0]), std::min(wl2.output_write_tiles, 2u));

    EXPECT_EQ(std::lround(result[odu_permute_idx + 0]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 4]), 1);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 5]), 0);
}

TEST_F(TestPreprocessing_Interface12, DescriptorContentTest_FLOAT_INT) {
    class Builder {
    public:
        static DPUWorkload makeSameWl(DataType Tin, DataType Tout) {
            return DPUWorkload{
                    VPUDevice::NPU_RESERVED,
                    Operation::ELTWISE,
                    {VPUTensor(15, 1972, 16, 1, Tin)},   // input dimensions
                    {VPUTensor(15, 1972, 16, 1, Tout)},  // output dimensions
                    {1, 1},                              // kernels
                    {1, 1},                              // strides
                    {0, 0, 0, 0},                        // padding
                    ExecutionMode::CUBOID_4x16,          //
                    ActivationFunction::NONE,            //
                    0.F,                                 // act sparsity
                    0.F,                                 // weight_sparsity
                    {swz_def, swz_def},                  // input_swizzling
                    {swz_def},                           // output_swizzling
                    1,                                   // owtiles
                    {0, 0, 0, 0},                        // offsets,
                    ISIStrategy::CLUSTERING,             //
                    false,                               // weight_sparsity_enabled
                    {
                            {1, 2, 3, 4, 41, 42},      // input_0_halo
                            {5, 6, 7, 8, 81, 82},      // output_0_halo
                            {9, 10, 11, 12, 91, 92},   // output_0_halo_broadcast_cnt
                            {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
                    }                                  // halo
            };
        }
    };

    const DPUWorkload wl_int_int{Builder::makeSameWl(DataType::UINT8, DataType::UINT8)};
    const DPUWorkload wl_float_float{Builder::makeSameWl(DataType::FLOAT16, DataType::FLOAT16)};
    const DPUWorkload wl_float_int{Builder::makeSameWl(DataType::FLOAT16, DataType::UINT8)};
    const DPUWorkload wl_int_float{Builder::makeSameWl(DataType::UINT8, DataType::FLOAT16)};
    const DPUWorkload wl_int4_float{Builder::makeSameWl(DataType::UINT4, DataType::FLOAT16)};
    const DPUWorkload wl_bf8_float{Builder::makeSameWl(DataType::BF8, DataType::FLOAT16)};

    auto show_desc = [](const std::vector<float>& result, std::string prefix) {
        std::cout << prefix;
        for (auto x : result) {
            std::cout << " " << std::lround(x) << " ";
        }
    };

    auto pp = Preprocessing_Interface12<float>();
    size_t data_written = 0;
    std::vector<float> result_i_i = pp.generate_descriptor(wl_int_int, data_written);
    std::vector<float> result_f_i = pp.generate_descriptor(wl_float_int, data_written);
    std::vector<float> result_i_f = pp.generate_descriptor(wl_int_float, data_written);
    std::vector<float> result_f_f = pp.generate_descriptor(wl_float_float, data_written);
    std::vector<float> result_i4_f = pp.generate_descriptor(wl_int4_float, data_written);
    std::vector<float> result_bf8_f = pp.generate_descriptor(wl_bf8_float, data_written);

    std::cout << "\n descriptors\n";

    show_desc(result_i_i, "\nii:");
    show_desc(result_f_i, "\nfi:");
    show_desc(result_i_f, "\nif:");
    show_desc(result_f_f, "\nff:");
    show_desc(result_i4_f, "\ni4f:");
    show_desc(result_bf8_f, "\nbf8f:");
    std::cout << std::endl;

    // EXPECT_EQ(result_i_i, result_i_f);

    using datatypeContent = std::array<int, (int)intf_12::DataType::__size>;
    const datatypeContent uint8_content{1, 0, 0};
    const datatypeContent f16_content{0, 1, 0};
    const datatypeContent bf8_content{0, 0, 1};
    const int cntToCheck{(int)std::tuple_size<decltype(uint8_content)>{}};

    const int dtype_idx = 4;

    {
        auto& r = result_i_i;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i_f;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_i;
        auto& exp_in_0{f16_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_f;
        auto& exp_in_0{f16_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i4_f;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_bf8_f;
        auto& exp_in_0{bf8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
}

//------------------------------------------------------------------------------------------------
/// interface 13

class TestPreprocessing_Interface13 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const DPUWorkload wl = {
            VPUDevice::NPU_RESERVED,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // output dimensions
            {3, 3},                                                          // kernels
            {1, 1},                                                          // strides
            {1, 1},                                                          // padding
            ExecutionMode::CUBOID_16x16,                                     //
            ActivationFunction::NONE,                                        //
            0.7F,                                                            // act sparsity
            0.3F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // owtiles
            {0, 0, 0, 0},                                                    // offsets,
            ISIStrategy::SPLIT_OVER_H,                                       //
            false,                                                           // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 41, 42},      // input_0_halo
                    {5, 6, 7, 8, 81, 82},      // output_0_halo
                    {9, 10, 11, 12, 91, 92},   // output_0_halo_broadcast_cnt
                    {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
            },                                 // halo
            {
                    // sep
                    false,             // sep on
                    {2050, 22, 1, 1},  // sep table,  4 bytes per element
                    {512, 5, 64, 1},   // actual activator input,  same datatype as compute tensor input
                    false,
            }  // sep

    };

    const int descriptor_expected_size{44 + 3 + 3};
    const int tensor_Dtype_idx = 4;
    const int tensorDescriptorSize = tensor_Dtype_idx + (int)intf_13::DataType::__size;

    const int op_idx = 0;
    const int in_0_idx{op_idx + (int)intf_12::Operation::__size};
    const int in_1_dtype_idx{in_0_idx + tensorDescriptorSize};
    const int out_0_idx{in_1_dtype_idx + (int)intf_13::DataType::__size};
    const int ksp_idx = out_0_idx + tensorDescriptorSize;
    const int execution_mode_idx = ksp_idx + 8; /*kernel:2 +strides:2+padding:4*/
    const int sparsities_idx = execution_mode_idx + (int)intf_12::ExecutionMode::__size;
    const int owt_idx = sparsities_idx + 2;  /*sparsities:2*/
    const int odu_permute_idx = owt_idx + 1; /*owt:1*/
    const int inplace_output_idx = odu_permute_idx + (int)intf_12::Layout::__size;
    const int inplace_input1_idx = inplace_output_idx + 1;
    const int superdense_idx = inplace_input1_idx + 1;

    const int first_after_end_idx = superdense_idx + 1;
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface13, SingleWLTestPreprocessing) {
    EXPECT_EQ(first_after_end_idx, descriptor_expected_size);
    EXPECT_EQ(descriptor_expected_size, 44 + 3 + 3);
    // Instantiate the preprocessing class
    auto pp = Preprocessing_Interface13<float>();
    // Transform a single workload
    std::vector<float> result = pp.transformSingle(wl);
    EXPECT_EQ(result.size(), descriptor_expected_size);
}
// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface13, CreationAndSize) {
    auto pp = Preprocessing_Interface13<float>();
    EXPECT_EQ(pp.output_size(), descriptor_expected_size);  //

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)NNVersions::VERSION_13_NPU_RESERVED);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, descriptor_expected_size);  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
}

TEST_F(TestPreprocessing_Interface13, DescriptorContentTest) {
    const DPUWorkload wl2 = {
            VPUDevice::NPU_RESERVED,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {1, 2},                                                             // kernels
            {3, 4},                                                             // strides
            {5, 6, 7, 8},                                                       // padding
            ExecutionMode::CUBOID_4x16,                                         //
            ActivationFunction::NONE,                                           //
            0.11F,                                                              // act sparsity
            0.22F,                                                              // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                               // swiz in
            {Swizzling::KEY_0},                                                 // swiz out
            100,                                                                // owtiles
            {0, 0, 0, 0},                                                       // offsets,
            ISIStrategy::SPLIT_OVER_K,                                          //
            false,                                                              // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 5, 6},        // input_0_halo
                    {7, 8, 9, 10, 11, 12},     // output_0_halo
                    {13, 14, 15, 16, 17, 18},  // output_0_halo_broadcast_cnt
                    {19, 20, 21, 22, 23, 24},  // output_0_inbound_halo
            }                                  // halo
    };

    auto pp = Preprocessing_Interface13<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << x /*std::lround(x) */ << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[in_0_idx]), wl2.inputs[0].width());
    EXPECT_EQ(std::lround(result[in_0_idx + 1]), wl2.inputs[0].height());
    EXPECT_EQ(std::lround(result[in_0_idx + 2]), wl2.inputs[0].z());
    EXPECT_EQ(std::lround(result[in_0_idx + 3]), wl2.inputs[0].b());

    EXPECT_EQ(std::lround(result[out_0_idx + 0]), wl2.outputs[0].width());
    EXPECT_EQ(std::lround(result[out_0_idx + 1]), wl2.outputs[0].height());
    EXPECT_EQ(std::lround(result[out_0_idx + 2]), wl2.outputs[0].z());
    EXPECT_EQ(std::lround(result[out_0_idx + 3]), wl2.outputs[0].b());

    EXPECT_EQ(std::lround(result[ksp_idx + 0]), 1);
    EXPECT_EQ(std::lround(result[ksp_idx + 1]), 2);
    EXPECT_EQ(std::lround(result[ksp_idx + 2]), 3);
    EXPECT_EQ(std::lround(result[ksp_idx + 3]), 4);
    EXPECT_EQ(std::lround(result[ksp_idx + 4]), 5);
    EXPECT_EQ(std::lround(result[ksp_idx + 5]), 6);
    EXPECT_EQ(std::lround(result[ksp_idx + 6]), 7);
    EXPECT_EQ(std::lround(result[ksp_idx + 7]), 8);

    EXPECT_EQ(result[sparsities_idx + 0], wl2.act_sparsity);
    EXPECT_EQ(result[sparsities_idx + 1], wl2.weight_sparsity);

    EXPECT_EQ(std::lround(result[owt_idx + 0]), std::min(wl2.output_write_tiles, 2u));

    EXPECT_EQ(std::lround(result[odu_permute_idx + 0]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 4]), 1);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 5]), 0);

    EXPECT_EQ(std::lround(result[inplace_output_idx + 0]), 1);  // in out are the same
    EXPECT_EQ(std::lround(result[inplace_input1_idx + 0]), 0);  //
    EXPECT_EQ(std::lround(result[superdense_idx + 0]), 0);
}

TEST_F(TestPreprocessing_Interface13, DescriptorContentTest_AddedTypes) {
    const DPUWorkload wl2 = {
            VPUDevice::NPU_RESERVED,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},    // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::FLOAT32, Layout::XYZ, false)},  // output dimensions
            {1, 2},                                                               // kernels
            {3, 4},                                                               // strides
            {5, 6, 7, 8},                                                         // padding
            ExecutionMode::CUBOID_4x16,                                           //
            ActivationFunction::NONE,                                             //
            0.11F,                                                                // act sparsity
            0.22F,                                                                // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                                 // swiz in
            {Swizzling::KEY_0},                                                   // swiz out
            100,                                                                  // owtiles
            {0, 0, 0, 0},                                                         // offsets,
            ISIStrategy::SPLIT_OVER_K,                                            //
            false,                                                                // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 5, 6},        // input_0_halo
                    {7, 8, 9, 10, 11, 12},     // output_0_halo
                    {13, 14, 15, 16, 17, 18},  // output_0_halo_broadcast_cnt
                    {19, 20, 21, 22, 23, 24},  // output_0_inbound_halo
            },                                 // halo  (irelevant)
            {
                    // sep
                    false,             // sep on
                    {2050, 22, 1, 1},  // sep table,  4 bytes per element
                    {512, 5, 64, 1},   // actual activator input,  same datatype as compute tensor input
                    false,
            },                // sep
            DataType::UINT4,  // wts dataype special (optional)
            "No info",        // layet info
            true,             // weightless op (ELM need) (optional)
            true,             // in place output op (ELM need) (optional)
            true,             // superdense op (optional)
    };

    auto pp = Preprocessing_Interface13<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << x /*std::lround(x) */ << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[in_0_idx]), wl2.inputs[0].width());
    EXPECT_EQ(std::lround(result[in_0_idx + 1]), wl2.inputs[0].height());
    EXPECT_EQ(std::lround(result[in_0_idx + 2]), wl2.inputs[0].z());
    EXPECT_EQ(std::lround(result[in_0_idx + 3]), wl2.inputs[0].b());

    EXPECT_EQ((int)intf_12::DataType::__size, 3);

    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 0]), 1);  // UINT8
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 3]), 0);

    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 0]), 1);  // UINT8
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 3]), 0);

    EXPECT_EQ(std::lround(result[out_0_idx + 0]), wl2.outputs[0].width());
    EXPECT_EQ(std::lround(result[out_0_idx + 1]), wl2.outputs[0].height());
    EXPECT_EQ(std::lround(result[out_0_idx + 2]), wl2.outputs[0].z());
    EXPECT_EQ(std::lround(result[out_0_idx + 3]), wl2.outputs[0].b());

    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 0]), 0);
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 1]), 0);  // FP16 eq
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 3]), 1);  // FP32

    EXPECT_EQ(std::lround(result[ksp_idx + 0]), 1);
    EXPECT_EQ(std::lround(result[ksp_idx + 1]), 2);
    EXPECT_EQ(std::lround(result[ksp_idx + 2]), 3);
    EXPECT_EQ(std::lround(result[ksp_idx + 3]), 4);
    EXPECT_EQ(std::lround(result[ksp_idx + 4]), 5);
    EXPECT_EQ(std::lround(result[ksp_idx + 5]), 6);
    EXPECT_EQ(std::lround(result[ksp_idx + 6]), 7);
    EXPECT_EQ(std::lround(result[ksp_idx + 7]), 8);

    EXPECT_EQ(result[sparsities_idx + 0], wl2.act_sparsity);
    EXPECT_EQ(result[sparsities_idx + 1], wl2.weight_sparsity);

    EXPECT_EQ(std::lround(result[owt_idx + 0]), std::min(wl2.output_write_tiles, 2u));

    EXPECT_EQ(std::lround(result[odu_permute_idx + 0]), 1);  // xyz
    EXPECT_EQ(std::lround(result[odu_permute_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 4]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 5]), 0);

    EXPECT_EQ(std::lround(result[inplace_output_idx + 0]), 1);  // in out are the same
    EXPECT_EQ(std::lround(result[inplace_input1_idx + 0]), 1);  //
    EXPECT_EQ(std::lround(result[superdense_idx + 0]), 1);
}

TEST_F(TestPreprocessing_Interface13, DescriptorContentTest_FLOAT_INT) {
    class Builder {
    public:
        static DPUWorkload makeSameWl(DataType Tin, DataType Tout) {
            return DPUWorkload{
                    VPUDevice::NPU_RESERVED,
                    Operation::ELTWISE,
                    {VPUTensor(15, 1972, 16, 1, Tin)},   // input dimensions
                    {VPUTensor(15, 1972, 16, 1, Tout)},  // output dimensions
                    {1, 1},                              // kernels
                    {1, 1},                              // strides
                    {0, 0, 0, 0},                        // padding
                    ExecutionMode::CUBOID_4x16,          //
                    ActivationFunction::NONE,            //
                    0.F,                                 // act sparsity
                    0.F,                                 // weight_sparsity
                    {swz_def, swz_def},                  // input_swizzling
                    {swz_def},                           // output_swizzling
                    1,                                   // owtiles
                    {0, 0, 0, 0},                        // offsets,
                    ISIStrategy::CLUSTERING,             //
                    false,                               // weight_sparsity_enabled
                    {
                            {1, 2, 3, 4, 41, 42},      // input_0_halo
                            {5, 6, 7, 8, 81, 82},      // output_0_halo
                            {9, 10, 11, 12, 91, 92},   // output_0_halo_broadcast_cnt
                            {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
                    }                                  // halo
            };
        }
    };

    const DPUWorkload wl_int_int{Builder::makeSameWl(DataType::UINT8, DataType::UINT8)};
    const DPUWorkload wl_float_float{Builder::makeSameWl(DataType::FLOAT16, DataType::FLOAT16)};
    const DPUWorkload wl_float_int{Builder::makeSameWl(DataType::FLOAT16, DataType::UINT8)};
    const DPUWorkload wl_int_float{Builder::makeSameWl(DataType::UINT8, DataType::FLOAT16)};
    const DPUWorkload wl_int4_float{Builder::makeSameWl(DataType::UINT4, DataType::FLOAT16)};
    const DPUWorkload wl_bf8_float{Builder::makeSameWl(DataType::BF8, DataType::FLOAT16)};

    auto show_desc = [](const std::vector<float>& result, std::string prefix) {
        std::cout << prefix;
        for (auto x : result) {
            std::cout << " " << std::lround(x) << " ";
        }
    };

    auto pp = Preprocessing_Interface13<float>();
    size_t data_written = 0;
    std::vector<float> result_i_i = pp.generate_descriptor(wl_int_int, data_written);
    std::vector<float> result_f_i = pp.generate_descriptor(wl_float_int, data_written);
    std::vector<float> result_i_f = pp.generate_descriptor(wl_int_float, data_written);
    std::vector<float> result_f_f = pp.generate_descriptor(wl_float_float, data_written);
    std::vector<float> result_i4_f = pp.generate_descriptor(wl_int4_float, data_written);
    std::vector<float> result_bf8_f = pp.generate_descriptor(wl_bf8_float, data_written);

    std::cout << "\n descriptors\n";

    show_desc(result_i_i, "\nii:");
    show_desc(result_f_i, "\nfi:");
    show_desc(result_i_f, "\nif:");
    show_desc(result_f_f, "\nff:");
    show_desc(result_i4_f, "\ni4f:");
    show_desc(result_bf8_f, "\nbf8f:");
    std::cout << std::endl;

    // EXPECT_EQ(result_i_i, result_i_f);

    using datatypeContent = std::array<int, (int)intf_12::DataType::__size>;
    const datatypeContent uint8_content{1, 0, 0};
    const datatypeContent f16_content{0, 1, 0};
    const datatypeContent bf8_content{0, 0, 1};
    const int cntToCheck{(int)std::tuple_size<decltype(uint8_content)>{}};

    const int dtype_idx = 4;

    {
        auto& r = result_i_i;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i_f;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_i;
        auto& exp_in_0{f16_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_f;
        auto& exp_in_0{f16_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i4_f;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_bf8_f;
        auto& exp_in_0{bf8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
}

//------------------------------------------------------------------------------------------------
/// interface 14

class TestPreprocessing_Interface14 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const DPUWorkload wl = {
            VPUDevice::NPU_RESERVED,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8, Layout::XYZ, true)},  // output dimensions
            {3, 3},                                                          // kernels
            {1, 1},                                                          // strides
            {1, 1},                                                          // padding
            ExecutionMode::CUBOID_16x16,                                     //
            ActivationFunction::NONE,                                        //
            0.7F,                                                            // act sparsity
            0.3F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // owtiles
            {0, 0, 0, 0},                                                    // offsets,
            ISIStrategy::SPLIT_OVER_H,                                       //
            false,                                                           // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 41, 42},      // input_0_halo
                    {5, 6, 7, 8, 81, 82},      // output_0_halo
                    {9, 10, 11, 12, 91, 92},   // output_0_halo_broadcast_cnt
                    {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
            },                                 // halo
            {
                    // sep
                    false,             // sep on
                    {2050, 22, 1, 1},  // sep table,  4 bytes per element
                    {512, 5, 64, 1},   // actual activator input,  same datatype as compute tensor input
                    false,
            }  // sep

    };

    const int descriptor_expected_size{44 + 3 + 3 + 3};
    const int tensor_Dtype_idx = 4;  // 4D shape
    const int tensorDescriptorSize = tensor_Dtype_idx + (int)intf_14::DataType::__size;

    const int op_idx = 0;
    const int in_0_idx{op_idx + (int)intf_12::Operation::__size};
    const int in_1_dtype_idx{in_0_idx + tensorDescriptorSize};
    const int out_0_idx{in_1_dtype_idx + (int)intf_14::DataType::__size};
    const int ksp_idx = out_0_idx + tensorDescriptorSize;
    const int execution_mode_idx = ksp_idx + 8; /*kernel:2 +strides:2+padding:4*/
    const int sparsities_idx = execution_mode_idx + (int)intf_12::ExecutionMode::__size;
    const int owt_idx = sparsities_idx + 2;  /*sparsities:2*/
    const int odu_permute_idx = owt_idx + 1; /*owt:1*/
    const int inplace_output_idx = odu_permute_idx + (int)intf_12::Layout::__size;
    const int inplace_input1_idx = inplace_output_idx + 1;
    const int superdense_idx = inplace_input1_idx + 1;

    const int first_after_end_idx = superdense_idx + 1;
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface14, SingleWLTestPreprocessing) {
    EXPECT_EQ(first_after_end_idx, descriptor_expected_size);
    EXPECT_EQ(descriptor_expected_size, 44 + 3 + 3 + 3);
    // Instantiate the preprocessing class
    auto pp = Preprocessing_Interface14<float>();
    // Transform a single workload
    std::vector<float> result = pp.transformSingle(wl);
    EXPECT_EQ(result.size(), descriptor_expected_size);
}
// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface14, CreationAndSize) {
    auto pp = Preprocessing_Interface14<float>();
    EXPECT_EQ(pp.output_size(), descriptor_expected_size);  //

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)NNVersions::VERSION_14_NPU_RESERVED);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, descriptor_expected_size);  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
}

TEST_F(TestPreprocessing_Interface14, DescriptorContentTest) {
    const DPUWorkload wl2 = {
            VPUDevice::NPU_RESERVED,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {1, 2},                                                             // kernels
            {3, 4},                                                             // strides
            {5, 6, 7, 8},                                                       // padding
            ExecutionMode::CUBOID_4x16,                                         //
            ActivationFunction::NONE,                                           //
            0.11F,                                                              // act sparsity
            0.22F,                                                              // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                               // swiz in
            {Swizzling::KEY_0},                                                 // swiz out
            100,                                                                // owtiles
            {0, 0, 0, 0},                                                       // offsets,
            ISIStrategy::SPLIT_OVER_K,                                          //
            false,                                                              // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 5, 6},        // input_0_halo
                    {7, 8, 9, 10, 11, 12},     // output_0_halo
                    {13, 14, 15, 16, 17, 18},  // output_0_halo_broadcast_cnt
                    {19, 20, 21, 22, 23, 24},  // output_0_inbound_halo
            }                                  // halo
    };

    auto pp = Preprocessing_Interface14<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << x /*std::lround(x) */ << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[in_0_idx]), wl2.inputs[0].width());
    EXPECT_EQ(std::lround(result[in_0_idx + 1]), wl2.inputs[0].height());
    EXPECT_EQ(std::lround(result[in_0_idx + 2]), wl2.inputs[0].z());
    EXPECT_EQ(std::lround(result[in_0_idx + 3]), wl2.inputs[0].b());

    EXPECT_EQ(std::lround(result[out_0_idx + 0]), wl2.outputs[0].width());
    EXPECT_EQ(std::lround(result[out_0_idx + 1]), wl2.outputs[0].height());
    EXPECT_EQ(std::lround(result[out_0_idx + 2]), wl2.outputs[0].z());
    EXPECT_EQ(std::lround(result[out_0_idx + 3]), wl2.outputs[0].b());

    EXPECT_EQ(std::lround(result[ksp_idx + 0]), 1);
    EXPECT_EQ(std::lround(result[ksp_idx + 1]), 2);
    EXPECT_EQ(std::lround(result[ksp_idx + 2]), 3);
    EXPECT_EQ(std::lround(result[ksp_idx + 3]), 4);
    EXPECT_EQ(std::lround(result[ksp_idx + 4]), 5);
    EXPECT_EQ(std::lround(result[ksp_idx + 5]), 6);
    EXPECT_EQ(std::lround(result[ksp_idx + 6]), 7);
    EXPECT_EQ(std::lround(result[ksp_idx + 7]), 8);

    EXPECT_EQ(result[sparsities_idx + 0], wl2.act_sparsity);
    EXPECT_EQ(result[sparsities_idx + 1], wl2.weight_sparsity);

    EXPECT_EQ(std::lround(result[owt_idx + 0]), std::min(wl2.output_write_tiles, 2u));

    EXPECT_EQ(std::lround(result[odu_permute_idx + 0]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 4]), 1);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 5]), 0);

    EXPECT_EQ(std::lround(result[inplace_output_idx + 0]), 1);  // in out are the same
    EXPECT_EQ(std::lround(result[inplace_input1_idx + 0]), 0);  //
    EXPECT_EQ(std::lround(result[superdense_idx + 0]), 0);
}

TEST_F(TestPreprocessing_Interface14, DescriptorContentTest_AddedTypes) {
    const DPUWorkload wl2 = {
            VPUDevice::NPU_RESERVED,
            Operation::ELTWISE,
            {VPUTensor(15, 1972, 16, 1, DataType::UINT8, Layout::ZXY, false)},    // input dimensions
            {VPUTensor(15, 1972, 16, 1, DataType::FLOAT32, Layout::XYZ, false)},  // output dimensions
            {1, 2},                                                               // kernels
            {3, 4},                                                               // strides
            {5, 6, 7, 8},                                                         // padding
            ExecutionMode::CUBOID_4x16,                                           //
            ActivationFunction::NONE,                                             //
            0.11F,                                                                // act sparsity
            0.22F,                                                                // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_1},                                 // swiz in
            {Swizzling::KEY_0},                                                   // swiz out
            100,                                                                  // owtiles
            {0, 0, 0, 0},                                                         // offsets,
            ISIStrategy::SPLIT_OVER_K,                                            //
            false,                                                                // weight_sparsity_enabled
            {
                    {1, 2, 3, 4, 5, 6},        // input_0_halo
                    {7, 8, 9, 10, 11, 12},     // output_0_halo
                    {13, 14, 15, 16, 17, 18},  // output_0_halo_broadcast_cnt
                    {19, 20, 21, 22, 23, 24},  // output_0_inbound_halo
            },                                 // halo  (irelevant)
            {
                    // sep
                    false,             // sep on
                    {2050, 22, 1, 1},  // sep table,  4 bytes per element
                    {512, 5, 64, 1},   // actual activator input,  same datatype as compute tensor input
                    false,
            },                // sep
            DataType::UINT4,  // wts dataype special (optional)
            "No info",        // layet info
            true,             // weightless op (ELM need) (optional)
            true,             // in place output op (ELM need) (optional)
            true,             // superdense op (optional)
    };

    auto pp = Preprocessing_Interface14<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << x /*std::lround(x) */ << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    EXPECT_EQ(std::lround(result[in_0_idx]), wl2.inputs[0].width());
    EXPECT_EQ(std::lround(result[in_0_idx + 1]), wl2.inputs[0].height());
    EXPECT_EQ(std::lround(result[in_0_idx + 2]), wl2.inputs[0].z());
    EXPECT_EQ(std::lround(result[in_0_idx + 3]), wl2.inputs[0].b());

    EXPECT_EQ((int)intf_14::DataType::__size, 5);

    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 0]), 1);  // UINT8
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[in_0_idx + tensor_Dtype_idx + 4]), 0);

    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 0]), 0);  // UINT8 was mapped in intf13
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[in_1_dtype_idx + 4]), 1);  // UINT4

    EXPECT_EQ(std::lround(result[out_0_idx + 0]), wl2.outputs[0].width());
    EXPECT_EQ(std::lround(result[out_0_idx + 1]), wl2.outputs[0].height());
    EXPECT_EQ(std::lround(result[out_0_idx + 2]), wl2.outputs[0].z());
    EXPECT_EQ(std::lround(result[out_0_idx + 3]), wl2.outputs[0].b());

    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 0]), 0);
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 1]), 0);  // FP16 eq
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 3]), 1);  // FP32
    EXPECT_EQ(std::lround(result[out_0_idx + tensor_Dtype_idx + 4]), 0);

    EXPECT_EQ(std::lround(result[ksp_idx + 0]), 1);
    EXPECT_EQ(std::lround(result[ksp_idx + 1]), 2);
    EXPECT_EQ(std::lround(result[ksp_idx + 2]), 3);
    EXPECT_EQ(std::lround(result[ksp_idx + 3]), 4);
    EXPECT_EQ(std::lround(result[ksp_idx + 4]), 5);
    EXPECT_EQ(std::lround(result[ksp_idx + 5]), 6);
    EXPECT_EQ(std::lround(result[ksp_idx + 6]), 7);
    EXPECT_EQ(std::lround(result[ksp_idx + 7]), 8);

    EXPECT_EQ(result[sparsities_idx + 0], wl2.act_sparsity);
    EXPECT_EQ(result[sparsities_idx + 1], wl2.weight_sparsity);

    EXPECT_EQ(std::lround(result[owt_idx + 0]), std::min(wl2.output_write_tiles, 2u));

    EXPECT_EQ(std::lround(result[odu_permute_idx + 0]), 1);  // xyz
    EXPECT_EQ(std::lround(result[odu_permute_idx + 1]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 2]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 3]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 4]), 0);
    EXPECT_EQ(std::lround(result[odu_permute_idx + 5]), 0);

    EXPECT_EQ(std::lround(result[inplace_output_idx + 0]), 1);  // in out are the same
    EXPECT_EQ(std::lround(result[inplace_input1_idx + 0]), 1);  //
    EXPECT_EQ(std::lround(result[superdense_idx + 0]), 1);
}

TEST_F(TestPreprocessing_Interface14, DescriptorContentTest_FLOAT_INT) {
    class Builder {
    public:
        static DPUWorkload makeSameWl(DataType Tin, DataType Tout) {
            return DPUWorkload{
                    VPUDevice::NPU_RESERVED,
                    Operation::ELTWISE,
                    {VPUTensor(15, 1972, 16, 1, Tin)},   // input dimensions
                    {VPUTensor(15, 1972, 16, 1, Tout)},  // output dimensions
                    {1, 1},                              // kernels
                    {1, 1},                              // strides
                    {0, 0, 0, 0},                        // padding
                    ExecutionMode::CUBOID_4x16,          //
                    ActivationFunction::NONE,            //
                    0.F,                                 // act sparsity
                    0.F,                                 // weight_sparsity
                    {swz_def, swz_def},                  // input_swizzling
                    {swz_def},                           // output_swizzling
                    1,                                   // owtiles
                    {0, 0, 0, 0},                        // offsets,
                    ISIStrategy::CLUSTERING,             //
                    false,                               // weight_sparsity_enabled
                    {
                            {1, 2, 3, 4, 41, 42},      // input_0_halo
                            {5, 6, 7, 8, 81, 82},      // output_0_halo
                            {9, 10, 11, 12, 91, 92},   // output_0_halo_broadcast_cnt
                            {13, 14, 15, 16, 17, 18},  // output_0_inbound_halo
                    }                                  // halo
            };
        }
    };

    const DPUWorkload wl_int_int{Builder::makeSameWl(DataType::UINT8, DataType::UINT8)};
    const DPUWorkload wl_float_float{Builder::makeSameWl(DataType::FLOAT16, DataType::FLOAT16)};
    const DPUWorkload wl_float_int{Builder::makeSameWl(DataType::FLOAT16, DataType::UINT8)};
    const DPUWorkload wl_int_float{Builder::makeSameWl(DataType::UINT8, DataType::FLOAT16)};
    const DPUWorkload wl_int4_float{Builder::makeSameWl(DataType::UINT4, DataType::FLOAT16)};
    const DPUWorkload wl_bf8_float{Builder::makeSameWl(DataType::BF8, DataType::FLOAT16)};

    auto show_desc = [](const std::vector<float>& result, std::string prefix) {
        std::cout << prefix;
        for (auto x : result) {
            std::cout << " " << std::lround(x) << " ";
        }
    };

    auto pp = Preprocessing_Interface14<float>();
    size_t data_written = 0;
    std::vector<float> result_i_i = pp.generate_descriptor(wl_int_int, data_written);
    std::vector<float> result_f_i = pp.generate_descriptor(wl_float_int, data_written);
    std::vector<float> result_i_f = pp.generate_descriptor(wl_int_float, data_written);
    std::vector<float> result_f_f = pp.generate_descriptor(wl_float_float, data_written);
    std::vector<float> result_i4_f = pp.generate_descriptor(wl_int4_float, data_written);
    std::vector<float> result_bf8_f = pp.generate_descriptor(wl_bf8_float, data_written);

    std::cout << "\n descriptors\n";

    show_desc(result_i_i, "\nii:");
    show_desc(result_f_i, "\nfi:");
    show_desc(result_i_f, "\nif:");
    show_desc(result_f_f, "\nff:");
    show_desc(result_i4_f, "\ni4f:");
    show_desc(result_bf8_f, "\nbf8f:");
    std::cout << std::endl;

    // EXPECT_EQ(result_i_i, result_i_f);

    using datatypeContent = std::array<int, (int)intf_14::DataType::__size>;
    const datatypeContent uint8_content{1, 0, 0, 0, 0};
    const datatypeContent f16_content{0, 1, 0, 0, 0};
    const datatypeContent bf8_content{0, 0, 1, 0, 0};
    const datatypeContent uint4_content{0, 0, 0, 0, 1};
    const int cntToCheck{(int)std::tuple_size<decltype(uint8_content)>{}};

    const int dtype_idx = 4;  // 4 channels before dtype

    {
        auto& r = result_i_i;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i_f;
        auto& exp_in_0{uint8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_i;
        auto& exp_in_0{f16_content};
        auto& exp_out_0{uint8_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_f_f;
        auto& exp_in_0{f16_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_i4_f;
        auto& exp_in_0{uint4_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
    {
        auto& r = result_bf8_f;
        auto& exp_in_0{bf8_content};
        auto& exp_out_0{f16_content};

        for (int i = 0; i < cntToCheck; ++i) {
            EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + i]), exp_in_0[i]) << i;
            EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + i]), exp_out_0[i]) << i;
        }
    }
}

TEST_F(TestPreprocessing_Interface14, wI2_MappingTo_wI4) {
    using datatypeContent = std::array<int, (int)intf_14::DataType::__size>;
    const datatypeContent uint4_content{0, 0, 0, 0, 1};

    auto pp = Preprocessing_Interface14<float>();
    size_t data_written = 0;

    auto wl_int2 = DPUWorkload(wl);
    auto wl_uint2 = DPUWorkload(wl);
    wl_int2.weight_type = DataType::INT2;
    wl_uint2.weight_type = DataType::UINT2;

    auto desc_int2 = pp.generate_descriptor(wl_int2, data_written);
    auto desc_uint2 = pp.generate_descriptor(wl_uint2, data_written);

    for (int i = 0; i < (int)intf_14::DataType::__size; ++i) {
        EXPECT_EQ(std::lround(desc_int2[in_1_dtype_idx + i]), uint4_content[i]) << i;
        EXPECT_EQ(std::lround(desc_uint2[in_1_dtype_idx + i]), uint4_content[i]) << i;
    }
}

TEST_F(TestPreprocessing_Interface14, EISXW164800_Tensor_CH_padding) {
    auto pp = Preprocessing_Interface14<float>();
    size_t data_written = 0;
    DPUWorkload wl_ref{
            VPUDevice::NPU_RESERVED,
            Operation::ELTWISE,
            {VPUTensor(28, 9, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(28, 9, 1, 1, DataType::UINT8)},     // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_8x16,                    // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    wl_ref.superdense_memory = true;

    DPUWorkload wl_output_autopad{wl_ref};
    wl_output_autopad.output_autopad = true;

    auto desc_autopad = pp.generate_descriptor(wl_output_autopad, data_written);
    EXPECT_EQ(desc_autopad[out_0_idx + 2], 16);  // output z dimension must be padded to 16

    DPUWorkload wl_input_autopad{wl_ref};
    wl_input_autopad.input_autopad = true;
    wl_input_autopad.inputs[0] = VPUTensor(
            {wl_ref.inputs[0].width(), wl_ref.inputs[0].height(), 5, wl_ref.inputs[0].batches()}, wl_ref.inputs[0]);
    wl_input_autopad.outputs[0] =
            VPUTensor({wl_ref.outputs[0].width(), wl_ref.outputs[0].height(), 16, wl_ref.outputs[0].batches()},
                      wl_ref.outputs[0]);

    auto desc_input_autopad = pp.generate_descriptor(wl_input_autopad, data_written);
    EXPECT_EQ(desc_input_autopad[in_0_idx + 2], 16);  // input z dimension must be padded to 16
}

// end ITF14

}  // namespace VPUNN_unit_tests
