// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common/common_helpers.h"
#include "vpu/compatibility/types01.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu_dma_cost_model.h"

#include "vpu/validation/dpu_operations_validator.h"
#include "vpu/validation/memory_calculator.h"

#include <algorithm>
#include <unordered_map>

#include <optional>
#include <variant>

#include "regression_test.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

// Add Regression unit tests for LNL here

class Regression_tests_DW_CONV_EISXW117314_NPU40 : public Regression_Tests {
    // base wl, before split
    // const DPUWorkload wl_2d_3{
    //         // orig Layer
    //         VPUDevice::VPU_4_0,
    //         Operation::DW_CONVOLUTION,
    //         {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // input dimensions
    //         {VPUTensor(56, 56, 128, 1, DataType::UINT8)},  // output dimensions
    //         {3, 3},                                        // kernels
    //         {1, 1},                                        // strides
    //         {1, 1, 1, 1},                                  // padding
    //         ExecutionMode::CUBOID_16x16,                   // execution mode
    //         ActivationFunction::NONE,                      // activation
    //         0.0F,                                          // act_sparsity
    //         0.0F,                                          // weight_sparsity
    //         {swz_def, swz_def},                            // input_swizzling
    //         {swz_def},                                     // output_swizzling
    //         1,                                             // output_write_tiles
    //         {0, 0, 0, 0},                                  // offsets
    //         ISIStrategy::CLUSTERING,                       // isi_strategy
    //         false,                                         // weight_sparsity_enabled
    // };

public:
protected:
    const DPUWorkload base_wl{
            VPUDevice::VPU_4_0,
            Operation::DW_CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::CLUSTERING,                                    // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    const DPUWorkload base_wl_SOK_CLU{
            VPUDevice::VPU_4_0,
            Operation::DW_CONVOLUTION,
            {VPUTensor(56, 56, 128, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(56, 56, 128, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            4,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::SPLIT_OVER_K,                                  // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    /// with this function, starting from a SOHO base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param in_w: the input tensor width
    /// @param in_h: the input tensor height
    /// @param out_w: the output tensor width
    /// @param out_h: output tensor height
    /// @param top_padd: top padding for workload
    /// @param btm_padd: bottom padding for workload
    /// @param Tint: input tensor Dtype
    /// @param Tout: output tensor Dtype
    /// @param kw: kernel width
    /// @param kh: kernel height
    /// @param stride: wl stride
    /// @return a new DPUWorkload
    static DPUWorkload mk_SOHO(unsigned int in_w, unsigned int in_h, unsigned int out_w, unsigned int out_h,
                               unsigned int top_padd, unsigned int btm_padd, unsigned int l_padd, unsigned int r_padd,
                               DataType Tin, DataType Tout, unsigned int kw, unsigned int kh, unsigned int stride = 1) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(in_w, in_h, 128, 1, Tin, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, 128, 1, Tout, Layout::ZXY)},  // output dimensions
                {kw, kh},                                              // kernels
                {stride, stride},                                      // strides
                {top_padd, btm_padd, l_padd, r_padd},                  // padding
                ExecutionMode::CUBOID_16x16,                           // execution mode
                ActivationFunction::NONE,                              // activation
                0.0F,                                                  // act_sparsity
                0.0F,                                                  // weight_sparsity
                {swz_def, swz_def},                                    // input_swizzling
                {swz_def},                                             // output_swizzling
                1,                                                     // output_write_tiles
                {0, 0, 0, 0},                                          // offsets
                ISIStrategy::CLUSTERING,                               // isi_strategy
                false,                                                 //
        };
    }

    /// with this function, starting from a SOK base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param in_w: the input tensor width
    /// @param in_h: the input tensor height
    /// @param out_w: the output tensor width
    /// @param out_h: output tensor height
    /// @param Tint: input tensor Dtype
    /// @param Tout: output tensor Dtype
    /// @param kw: kernel width
    /// @param kh: kernel height
    /// @param stride: wl stride
    /// @return a new DPUWorkload
    static DPUWorkload mk_SOK(unsigned int in_w, unsigned int in_h, unsigned int out_w, unsigned int out_h,
                              DataType Tin, DataType Tout, unsigned int kw, unsigned int kh, unsigned int padd = 1,
                              unsigned int stride = 1) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(in_w, in_h, 128, 1, Tin, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, 128, 1, Tout, Layout::ZXY)},  // output dimensions
                {kw, kh},                                              // kernels
                {stride, stride},                                      // strides
                {padd, padd, padd, padd},                              // padding
                ExecutionMode::CUBOID_16x16,                           // execution mode
                ActivationFunction::NONE,                              // activation
                0.0F,                                                  // act_sparsity
                0.0F,                                                  // weight_sparsity
                {swz_def, swz_def},                                    // input_swizzling
                {swz_def},                                             // output_swizzling
                4,                                                     // output_write_tiles
                {0, 0, 0, 0},                                          // offsets
                ISIStrategy::SPLIT_OVER_K,                             // isi_strategy
                false,                                                 //
        };
    }

    // this enum help us to better visualize the number of channels of a tensor
    enum Channels {
        ch64 = 0,
        ch32 = 1,
        ch16 = 2

    };

    Regression_tests_DW_CONV_EISXW117314_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              DataType_ToText.at(static_cast<int>(wl.inputs[0].get_dtype())) + 
                              " C:" + std::to_string(wl.inputs[0].get_shape()[2]) + 
                              " k:" + std::to_string(wl.kernels[Dim::Grid::W]) + "x" + std::to_string(wl.kernels[Dim::Grid::H]) + 
                              " s:" + std::to_string(wl.strides[Dim::Grid::W]) + "x" + std::to_string(wl.strides[Dim::Grid::H]) +"\n";

        // clang-format on

        return message;
    }
};
TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_3x3) {
    const int p{1};
    const int k{3};
    // STRIDE=1
    DPUWorkload wl_CLU_top{mk_SOHO(56, 15, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    DPUWorkload wl_CLU_mid{mk_SOHO(56, 16, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    DPUWorkload wl_CLU_btm{mk_SOHO(56, 15, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    DPUWorkload wl_CLU_no_broadcast{mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
        //SOHO
            {tc(wls_CLU_top[ch64], 4335,  "top")},  // 3439   GT:4335
            {tc(wls_CLU_top[ch32], 1412,  "top")},  // 2352   GT:1412
            {tc(wls_CLU_top[ch16], 1409,  "top")},  // 2139   GT:1409   

            {tc(wls_CLU_mid[ch64], 4406,  "middle")},  // 3441   GT:4406
            {tc(wls_CLU_mid[ch32], 1410,  "middle")},  // 2352   GT:1410
            {tc(wls_CLU_mid[ch16], 1404,  "middle")},  // 2136   GT:1404
                                                
            {tc(wls_CLU_btm[ch64], 4355,  "bottom")},  // 3457   GT:4355
            {tc(wls_CLU_btm[ch32], 1400,  "bottom")},  // 2342   GT:1400
            {tc(wls_CLU_btm[ch16], 1391,  "bottom")},  // 2117   GT:1391  

           //CLU + no broadcast
            {tc(wls_CLU_no_broadcast[ch64], 14578, "CLU + no broadcast")},  // 21646  GT:14578
            {tc(wls_CLU_no_broadcast[ch32], 4549,  "CLU + no broadcast")},  // 9633   GT:4549
            {tc(wls_CLU_no_broadcast[ch16], 4563,  "CLU + no broadcast")},  // 9087   GT:4563

           //SOK
            {tc(wls_SOK[ch64], 14668,  "SOK, all tensors are the same")},  // 22035   GT:14668
            {tc(wls_SOK[ch32], 4807,  "SOK, all tensors are the same")},  // 9724    GT:4807
            {tc(wls_SOK[ch16], 4854 ,  "SOK, all tensors are the same")},  // 2117    GT:4854

            // clang-format on
    };

    executeTests(tests);

    //****************************************************************************************************************************************************
    // STRIDE=2
    DPUWorkload wl_CLU_top_stride2{mk_SOHO(112, 28, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    DPUWorkload wl_CLU_mid_stride2{mk_SOHO(112, 29, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    DPUWorkload wl_CLU_btm_stride2{mk_SOHO(112, 28, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};

    DPUWorkload wl_CLU_stride2_no_broadcast{
            mk_SOHO(112, 112, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};

    DPUWorkload wl_SOK_stride2{mk_SOK(112, 112, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_stride2_no_broadcast{change_wl_channels_64_32_16(wl_CLU_stride2_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};
    wls_SOK_stride2[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
        //SOHO
            {tc(wls_CLU_top_stride2[ch64], 4572, "top")},  // 3438   GT:4572
            {tc(wls_CLU_top_stride2[ch32], 2248, "top")},  // 2348   GT:2248  v17:1501
            {tc(wls_CLU_top_stride2[ch16], 2282, "top")},  // 2140   GT:2282  v17:1562 
                                                                                      
            {tc(wls_CLU_mid_stride2[ch64], 4547, "middle")},  // 3440   GT:4547
            {tc(wls_CLU_mid_stride2[ch32], 2293, "middle")},  // 2349   GT:2293  v17:1532
            {tc(wls_CLU_mid_stride2[ch16], 2282, "middle")},  // 2136   GT:2282  v17:1589
                                                                 
            {tc(wls_CLU_btm_stride2[ch64], 4576, "bottom")},  // 3453   GT:4576
            {tc(wls_CLU_btm_stride2[ch32], 2292, "bottom")},  // 2339   GT:2292  v17:1509 
            {tc(wls_CLU_btm_stride2[ch16], 2282, "bottom")},  // 2121   GT:2282  v17:1568 
                                                                                   
           //CLU + no broadcast                                                                
           {tc(wls_CLU_stride2_no_broadcast[ch64], 15465, "CLU + no broadcast")},  // 21536  GT:15465
           {tc(wls_CLU_stride2_no_broadcast[ch32], 7681,  "CLU + no broadcast" )},  // 9656   GT:7681  v17:5369
           {tc(wls_CLU_stride2_no_broadcast[ch16], 7708,  "CLU + no broadcast")},  // 9112   GT:7708  v17:5642

            //SOK
            //{tc(wls_SOK_stride2[ch64], 16465,  "SOK, all tensors are the same")},    // 21896  GT:16465  because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK_stride2[ch32], 7818,     "SOK, all tensors are the same")},  // 9745   GT:7818
            {tc(wls_SOK_stride2[ch16], 8252  ,   "SOK, all tensors are the same")},  // 9332   GT:8252

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_3x5) {
    DPUWorkload wl_SOK{mk_SOK(56, 58, 56, 56, DataType::UINT8, DataType::UINT8, 3, 5)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller
    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 24080,  "SOK, all tensors are the same")},  //    GT:25529
            {tc(wls_SOK[ch32], 7301,  "SOK, all tensors are the same")},  //    GT:12655
            {tc(wls_SOK[ch16], 7395,  "SOK, all tensors are the same")},  //    GT:12900

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_5x3) {
    DPUWorkload wl_SOK{mk_SOK(58, 56, 56, 56, DataType::UINT8, DataType::UINT8, 5, 3)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 25545,  "SOK, all tensors are the same")},  // 26708   GT:24078
            {tc(wls_SOK[ch32], 12718 ,  "SOK, all tensors are the same")},  // 13313   GT:7456 
            {tc(wls_SOK[ch16], 12953 ,  "SOK, all tensors are the same")},  // 12736   GT:7490

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_3x7) {
    DPUWorkload wl_SOK{mk_SOK(56, 60, 56, 56, DataType::UINT8, DataType::UINT8, 3, 7)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 33485,  "SOK, all tensors are the same")},  //    GT:35487
            {tc(wls_SOK[ch32], 10009,  "SOK, all tensors are the same")},  //    GT:17604
            {tc(wls_SOK[ch16], 10032,  "SOK, all tensors are the same")},  //    GT:17841

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_7x3) {
    DPUWorkload wl_SOK{mk_SOK(60, 56, 56, 56, DataType::UINT8, DataType::UINT8, 7, 3)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 35532,  "SOK, all tensors are the same")},  //    GT:33552
            {tc(wls_SOK[ch32], 17712,  "SOK, all tensors are the same")},  //    GT:10157
            {tc(wls_SOK[ch16], 17966,  "SOK, all tensors are the same")},  //    GT:10172

            // clang-format on
    };

    executeTests(tests);
}
TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, FLOAT16_kernel_3x3) {
    const int p{1};
    const int k{3};
    // STRIDE=1
    DPUWorkload wl_CLU_top{mk_SOHO(56, 15, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    DPUWorkload wl_CLU_mid{mk_SOHO(56, 16, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    DPUWorkload wl_CLU_btm{mk_SOHO(56, 15, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    DPUWorkload wl_CLU_no_broadcast{mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
        //SOHO 
            {tc(wls_CLU_top[ch64], 8995,  "top")},  // 7737  GT:8995
            {tc(wls_CLU_top[ch32], 3106,  "top")},  // 4180   GT:3106
            {tc(wls_CLU_top[ch16], 1575,  "top")},  // 2297   GT:1575   
                                                     
            {tc(wls_CLU_mid[ch64], 8991,  "middle")},  // 7735   GT:8991
            {tc(wls_CLU_mid[ch32], 3072,  "middle")},  // 4193   GT:3072
            {tc(wls_CLU_mid[ch16], 1621,  "middle")},  // 2305   GT:1621  
                                                     
            {tc(wls_CLU_btm[ch64], 9062,  "bottom")},  // 7712   GT:9062
            {tc(wls_CLU_btm[ch32], 3072,  "bottom")},  // 4191   GT:3072
            {tc(wls_CLU_btm[ch16], 1566,  "bottom")},  // 2270   GT:1566  

           //CLU + no broadcast
            {tc( wls_CLU_no_broadcast[ch64], 29153,"CLU + no broadcast")},  // 50494  GT:29153
            {tc( wls_CLU_no_broadcast[ch32], 9413, "CLU + no broadcast", pc, 27)},  // 19732  GT:9413  v17: 10377 ---> delta 10.24%
            {tc( wls_CLU_no_broadcast[ch16], 4692, "CLU + no broadcast")},  // 8951   GT:4692  v17: 5303 --->delta 13%

           //SOK
            //{tc(wls_SOK[ch64], 30153,  "SOK, all tensors are the same")},  // 51880   GT:30153 because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK[ch32], 9454,     "SOK, all tensors are the same", pc, 42)},  // 19733   GT:9454
            {tc(wls_SOK[ch16], 4756 ,    "SOK, all tensors are the same")},  // 8966   GT:4756

            // clang-format on
    };

    executeTests(tests);

    //****************************************************************************************************************************************************8
    // STRIDE=2
    DPUWorkload wl_CLU_top_stride2{mk_SOHO(112, 28, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    DPUWorkload wl_CLU_mid_stride2{mk_SOHO(112, 29, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    DPUWorkload wl_CLU_btm_stride2{mk_SOHO(112, 28, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};

    DPUWorkload wl_CLU_stride2_no_broadcast{
            mk_SOHO(112, 112, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    DPUWorkload wl_SOK_stride2{mk_SOK(112, 112, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_stride2_no_broadcast{change_wl_channels_64_32_16(wl_CLU_stride2_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};
    wls_SOK_stride2[ch16].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
        //SOHO
            {tc(wls_CLU_top_stride2[ch64], 9352,  "top")},  // 7722   GT:9352
            {tc(wls_CLU_top_stride2[ch32], 4602,  "top")},  // 4193   GT:4602  v17:3086
            {tc(wls_CLU_top_stride2[ch16], 2371,  "top")}, // 2297   GT:2371  v17:1719  
                                                                                                                 
            {tc(wls_CLU_mid_stride2[ch64], 9311,  "middle")},  // 7720   GT:9311
            {tc(wls_CLU_mid_stride2[ch32], 4602,  "middle")},  // 4206   GT:4602   v17:3148
            {tc(wls_CLU_mid_stride2[ch16], 2371,  "middle")},  // 2305   GT:2371   v17:1752
                                                                                                                  
            {tc(wls_CLU_btm_stride2[ch64], 9311,  "bottom")},  // 7699   GT:9311
            {tc(wls_CLU_btm_stride2[ch32], 4600,  "bottom")},  // 4202   GT:4600  v17:3103
            {tc(wls_CLU_btm_stride2[ch16], 2371,  "bottom")},  // 2271   GT:2371  v17:1729
                                                                                                                                          
           //CLU + no broadcast                                                                                                                         
            {tc(wls_CLU_stride2_no_broadcast[ch64],  no_gt, "CLU + no broadcast")},  // ERROR_INPUT_TOO_BIG    
            {tc(wls_CLU_stride2_no_broadcast[ch32], 14865,  "CLU + no broadcast")},  // 19762   GT:14865  v17:10697
            {tc(wls_CLU_stride2_no_broadcast[ch16],  7733,  "CLU + no broadcast")},  // 8982   GT:7733    v17:5986

         //SOK
            //{tc(wls_SOK_stride2[ch64], no_gt,  "SOK, all tensors are the same")},  // ERROR_INPUT_TOO_BIG  because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            //{tc(wls_SOK_stride2[ch32], 15865,  "SOK, all tensors are the same")}, // 9724   GT:15865 because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK_stride2[ch16], 7847,  "SOK, all tensors are the same")},  // 8991   GT:7847

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_5x5) {
    const int p{2};
    const int k{5};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 16, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 18, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 16, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO     
             {tc(wls_CLU_top[ch64], 12106, "top")}, // 10463   GT:12106
             {tc(wls_CLU_top[ch32], 6049,  "top")},     // 5708   GT:6049
             {tc(wls_CLU_top[ch16], 6041,  "top")},     // 5726   GT:6041
                                                             
             {tc(wls_CLU_mid[ch64], 12131, "middle")},  // 10490   GT:12131
             {tc(wls_CLU_mid[ch32], 6052 , "middle")},      // 5735   GT:6052
             {tc(wls_CLU_mid[ch16], 6046 , "middle")},      // 5766   GT:6046
                                                            
             {tc(wls_CLU_btm[ch64], 12128, "bottom")},  // 10416   GT:12128
             {tc(wls_CLU_btm[ch32], 6005 , "bottom")},      // 5654   GT:6005
             {tc(wls_CLU_btm[ch16], 6041 , "bottom")},      // 5669   GT:6041
                                                            
            //CLU + no broadcast                                    
             {tc(wls_CLU_no_broadcast[ch64], 42069, "CLU + no broadcast")}, // 54278  GT:42069
             {tc(wls_CLU_no_broadcast[ch32], 20752, "CLU + no broadcast")}, // 22468  GT:20752
             {tc(wls_CLU_no_broadcast[ch16], 20882, "CLU + no broadcast")}, // 22937  GT:20882

            //SOK
            {tc(wls_SOK[ch64], 42147,  "SOK, all tensors are the same")},  // 123718   GT:42147
            {tc(wls_SOK[ch32], 20889,  "SOK, all tensors are the same")},  // 50729    GT:20889
            {tc(wls_SOK[ch16], 21176,  "SOK, all tensors are the same")},  // 23062    GT:21176

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, FLOAT16_kernel_5x5) {
    const int p{2};
    const int k{5};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 16, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 18, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 16, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{
            mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail

    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top[ch64], 24459, "top")},     // 22784   GT:24459
             {tc(wls_CLU_top[ch32], 11959, "top")}, // 10517   GT:11959
             {tc(wls_CLU_top[ch16], 6129 , "top")},     // 5657   GT:6129

             {tc(wls_CLU_mid[ch64], 24458, "middle")},      // 22844   GT:24458
             {tc(wls_CLU_mid[ch32], 12011, "middle")},  // 10521   GT:12011
             {tc(wls_CLU_mid[ch16], 6133 , "middle")},      // 5702   GT:6133

             {tc(wls_CLU_btm[ch64], 24457, "bottom")},      // 22685   GT:24457
             {tc(wls_CLU_btm[ch32], 11959, "bottom")},  // 10363   GT:11959
             {tc(wls_CLU_btm[ch16], 6094 , "bottom")},      // 5594   GT:6094

            //CLU + no broadcast
             {tc(wls_CLU_no_broadcast[ch64], 83926, "CLU + no broadcast")},     // 121388   GT:83926  v17:94226 -->delta 12.27
             {tc(wls_CLU_no_broadcast[ch32], 40906, "CLU + no broadcast")},     // 50295  GT:40906
             {tc(wls_CLU_no_broadcast[ch16], 20816, "CLU + no broadcast")}, // 23058   GT:20816

           //SOK 
            //{tc(wls_SOK[ch64], 84926,  "SOK, all tensors are the same",pc, 13)},  // 123718   GT:84926 (simulated!) because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK[ch32], 40904,  "SOK, all tensors are the same")},  // 50729    GT:40904
            {tc(wls_SOK[ch16], 20868,  "SOK, all tensors are the same")},  // 23062    GT:20868

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_6x6) {
    const int p{2};
    const int k{6};
    const DataType dt{DataType::UINT8};
    const DPUWorkload wl_CLU_top{mk_SOHO(57, 17, 56, 14, p, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(57, 19, 56, 14, 0, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(57, 17, 56, 14, 0, p, p, p, dt, dt, k, k)};  // s

    const DPUWorkload wl_CLU_no_broadcast{mk_SOHO(57, 57, 56, 56, p, p, p, p, dt, dt, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(57, 57, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top[ch64],  17296, "top")},      // 16407  GT:17296
             {tc(wls_CLU_top[ch32],  8613 , "top")},  // 7617   GT:8613
             {tc(wls_CLU_top[ch16],  8599 , "top")},      // 7902   GT:8599

             {tc(wls_CLU_mid[ch64], 17301, "middle")},      // 16482  GT:17301
             {tc(wls_CLU_mid[ch32], 8577 , "middle")},  // 7616   GT:8577
             {tc(wls_CLU_mid[ch16], 8618 , "middle")},      // 7862   GT:8618

             {tc(wls_CLU_btm[ch64], 17321, "bottom")},  // 14314  GT:17321  v17:13328 
             {tc(wls_CLU_btm[ch32], 8611 , "bottom")},  // 6238   GT:8611   v17:6737
             {tc(wls_CLU_btm[ch16], 8607 , "bottom")},  // 6612   GT:8607   v17:6543

            //CLU + no broadcast
             {tc(wls_CLU_no_broadcast[ch64], 60412, "CLU + no broadcast")},  // 90987   GT:60412
             {tc(wls_CLU_no_broadcast[ch32], 29854, "CLU + no broadcast")},  // 35800   GT:29854
             {tc(wls_CLU_no_broadcast[ch16], 29918, "CLU + no broadcast")},  // 35910   GT:29918

            //SOK
            {tc(wls_SOK[ch64], 60477,  "SOK, all tensors are the same")},  // 87222   GT:60477
            {tc(wls_SOK[ch32], 29899,  "SOK, all tensors are the same")},  // 35171    GT:29899
            {tc(wls_SOK[ch16], 30211,  "SOK, all tensors are the same")},  // 35172   GT:30211

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, FLOAT16_kernel_6x6) {
    const int p{2};
    const int k{6};
    const DataType dt{DataType::FLOAT16};
    const DPUWorkload wl_CLU_top{mk_SOHO(57, 17, 56, 14, p, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(57, 19, 56, 14, 0, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(57, 17, 56, 14, 0, p, p, p, dt, dt, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{mk_SOHO(57, 57, 56, 56, p, p, p, p, dt, dt, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(57, 57, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wlS_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    //  no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top[ch64], 34840, "top")},  // 35078  GT:34840
             {tc(wls_CLU_top[ch32], 17035, "top")},  // 15905  GT:17035
             {tc(wls_CLU_top[ch16], 8694 , "top")},  // 8079   GT:8694

             {tc(wls_CLU_mid[ch64], 34841, "middle")},  // 35223  GT:34841
             {tc(wls_CLU_mid[ch32], 17073, "middle")},  // 15906  GT:17073
             {tc(wls_CLU_mid[ch16], 8698 , "middle")},  // 8008  GT:8698

             {tc(wls_CLU_btm[ch64], 34840, "bottom")},  // 28362  GT:34840  v17:27023
             {tc(wls_CLU_btm[ch32], 17027, "bottom")},  // 11728  GT:17027  v17:13329
             {tc(wls_CLU_btm[ch16], 8694 , "bottom")},  // 6579   GT:8694   v17:6801

            //CLU + no broadcast
             {tc(wlS_CLU_no_broadcast[ch64], 120547, "CLU + no broadcast")},  // 171159   GT:120547  v17:133903 delta--->11.08
             {tc(wlS_CLU_no_broadcast[ch32], 58724,  "CLU + no broadcast")},  // 73100   GT:58724
             {tc(wlS_CLU_no_broadcast[ch16], 29904,  "CLU + no broadcast")},  // 35936   GT:29904

             //SOK
            //{tc(wls_SOK[ch64], 121547,  "SOK, all tensors are the same")}, // 165053   GT:121547 because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK[ch32], 58698,  "SOK, all tensors are the same")},  // 71881    GT:58698
            {tc(wls_SOK[ch16], 29888,  "SOK, all tensors are the same")},  // 34900    GT:29888

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_7x7) {
    const int p{3};
    const int k{7};
    // STRIDE=1
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 17, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 20, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 17, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO
             {tc(wls_CLU_top[ch64], 23434,  "top")},  // 22468  GT:23434
             {tc(wls_CLU_top[ch32], 11630,  "top")},  // 9734   GT:11630
             {tc(wls_CLU_top[ch16], 11667,  "top")},  // 10539   GT:11667
                                                       
             {tc(wls_CLU_mid[ch64], 23462,  "middle")},  // 22529  GT:23462
             {tc(wls_CLU_mid[ch32], 11645,  "middle")},  // 9757   GT:11645
             {tc(wls_CLU_mid[ch16], 11682,  "middle")},  // 10519   GT:11682
                                                     
             {tc(wls_CLU_btm[ch64], 23452,  "bottom")},  // 22335  GT:23452
             {tc(wls_CLU_btm[ch32], 11627,  "bottom")},  // 9650   GT:11627
             {tc(wls_CLU_btm[ch16], 11668,  "bottom")},  // 10353   GT:11668
                                                            
            //CLU + no broadcast                                       
             {tc(wls_CLU_no_broadcast[ch64], 82014,  "CLU + no broadcast")},   // 118787   GT:82014
             {tc(wls_CLU_no_broadcast[ch32], 40625,  "CLU + no broadcast")},   // 43496   GT:40625
             {tc(wls_CLU_no_broadcast[ch16], 40640,  "CLU + no broadcast")},   // 44269   GT:40640

            //SOK
            {tc(wls_SOK[ch64], 82071,  "SOK, all tensors are the same")},  // 119365   GT:82071
            {tc(wls_SOK[ch32], 40668,  "SOK, all tensors are the same")},  // 43998    GT:40668
            {tc(wls_SOK[ch16], 40927,  "SOK, all tensors are the same")},  // 44717    GT:40927

            // clang-format on
    };

    executeTests(tests);

    //********************************************************************************************************
    // STRIDE=2
    const DPUWorkload wl_CLU_top_stride2{
            mk_SOHO(111, 30, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    const DPUWorkload wl_CLU_mid_stride2{
            mk_SOHO(111, 33, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    const DPUWorkload wl_CLU_btm_stride2{
            mk_SOHO(111, 30, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};

    const DPUWorkload wl_CLU_stride2_no_broadcast{
            mk_SOHO(111, 111, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    const DPUWorkload wl_SOK_stride2{mk_SOK(112, 112, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_stride2_no_broadcast{change_wl_channels_64_32_16(wl_CLU_stride2_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};
    wls_SOK_stride2[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top_stride2[ch64], 23455, "top")},  // 22463  GT:23455
             {tc(wls_CLU_top_stride2[ch32], 11658, "top")},  // 9729   GT:11658  v17:12866--->delta 10.36%
             {tc(wls_CLU_top_stride2[ch16], 11653, "top")},  // 10583   GT:11653 v17:13504--->delta 16.45%
                                                                                                         
             {tc(wls_CLU_mid_stride2[ch64], 23437, "middle")},  // 22523  GT:23437
             {tc(wls_CLU_mid_stride2[ch32], 11664, "middle")},  // 9770   GT:11664  v17:13102 --->delta 12.33
             {tc(wls_CLU_mid_stride2[ch16], 11659, "middle")},  // 10562   GT:11659 v17:13707 --->delta 17.57
                                                                     
             {tc(wls_CLU_btm_stride2[ch64], 23456, "bottom")},  // 22328  GT:23456
             {tc(wls_CLU_btm_stride2[ch32], 11662, "bottom")},  // 9662   GT:11662 v17:12829 --->delta:10.01
             {tc(wls_CLU_btm_stride2[ch16], 11660, "bottom")},  // 10398  GT:11660 v17:13477 --->delta:15.58
                                                                               
            //CLU + no broadcast                                                             
             {tc(wls_CLU_stride2_no_broadcast[ch64], 82012, "CLU + no broadcast")},   // 116428   GT:82012
             {tc(wls_CLU_stride2_no_broadcast[ch32], 41089, "CLU + no broadcast")},   // 43374   GT:41089  v17:47633 --->delta:15.93%
             {tc(wls_CLU_stride2_no_broadcast[ch16], 40549, "CLU + no broadcast")},   // 44304   GT:40549  v17:48309 --->delta:19.14%

             //SOK
            //{tc(wls_SOK_stride2[ch64], 83012,  "SOK, all tensors are the same")},  // 119365   GT:83012 because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK_stride2[ch32], 41160,  "SOK, all tensors are the same")},  // 43998    GT:41160
            {tc(wls_SOK_stride2[ch16], 41698,  "SOK, all tensors are the same")},  // 44717    GT:41698

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, FLOAT16_kernel_7x7) {
    const int p{3};
    const int k{7};
    // STRIDE=1
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 17, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 20, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 17, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{
            mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top[ch64], 47085, "top")},  // 44521  GT:47085
             {tc(wls_CLU_top[ch32], 23060, "top")},  // 20558   GT:23060
             {tc(wls_CLU_top[ch16], 11722, "top")},  // 10680   GT:11722
                                                      
             {tc(wls_CLU_mid[ch64], 47116, "middle")},  // 44585  GT:47116
             {tc(wls_CLU_mid[ch32], 23065, "middle")},  // 20512   GT:23065
             {tc(wls_CLU_mid[ch16],11768 , "middle")},  // 10594   GT:11768
                                                      
             {tc(wls_CLU_btm[ch64], 47109, "bottom")},  // 43937  GT:47109
             {tc(wls_CLU_btm[ch32], 23052, "bottom")},  // 20168   GT:23052
             {tc(wls_CLU_btm[ch16], 11756, "bottom")},  // 10411   GT:11756
                                                           
            //CLU + no broadcast                                           
             {tc(wls_CLU_no_broadcast[ch64], 163846, "CLU + no broadcast ")},   // 203915   GT:163846
             {tc(wls_CLU_no_broadcast[ch32], 79891 , "CLU + no broadcast ")},   // 88575   GT:79891
             {tc(wls_CLU_no_broadcast[ch16], 40699 , "CLU + no broadcast ")},   // 44065   GT:40699

            //SOK
            //{tc(wls_SOK[ch64], 164846,  "SOK, all tensors are the same")},  // 206750   GT:164846 because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK[ch32], 79914,  "SOK, all tensors are the same")},  // 89222    GT:79914
            {tc(wls_SOK[ch16], 40674,  "SOK, all tensors are the same")},  // 44095    GT:40674

            // clang-format on
    };

    executeTests(tests);

    //********************************************************************************************************
    // STRIDE=2
    const DPUWorkload wl_CLU_top_stride2{
            mk_SOHO(111, 30, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    const DPUWorkload wl_CLU_mid_stride2{
            mk_SOHO(111, 33, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    const DPUWorkload wl_CLU_btm_stride2{
            mk_SOHO(111, 30, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};

    const DPUWorkload wl_CLU_stride2_no_broadcast{
            mk_SOHO(111, 111, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    const DPUWorkload wl_SOK_stride2{mk_SOK(56 * 2, 56 * 2, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_stride2_no_broadcast{change_wl_channels_64_32_16(wl_CLU_stride2_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};
    wls_SOK_stride2[ch16].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top_stride2[ch64],47125 ,  "top")},  // 44477  GT:47125
             {tc(wls_CLU_top_stride2[ch32], 23048,  "top")},  // 20549   GT:23048
             {tc(wls_CLU_top_stride2[ch16],11709 ,  "top")},  // 10717   GT:11709 v17:13449 --->delta:14.86
                                                                                                       
             {tc(wls_CLU_mid_stride2[ch64], 47079 , "middle")},  // 44443  GT:47079
             {tc(wls_CLU_mid_stride2[ch32],  23101, "middle")},  // 20508   GT:23101
             {tc(wls_CLU_mid_stride2[ch16], 11746 , "middle")},  // 10630   GT:11746 v17:13607 --->delta:15.84
                                                                     
             {tc(wls_CLU_btm_stride2[ch64],47119 ,  "bottom")},  // 43874  GT:47119
             {tc(wls_CLU_btm_stride2[ch32], 23099,  "bottom")},  // 20150   GT:23099
             {tc(wls_CLU_btm_stride2[ch16], 11747,  "bottom")},  // 10449   GT:11747  v17:13427 --->delta:14.30%
                                                                            
            //CLU + no broadcast                                                  
             {tc(wls_CLU_stride2_no_broadcast[ch64],  no_gt , "CLU + no broadcast ")},   //  ERROR_INPUT_TOO_BIG
             {tc(wls_CLU_stride2_no_broadcast[ch32],  79843 , "CLU + no broadcast ")},   // 88402   GT:79843
             {tc(wls_CLU_stride2_no_broadcast[ch16],  41217 , "CLU + no broadcast ")},   // 44067   GT:41217  v17:48282 --->delta:17.14%

             //SOK
            //{tc(wls_SOK_stride2[ch64], no_gt,  "SOK, all tensors are the same")},  // 206750  ERROR_INPUT_TOO_BIG ??????  because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            //{tc(wls_SOK_stride2[ch32], 80834,  "SOK, all tensors are the same")},  // 89222     GT:80834  because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK_stride2[ch16], 41057,  "SOK, all tensors are the same")},  // 44095     GT:41057

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_9x9) {
    const int p{4};
    const int k{9};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 18, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 22, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 18, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO
             {tc(wls_CLU_top[ch64], 38530, "top")},  // 37218  GT:38530
             {tc(wls_CLU_top[ch32], 19174, "top")},  // 177757   GT:19174
             {tc(wls_CLU_top[ch16], 19169, "top")},  // 18800   GT:19169
                                                      
             {tc(wls_CLU_mid[ch64], 38563, "middle")},  // 37011  GT:38563
             {tc(wls_CLU_mid[ch32], 19193, "middle")},  // 17409   GT:19193
             {tc(wls_CLU_mid[ch16], 19193, "middle")},  // 18380   GT:19193
                                                     
             {tc(wls_CLU_btm[ch64], 38532, "bottom")},  // 35836  GT:38532
             {tc(wls_CLU_btm[ch32], 19170, "bottom", 13)},  // 16800   GT:19170
             {tc(wls_CLU_btm[ch16], 19169, "bottom")},  // 17723   GT:19169

            //CLU + no broadcast
             {tc(wls_CLU_no_broadcast[ch64], 135239, "CLU + no broadcast")},  // 179718  GT:135239
             {tc(wls_CLU_no_broadcast[ch32], 66941,  "CLU + no broadcast")},   // 64105   GT:66941
             {tc(wls_CLU_no_broadcast[ch16], 67004 , "CLU + no broadcast")},   // 68409   GT:67004

            //SOK
            {tc(wls_SOK[ch64], 135328, "SOK, all tensors are the same")},  // 180794   GT:135328
            {tc(wls_SOK[ch32], 67029,  "SOK, all tensors are the same")},  // 63898    GT:67029
            {tc(wls_SOK[ch16], 67242,  "SOK, all tensors are the same")},  // 68892    GT:67242

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, FLOAT16_kernel_9x9) {
    const int p{4};
    const int k{9};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 18, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 22, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 18, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{
            mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top[ch64], 77286, "top")},  // 71826  GT:77286
             {tc(wls_CLU_top[ch32], 37903, "top")},  // 34421   GT:37903
             {tc(wls_CLU_top[ch16], 19258, "top")},  // 18547   GT:19258
                                                    
             {tc(wls_CLU_mid[ch64], 77324, "middle")},  // 72418  GT:77324
             {tc(wls_CLU_mid[ch32], 37878, "middle")},  // 33494   GT:37878
             {tc(wls_CLU_mid[ch16], 19282, "middle")},  // 17936   GT:19282
                                                      
             {tc(wls_CLU_btm[ch64], 77319, "bottom")},  // 71128  GT:77319
             {tc(wls_CLU_btm[ch32], 37849, "bottom")},  // 31918   GT:37849
             {tc(wls_CLU_btm[ch16], 19221, "bottom")},  // 17456   GT:19221

            //CLU + no broadcast
             {tc(wls_CLU_no_broadcast[ch64], 270310, "CLU + no broadcast")},  // 263261  GT:270310
             {tc(wls_CLU_no_broadcast[ch32], 131901, "CLU + no broadcast")},   // 124870   GT:131901
             {tc(wls_CLU_no_broadcast[ch16], 67009 , "CLU + no broadcast")},   // 66494   GT:67009

            //SOK
            //{tc(wls_SOK[ch64], 271310,  "SOK, all tensors are the same")},  // 266443   GT:271310 because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK[ch32], 131898, "SOK, all tensors are the same")},   // 125053   GT:131898
            {tc(wls_SOK[ch16], 67009,  "SOK, all tensors are the same")},   // 66626    GT:67009

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, UINT8_kernel_11x11) {
    const int p{5};
    const int k{11};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 19, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 24, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 19, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch64].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    //  no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top[ch64], 57443, "top")},  // 55052   GT:57443
             {tc(wls_CLU_top[ch32], 28552, "top")},  // 28204   GT:28552
             {tc(wls_CLU_top[ch16], 28547, "top")},  // 29199   GT:28547
                                                       
             {tc(wls_CLU_mid[ch64], 57417, "middle")},  // 54143   GT:57417
             {tc(wls_CLU_mid[ch32], 28590, "middle")},  // 27629   GT:28590
             {tc(wls_CLU_mid[ch16], 28587, "middle")},  // 28074   GT:28587
                                                      
             {tc(wls_CLU_btm[ch64], 57434, "bottom")},  // 52628  GT:57434
             {tc(wls_CLU_btm[ch32], 28504, "bottom")},  // 26211   GT:28504
             {tc(wls_CLU_btm[ch16], 28547, "bottom")},  // 27249   GT:28547

            //CLU + no broadcast
             {tc(wls_CLU_no_broadcast[ch64], 201811, "CLU + no broadcast")},  // 237816  GT:201811
             {tc(wls_CLU_no_broadcast[ch32], 100011, "CLU + no broadcast")}, // 92387   GT:100011
             {tc(wls_CLU_no_broadcast[ch16], 100005, "CLU + no broadcast")}, // 97696   GT:100005

            //SOK
            {tc(wls_SOK[ch64], 201902, "SOK, all tensors are the same")},  // 239444   GT:201902
            {tc(wls_SOK[ch32], 100009, "SOK, all tensors are the same")},  // 91744    GT:100009
            {tc(wls_SOK[ch16], 100234, "SOK, all tensors are the same")},  // 98322    GT:100234

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU40, FLOAT16_kernel_11x11) {
    const int p{5};
    const int k{11};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 19, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 24, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 19, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_CLU_no_broadcast{
            mk_SOHO(56, 56, 56, 56, p, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_CLU_no_broadcast{change_wl_channels_64_32_16(wl_CLU_no_broadcast)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};
    wls_SOK[ch32].output_write_tiles =
            2;  // Initially, OWT=4, but SOK broadcasts to all 4 tiles at the end, which resulted in us getting a CMX
                // memory that was too large. We did a hack to have a smaller OWT, in this case owt=2, so that we would
                // only broadcast to 2 tiles and the output memory would be smaller

    // no_fail = 0; //tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO
             {tc(wls_CLU_top[ch64], 115074, "top")},  // 111030  GT:115074
             {tc(wls_CLU_top[ch32], 56401 , "top")},  // 49338   GT:56401
             {tc(wls_CLU_top[ch16], 28636 , "top")},  // 29200   GT:28636
                                                       
             {tc(wls_CLU_mid[ch64], 115036, "middle")},  // 112274  GT:115036
             {tc(wls_CLU_mid[ch32], 56441 , "middle")},  // 47447   GT:56441
             {tc(wls_CLU_mid[ch16], 28676 , "middle")},  // 27752   GT:28676
                                                      
             {tc(wls_CLU_btm[ch64], 115036, "bottom")},  // 110743  GT:115036
             {tc(wls_CLU_btm[ch32], 56401 , "bottom")},  // 44930   GT:56401
             {tc(wls_CLU_btm[ch16], 28599 , "bottom")},  // 26829   GT:28599

            //CLU + no broadcast
             {tc(wls_CLU_no_broadcast[ch64], 403414, "CLU + no broadcast")},  // 322721  GT:403414
             {tc(wls_CLU_no_broadcast[ch32], 196894, "CLU + no broadcast")}, // 162516   GT:196894
             {tc(wls_CLU_no_broadcast[ch16], 100006, "CLU + no broadcast")}, // 96860   GT:100006

            //SOK
            //{tc(wls_SOK[ch64], 404414,  "SOK, all tensors are the same")},  // 325934   GT:404414  because SOK broadcasts to all tiles, even if we tried to split across 4 tiles or a smaller number of tiles, the output memory caused the CMX memory to be too big => these tests cannot be profiled
            {tc(wls_SOK[ch32], 196893, "SOK, all tensors are the same")},  // 162653    GT:196893
            {tc(wls_SOK[ch16], 100005,  "SOK, all tensors are the same")}, // 97522    GT:100005
            // clang-format on
    };

    executeTests(tests);
}

class Regression_tests_MAXPOOL_EISXW_99246_NPU40 : public Regression_Tests {
public:
protected:
    // base wl, before split
    // const VPUNN::DPUWorkload base_wl{
    //        VPUNN::VPUDevice::VPU_4_0,
    //        VPUNN::Operation::MAXPOOL,
    //        {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
    //        {VPUNN::VPUTensor(28, 28, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
    //        {3, 3},                                                     // kernels
    //        {2, 2},                                                     // strides
    //        {1, 0, 1, 0},                                               // padding
    //        VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
    //        VPUNN::ActivationFunction::NONE,                            // activation
    //        0.0F,                                                       // act_sparsity
    //        0.0F,                                                       // weight_sparsity
    //        {swz_def, swz_def},                                         // input_swizzling
    //        {swz_def},                                                  // output_swizzling
    //        1,                                                          // output_write_tiles
    //        {0, 0, 0, 0},                                               // offsets
    //        VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
    //        false,                                                      // weight_sparsity_enabled
    //};

    // base wl, before split, EISXW_99246
    //  const VPUNN::DPUWorkload wl_MXP_layer{
    //         VPUNN::VPUDevice::VPU_4_0,
    //         VPUNN::Operation::MAXPOOL,
    //         {VPUNN::VPUTensor(112, 112, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
    //         {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},    // output dimensions
    //         {3, 3},                                                       // kernels
    //         {2, 2},                                                       // strides
    //         {1, 0, 1, 0},                                                 // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //         VPUNN::ActivationFunction::NONE,                              // activation
    //         0.0F,                                                         // act_sparsity
    //         0.0F,                                                         // weight_sparsity
    //         {swz_def, swz_def},                                           // input_swizzling
    //         {swz_def},                                                    // output_swizzling
    //         1,                                                            // output_write_tiles
    //         {0, 0, 0, 0},                                                 // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
    //         false,                                                        // weight_sparsity_enabled
    // };

    /// with this function, starting from a SOHO base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param in_w: the input tensor width
    /// @param in_h: the input tensor height
    /// @param out_w: the output tensor width
    /// @param out_h: output tensor height
    /// @param top_padd: top padding for workload
    /// @param btm_padd: bottom padding for workload
    /// @param Tint: input tensor Dtype
    /// @param Tout: output tensor Dtype
    /// @param kw: kernel width
    /// @param kh: kernel height
    /// @param stride: wl stride
    /// @return a new DPUWorkload
    static DPUWorkload mk_SOHO(unsigned int in_w, unsigned int in_h, unsigned int out_w, unsigned int out_h,
                               unsigned int top_padd, unsigned int btm_padd, DataType Tin, DataType Tout,
                               unsigned int kw, unsigned int kh, unsigned int stride = 1) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::MAXPOOL,
                {VPUTensor(in_w, in_h, 64, 1, Tin, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, 64, 1, Tout, Layout::ZXY)},  // output dimensions
                {kw, kh},                                             // kernels
                {stride, stride},                                     // strides
                {top_padd, btm_padd, 1, 0},                           // padding
                ExecutionMode::CUBOID_16x16,                          // execution mode
                ActivationFunction::NONE,                             // activation
                0.0F,                                                 // act_sparsity
                0.0F,                                                 // weight_sparsity
                {swz_def, swz_def},                                   // input_swizzling
                {swz_def},                                            // output_swizzling
                1,                                                    // output_write_tiles
                {0, 0, 0, 0},                                         // offsets
                ISIStrategy::CLUSTERING,                              // isi_strategy
                false,                                                //
        };
    }

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_tests_MAXPOOL_EISXW_99246_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " in_H:" + std::to_string(wl.inputs[0].get_shape()[1]) +
                              " out_H:" + std::to_string(wl.outputs[0].get_shape()[1]) + 
                              " wt_spars_enabled:" +(wl.weight_sparsity_enabled ? "true" : "false") +
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }
};

TEST_F(Regression_tests_MAXPOOL_EISXW_99246_NPU40, L112x112_UINT8_k_3x3_stride2) {
    int k{3};  // kernel
    int s{2};  // stride

    // execution mode 16x16
    DPUWorkload SOHO_16x16_top{mk_SOHO(112, 28, 56, 14, 1, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    DPUWorkload SOHO_16x16_mid_btm{mk_SOHO(112, 29, 56, 14, 0, 0, DataType::UINT8, DataType::UINT8, k, k, s)};

    // execution mode 8x16
    const DPUWorkload SOHO_8x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_mid_btm{mod_execution(SOHO_16x16_mid_btm, ExecutionMode::CUBOID_8x16)};

    // execution mode 4x16
    const DPUWorkload SOHO_4x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload SOHO_4x16_mid_btm{mod_execution(SOHO_16x16_mid_btm, ExecutionMode::CUBOID_4x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
        //VERY BIG ERRORS
              // CUBOID_16x16
              {tc(SOHO_16x16_top    , 4508, "SOHO, top"   )},       //v17: 6023   GT:4508  3311
              {tc(SOHO_16x16_mid_btm, 4533, "SOHO, middle + btm")}, //v17: 6132   GT:4533  3309

              //other MPE modes do not get selected by design
               // CUBOID_8x16
              {tc(SOHO_8x16_top    , 6313, "SOHO, top"   )},       //v17: 11019   GT:6313  4833
              {tc(SOHO_8x16_mid_btm, 6378, "SOHO, middle + btm")}, //v17: 11221   GT:6378  4827

               // CUBOID_4x16
              {tc(SOHO_4x16_top    , 6421, "SOHO, top"   )},       //v17: 10874   GT:6421  5067
              {tc(SOHO_4x16_mid_btm, 6421, "SOHO, middle + btm")}, //v17: 11077   GT:6396  5025

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_MAXPOOL_EISXW_99246_NPU40, L112x112_UINT8_k_3x3_stride2_Broadcast) {
    int k{3};  // kernel
    int s{2};  // stride

    // execution mode 16x16
    DPUWorkload SOHO_16x16_top{mk_SOHO(112, 28, 56, 14, 1, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    DPUWorkload SOHO_16x16_mid_btm{mk_SOHO(112, 29, 56, 14, 0, 0, DataType::UINT8, DataType::UINT8, k, k, s)};

    SOHO_16x16_top.output_write_tiles = 4;
    SOHO_16x16_mid_btm.output_write_tiles = 4;

    // execution mode 8x16
    const DPUWorkload SOHO_8x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_mid_btm{mod_execution(SOHO_16x16_mid_btm, ExecutionMode::CUBOID_8x16)};

    // execution mode 4x16
    const DPUWorkload SOHO_4x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload SOHO_4x16_mid_btm{mod_execution(SOHO_16x16_mid_btm, ExecutionMode::CUBOID_4x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
        //VERY BIG ERRORS
              // CUBOID_16x16
              {tc(SOHO_16x16_top    , 4665, "SOHO B, top"   )},       //v17:
              {tc(SOHO_16x16_mid_btm, 4669, "SOHO B, middle + btm")}, //v17:

              //other MPE modes do not get selected by design
               // CUBOID_8x16
              {tc(SOHO_8x16_top    , 6520, "SOHO B, top"   )},       //v17: 
              {tc(SOHO_8x16_mid_btm, 6523, "SOHO B, middle + btm")}, //v17: 

               // CUBOID_4x16
              {tc(SOHO_4x16_top    , 6450, "SOHO B, top"   )},       //v17: 
              {tc(SOHO_4x16_mid_btm, 6475, "SOHO B, middle + btm")}, //v17:

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_MAXPOOL_EISXW_99246_NPU40, L54x54_UINT8_k_3x3_s_1x1) {
    int k{3};  // kernel
    int s{1};  // stride

    // execution mode 16x16
    const DPUWorkload SOHO_16x16_top{mk_SOHO(54, 15, 53, 14, 1, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    const DPUWorkload SOHO_16x16_mid{mk_SOHO(54, 16, 53, 14, 0, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    const DPUWorkload SOHO_16x16_btm{mk_SOHO(54, 13, 53, 11, 0, 0, DataType::UINT8, DataType::UINT8, k, k, s)};

    // execution mode 8x16
    const DPUWorkload SOHO_8x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_mid{mod_execution(SOHO_16x16_mid, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_btm{mod_execution(SOHO_16x16_btm, ExecutionMode::CUBOID_8x16)};

    // execution mode 4x16
    const DPUWorkload SOHO_4x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload SOHO_4x16_mid{mod_execution(SOHO_16x16_mid, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload SOHO_4x16_btm{mod_execution(SOHO_16x16_btm, ExecutionMode::CUBOID_4x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off

             // CUBOID_16x16
             {tc(SOHO_16x16_top, 4501, "SOHO, top"    )},  // 3164  v17:5663    GT:
             {tc(SOHO_16x16_mid, 4502, "SOHO, middle ")},  // 3168  v17:5769    GT:
             {tc(SOHO_16x16_btm, 3509, "SOHO, btm "   )},  // 2863  v17:4414    GT:

              // CUBOID_8x16
             {tc(SOHO_8x16_top, 6202, "SOHO, top "   )},   // 4643  v17:10609    GT:
             {tc(SOHO_8x16_mid, 6199, "SOHO, middle ")},   // 4638  v17:10797    GT:
             {tc(SOHO_8x16_btm, 4855, "SOHO, btm "   )},   // 4234  v17:8089     GT:

              // CUBOID_4x16
             {tc(SOHO_4x16_top, 6171, "SOHO, top "   )},   // 4814  v17:10462    GT:
             {tc(SOHO_4x16_mid, 6204, "SOHO, middle ")},   // 4770  v17:10662    GT:
             {tc(SOHO_4x16_btm, 5272, "SOHO, btm "   )},   // 4116  v17:8355     GT:

            // clang-format on
    };

    executeTests(tests);
}
TEST_F(Regression_tests_MAXPOOL_EISXW_99246_NPU40, L54x54_UINT8_k_3x3_s_1x1_Broadcast) {
    int k{3};  // kernel
    int s{1};  // stride

    // execution mode 16x16
    DPUWorkload SOHO_16x16_top{mk_SOHO(54, 15, 53, 14, 1, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    DPUWorkload SOHO_16x16_mid{mk_SOHO(54, 16, 53, 14, 0, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    DPUWorkload SOHO_16x16_btm{mk_SOHO(54, 13, 53, 11, 0, 0, DataType::UINT8, DataType::UINT8, k, k, s)};
    SOHO_16x16_top.output_write_tiles = 4;
    SOHO_16x16_mid.output_write_tiles = 4;
    SOHO_16x16_btm.output_write_tiles = 4;

    // execution mode 8x16
    const DPUWorkload SOHO_8x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_mid{mod_execution(SOHO_16x16_mid, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_btm{mod_execution(SOHO_16x16_btm, ExecutionMode::CUBOID_8x16)};

    // execution mode 4x16
    const DPUWorkload SOHO_4x16_top{mod_execution(SOHO_16x16_top, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload SOHO_4x16_mid{mod_execution(SOHO_16x16_mid, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload SOHO_4x16_btm{mod_execution(SOHO_16x16_btm, ExecutionMode::CUBOID_4x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off

             // CUBOID_16x16
             {tc(SOHO_16x16_top, 4593, "SOHO,brdcst, top"    )},  //   v17:   GT:
             {tc(SOHO_16x16_mid, 4595, "SOHO,brdcst, middle ")},  //   v17:   GT:
             {tc(SOHO_16x16_btm, 3514, "SOHO,brdcst, btm "   )},  //   v17:   GT:

              // CUBOID_8x16
             {tc(SOHO_8x16_top, 6210, "SOHO,brdcst, top "   )},   //   v17:    GT:
             {tc(SOHO_8x16_mid, 6237, "SOHO,brdcst, middle ")},   //   v17:    GT:
             {tc(SOHO_8x16_btm, 4829, "SOHO,brdcst, btm "   )},   //   v17:    GT:

              // CUBOID_4x16
             {tc(SOHO_4x16_top, 6270, "SOHO,brdcst, top "   )},   //   v17:    GT:
             {tc(SOHO_4x16_mid, 6283, "SOHO,brdcst, middle ")},   //   v17:    GT:
             {tc(SOHO_4x16_btm, 5370, "SOHO,brdcst, btm "   )},   //   v17:    GT:

            // clang-format on
    };

    executeTests(tests);
}

class Regression_tests_CONV_EISXW_127649_Model_E_NPU40 : public Regression_Tests {
    // conv4 base wl, before split
    // const DPUWorkload conv4{
    //        VPUDevice::VPU_4_0,
    //        Operation::CONVOLUTION,
    //        {VPUTensor(56, 32, 64, 1, DataType::UINT8)},  // input dimensions
    //        {VPUTensor(56, 32, 64, 1, DataType::UINT8)},  // output dimensions
    //        {3, 3},                                       // kernels
    //        {1, 1},                                       // strides
    //        {1, 1, 1, 1},                                 // padding
    //        ExecutionMode::CUBOID_16x16,                  // execution mode
    //        ActivationFunction::NONE,                     // activation
    //        0.0F,                                         // act_sparsity
    //        0.0F,                                         // weight_sparsity
    //        {swz_def, swz_def},                           // input_swizzling
    //        {swz_def},                                    // output_swizzling
    //        1,                                            // output_write_tiles
    //        {0, 0, 0, 0},                                 // offsets
    //        ISIStrategy::CLUSTERING,                      // isi_strategy
    //        false,                                        // weight_sparsity_enabled
    //};

    // conv8 base wl, before split
    // const DPUWorkload conv8{
    //         VPUDevice::VPU_4_0,
    //         Operation::CONVOLUTION,
    //         {VPUTensor(28, 16, 96, 1, DataType::UINT8)},  // input dimensions
    //         {VPUTensor(28, 16, 96, 1, DataType::UINT8)},  // output dimensions
    //         {3, 3},                                       // kernels
    //         {1, 1},                                       // strides
    //         {1, 1, 1, 1},                                 // padding
    //         ExecutionMode::CUBOID_16x16,                  // execution mode
    //         ActivationFunction::NONE,                     // activation
    //         0.0F,                                         // act_sparsity
    //         0.0F,                                         // weight_sparsity
    //         {swz_def, swz_def},                           // input_swizzling
    //         {swz_def},                                    // output_swizzling
    //         1,                                            // output_write_tiles
    //         {0, 0, 0, 0},                                 // offsets
    //         ISIStrategy::CLUSTERING,                      // isi_strategy
    //         false,                                        // weight_sparsity_enabled
    // };

protected:
    /// with this function, starting from a SOHO base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param in_w: the input tensor width
    /// @param in_h: the input tensor height
    /// @param out_w: the output tensor width
    /// @param out_h: output tensor height
    /// @param top_padd: top padding for workload
    /// @param btm_padd: bottom padding for workload
    /// @param Tint: input tensor Dtype
    /// @param Tout: output tensor Dtype
    /// @param kw: kernel width
    /// @param kh: kernel height
    /// @param stride: wl stride
    /// @return a new DPUWorkload
    static DPUWorkload mk_SOHO(unsigned int in_w, unsigned int in_h, unsigned int in_c, unsigned int out_w,
                               unsigned int out_h, unsigned int out_c, unsigned int top_padd, unsigned int btm_padd) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                             // kernels
                {1, 1},                                                             // strides
                {top_padd, btm_padd, 1, 1},                                         // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                1,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::CLUSTERING,                                            // isi_strategy
                false,                                                              //
        };
    }
    static DPUWorkload mk_SOK(unsigned int in_w, unsigned int in_h, unsigned int in_c, unsigned int out_w,
                              unsigned int out_h, unsigned int out_c) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                             // kernels
                {1, 1},                                                             // strides
                {1, 1, 1, 1},                                                       // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                6,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::SPLIT_OVER_K,                                          // isi_strategy
                false,                                                              //
        };
    }

    const float sparsity = 0.128038f;
    // activate weight sparsity
    const DPUWorkload act_wt_sp(const DPUWorkload& wl_ref, float wt_spars) {
        DPUWorkload wl{wl_ref};

        wl.weight_sparsity_enabled = true;
        wl.weight_sparsity = wt_spars;

        return wl;
    }

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_tests_CONV_EISXW_127649_Model_E_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " out_H:" + std::to_string(wl.outputs[0].get_shape()[1]) + 
                              " wt_spars_enabled:" +(wl.weight_sparsity_enabled ? "true" : "false") +
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }
};

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 7, 64, 56, 6, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 8, 64, 56, 6, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 3, 64, 56, 2, 64, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv4_8x16_T{mod_execution(SOHO_conv4_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_M{mod_execution(SOHO_conv4_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_B{mod_execution(SOHO_conv4_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv4_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv4_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv4_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 6;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            // 
             //SOH conv4 CUBOID_8x16 
             {tc(SOHO_conv4_8x16_T, 8294, "SOHO, top"   )},    // 8045   GT: 8294
             {tc(SOHO_conv4_8x16_M, 8296, "SOHO, middle")}, // 8140   GT: 8296
             {tc(SOHO_conv4_8x16_B, 4231, "SOHO, bottom")}, // 4142   GT: 4231

              //conv4 CUBOID_8x16 + broadcast
             {tc(CLU_broadcast_8x16_T, 8359, "SOHO, top + broadcast"   )},    // 8083   GT: 8359 
             {tc(CLU_broadcast_8x16_M, 8361, "SOHO, middle + broadcast")}, // 8176   GT: 8361
             {tc(CLU_broadcast_8x16_B, 4248, "SOHO, bottom + broadcast")}, // 4160   GT: 4248

            // clang-format on
    };

    executeTests(tests);
}
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_EXEC_H662_8x16_SPARSITY) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 7, 64, 56, 6, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 8, 64, 56, 6, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 3, 64, 56, 2, 64, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv4_8x16_T{mod_execution(SOHO_conv4_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_M{mod_execution(SOHO_conv4_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_B{mod_execution(SOHO_conv4_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv4_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv4_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv4_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 6;

    // execution mode 8x16 + sparsity
    DPUWorkload SOHO_conv4s_8x16_T{act_wt_sp(SOHO_conv4_8x16_T, sparsity)};
    DPUWorkload SOHO_conv4s_8x16_M{act_wt_sp(SOHO_conv4_8x16_M, sparsity)};
    DPUWorkload SOHO_conv4s_8x16_B{act_wt_sp(SOHO_conv4_8x16_B, sparsity)};

    // execution mode 8x16 + broadcast + sparsity
    DPUWorkload CLU_broadcast_s_8x16_T{act_wt_sp(CLU_broadcast_8x16_T, sparsity)};
    DPUWorkload CLU_broadcast_s_8x16_M{act_wt_sp(CLU_broadcast_8x16_M, sparsity)};
    DPUWorkload CLU_broadcast_s_8x16_B{act_wt_sp(CLU_broadcast_8x16_B, sparsity)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_8x16 + sparsity
             {tc(SOHO_conv4s_8x16_T, 8329, "SOHO, top + wt sparsity"   )},    // 7725   GT: 8329
             {tc(SOHO_conv4s_8x16_M, 8330, "SOHO, middle + wt sparsity")}, // 7756   GT: 8330
             {tc(SOHO_conv4s_8x16_B, 4335, "SOHO, bottom + wt sparsity")}, // 3958   GT: 4335 

             //conv4 CUBOID_8x16 + broadcast + sparsity
             {tc(CLU_broadcast_s_8x16_T, 8396, "SOHO, top + broadcast + wt sparsity"   )},    // 7763   GT: 8396
             {tc(CLU_broadcast_s_8x16_M, 8395, "SOHO, middle + broadcast + wt sparsity")}, // 7792   GT: 8395
             {tc(CLU_broadcast_s_8x16_B, 4322, "SOHO, bottom + broadcast + wt sparsity")}, // 3978   GT: 4322
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_EXEC_16x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 7, 64, 56, 6, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 8, 64, 56, 6, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 3, 64, 56, 2, 64, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv4_T};
    CLU_broadcast_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_M{SOHO_conv4_M};
    CLU_broadcast_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_B{SOHO_conv4_B};
    CLU_broadcast_B.output_write_tiles = 6;

    // EXPECT_TRUE(false);

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
             {tc(SOHO_conv4_T, 8314, "SOHO, top"   )},    // 7516   GT: 8314
             {tc(SOHO_conv4_M, 8313, "SOHO, middle")}, // 7588   GT: 8313
             {tc(SOHO_conv4_B, 4294, "SOHO, bottom", pc)}, // 4012   GT: 4294  v17:4942 ---> delta:15.09%  WHY

             //conv4 CUBOID_16x16 + broadcast
             {tc(CLU_broadcast_T, 8379, "SOHO, top + broadcast"   )},    // 7559   GT: 8379 
             {tc(CLU_broadcast_M, 8378, "SOHO, middle + broadcast")}, // 7629   GT: 8378
             {tc(CLU_broadcast_B, 4295, "SOHO, bottom + broadcast", pc)}, // 4027   GT: 4295  v17:4952 ---> delta:15.30% WHY
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_EXEC_H662_16x16_SPARSITY) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 7, 64, 56, 6, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 8, 64, 56, 6, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 3, 64, 56, 2, 64, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv4_T};
    CLU_broadcast_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_M{SOHO_conv4_M};
    CLU_broadcast_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_B{SOHO_conv4_B};
    CLU_broadcast_B.output_write_tiles = 6;

    // execution mode 16x16 + sparsity
    DPUWorkload SOHO_conv4s_T{act_wt_sp(SOHO_conv4_T, sparsity)};
    DPUWorkload SOHO_conv4s_M{act_wt_sp(SOHO_conv4_M, sparsity)};
    DPUWorkload SOHO_conv4s_B{act_wt_sp(SOHO_conv4_B, sparsity)};

    // execution mode 16x16 + broadcast + sparsity
    DPUWorkload CLU_broadcast_s_T{act_wt_sp(CLU_broadcast_T, sparsity)};
    DPUWorkload CLU_broadcast_s_M{act_wt_sp(CLU_broadcast_M, sparsity)};
    DPUWorkload CLU_broadcast_s_B{act_wt_sp(CLU_broadcast_B, sparsity)};
    // EXPECT_TRUE(false);

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_16x16 + sparsity
             {tc(SOHO_conv4s_T, 8362, "SOHO, top + wt sparsity")},    // 7075   GT: 8362
             {tc(SOHO_conv4s_M, 8356, "SOHO, middle + wt sparsity")}, // 7086   GT: 8356
             {tc(SOHO_conv4s_B, 4373, "SOHO, bottom + wt sparsity")}, // 3917   GT: 4373  v17:5166 ---> delta:18.13% WHY

             //conv4 CUBOID_16x16 + broadcast + sparsity
             {tc(CLU_broadcast_s_T, 8426, "SOHO, top + broadcast + wt sparsity")},    // 7111   GT: 8426
             {tc(CLU_broadcast_s_M, 8425, "SOHO, middle + broadcast + wt sparsity")}, // 7118   GT: 8425
             {tc(CLU_broadcast_s_B, 4403, "SOHO, bottom + broadcast + wt sparsity")}, // 3937   GT: 4403  v17:5183 ---> delta:17.72% WHY
            // clang-format on
    };

    executeTests(tests);
}
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_H5_EXEC_16x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 6, 64, 56, 5, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 7, 64, 56, 5, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 6, 64, 56, 5, 64, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv4_T};
    CLU_broadcast_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_M{SOHO_conv4_M};
    CLU_broadcast_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_B{SOHO_conv4_B};
    CLU_broadcast_B.output_write_tiles = 6;

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
             {tc(SOHO_conv4_T, 7319, "SOHO, top"    )},    // 7149   GT: 7319   v17:8274 ---> delta:13.05%
             {tc(SOHO_conv4_M, 8312, "SOHO, middle")}, // 7251   GT: 8312
             {tc(SOHO_conv4_B, 7313, "SOHO, bottom")}, // 7382   GT: 7313   v17:8449 ---> delta:15.53%

             //conv4 CUBOID_16x16 + broadcast
             {tc(CLU_broadcast_T, 7383, "SOHO, top + broadcast"   )},    // 7189   GT: 7383   v17:8384 ---> delta:13.56%
             {tc(CLU_broadcast_M, 8376, "SOHO, middle + broadcast")}, // 7288   GT: 8376
             {tc(CLU_broadcast_B, 7402, "SOHO, bottom + broadcast")}, // 7422   GT: 7402   v17:8552 ---> delta:15.54%
            // clang-format on

            ///
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_H5_EXEC_16x16_SPARSITY) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 6, 64, 56, 5, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 7, 64, 56, 5, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 6, 64, 56, 5, 64, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv4_T};
    CLU_broadcast_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_M{SOHO_conv4_M};
    CLU_broadcast_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_B{SOHO_conv4_B};
    CLU_broadcast_B.output_write_tiles = 6;

    // execution mode 16x16 + sparsity
    DPUWorkload SOHO_conv4s_T{act_wt_sp(SOHO_conv4_T, sparsity)};
    DPUWorkload SOHO_conv4s_M{act_wt_sp(SOHO_conv4_M, sparsity)};
    DPUWorkload SOHO_conv4s_B{act_wt_sp(SOHO_conv4_B, sparsity)};

    // execution mode 16x16 + broadcast + sparsity
    DPUWorkload CLU_broadcast_s_T{act_wt_sp(CLU_broadcast_T, sparsity)};
    DPUWorkload CLU_broadcast_s_M{act_wt_sp(CLU_broadcast_M, sparsity)};
    DPUWorkload CLU_broadcast_s_B{act_wt_sp(CLU_broadcast_B, sparsity)};

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_16x16 + sparsity
             {tc(SOHO_conv4s_T, 7360, "SOHO, top + wt sparsity" )},    // 6741   GT: 7366   v17:8344 ---> delta:13.28%
             {tc(SOHO_conv4s_M, 8420, "SOHO, middle + wt sparsity")}, // 6763   GT: 8419
             {tc(SOHO_conv4s_B, 7354, "SOHO, bottom + wt sparsity")}, // 6771   GT: 7354   v17:8440 ---> delta:14.77%

             //conv4 CUBOID_16x16 + broadcast + sparsity
             {tc(CLU_broadcast_s_T, 7407, "SOHO, top + broadcast + wt sparsity"  )},    // 6784   GT: 7433   v17:8447 ---> delta:13.64%
             {tc(CLU_broadcast_s_M, 8420, "SOHO, middle + broadcast + wt sparsity")}, // 6805   GT: 8420
             {tc(CLU_broadcast_s_B, 7407, "SOHO, bottom + broadcast + wt sparsity")}, // 6808   GT: 7407   v17:8530 ---> delta:15.16%
            // clang-format on

            ///
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_H5_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 6, 64, 56, 5, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 7, 64, 56, 5, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 6, 64, 56, 5, 64, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv4_8x16_T{mod_execution(SOHO_conv4_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_M{mod_execution(SOHO_conv4_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_B{mod_execution(SOHO_conv4_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv4_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv4_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv4_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 6;

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off

             //SOHO conv4 CUBOID_8x16 
             {tc(SOHO_conv4_8x16_T, 7305, "SOHO, top"   )},    // 7484   GT: 7305
             {tc(SOHO_conv4_8x16_M, 8295, "SOHO, middle")}, // 7653   GT: 8295
             {tc(SOHO_conv4_8x16_B, 7299, "SOHO, bottom")}, // 7770   GT: 7299

              //conv4 CUBOID_8x16 + broadcast
             {tc(CLU_broadcast_8x16_T, 7371, "SOHO, top + broadcast"   )},    // 7530   GT: 7371 
             {tc(CLU_broadcast_8x16_M, 8357, "SOHO, middle + broadcast")}, // 7696   GT: 8357
             {tc(CLU_broadcast_8x16_B, 7391, "SOHO, bottom + broadcast")}, // 7806   GT: 7391
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv4_H5_EXEC_8x16_SPARSITY) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 6, 64, 56, 5, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 7, 64, 56, 5, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 6, 64, 56, 5, 64, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv4_8x16_T{mod_execution(SOHO_conv4_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_M{mod_execution(SOHO_conv4_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_B{mod_execution(SOHO_conv4_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv4_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv4_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv4_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 6;

    // execution mode 8x16 + sparsity
    DPUWorkload SOHO_conv4s_8x16_T{act_wt_sp(SOHO_conv4_8x16_T, sparsity)};
    DPUWorkload SOHO_conv4s_8x16_M{act_wt_sp(SOHO_conv4_8x16_M, sparsity)};
    DPUWorkload SOHO_conv4s_8x16_B{act_wt_sp(SOHO_conv4_8x16_B, sparsity)};

    // execution mode 8x16 + broadcast + sparsity
    DPUWorkload CLU_broadcast_s_8x16_T{act_wt_sp(CLU_broadcast_8x16_T, sparsity)};
    DPUWorkload CLU_broadcast_s_8x16_M{act_wt_sp(CLU_broadcast_8x16_M, sparsity)};
    DPUWorkload CLU_broadcast_s_8x16_B{act_wt_sp(CLU_broadcast_8x16_B, sparsity)};

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_8x16 + sparsity
             {tc(SOHO_conv4s_8x16_T, 7338, "SOHO, top + wt sparsity"   )},    // 7185   GT: 7338
             {tc(SOHO_conv4s_8x16_M, 8350, "SOHO, middle + wt sparsity", pc, 27)}, // 7218   GT: 8395
             {tc(SOHO_conv4s_8x16_B, 7337, "SOHO, bottom + wt sparsity")}, // 7214   GT: 7337 

             //conv4 CUBOID_8x16 + broadcast + sparsity
             {tc(CLU_broadcast_s_8x16_T, 7403, "SOHO, top + broadcast + wt sparsity"   )},    // 7226   GT: 7403
             {tc(CLU_broadcast_s_8x16_M, 8391, "SOHO, middle + broadcast + wt sparsity",pc)}, // 7257   GT: 8392
             {tc(CLU_broadcast_s_8x16_B, 7424, "SOHO, bottom + broadcast + wt sparsity")}, // 7257   GT: 7424
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv8_H3_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 4, 96, 28, 3, 96, 1, 0)};
    const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 5, 96, 28, 3, 96, 0, 0)};
    // const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    const DPUWorkload SOK_conv8{mk_SOK(28, 16, 96, 28, 16, 16)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv8_8x16_T{mod_execution(SOHO_conv8_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv8_8x16_M{mod_execution(SOHO_conv8_M, ExecutionMode::CUBOID_8x16)};
    // const DPUWorkload SOHO_conv8_8x16_B{mod_execution(SOHO_conv8_B, ExecutionMode::CUBOID_8x16)};

    const DPUWorkload SOK_conv8_8x16{mod_execution(SOK_conv8, ExecutionMode::CUBOID_8x16)};
    DPUWorkload no_br{SOK_conv8_8x16};
    no_br.output_write_tiles = 1;
    no_br.isi_strategy = ISIStrategy::CLUSTERING;

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv8_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv8_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 6;

    // DPUWorkload CLU_broadcast_8x16_B{SOHO_conv8_8x16_B};
    // CLU_broadcast_8x16_B.output_write_tiles = 6;

    // no_fail = 0;
    const std::vector<GTestCase> tests{
            // clang-format off
             //SOH conv4 CUBOID_8x16 
             {tc(SOHO_conv8_8x16_T, 4753, "SOHO, top "   )},    // 4419   GT: 4753
             {tc(SOHO_conv8_8x16_M, 4746, "SOHO, middle" )}, // 4403   GT: 4746
             //{tc(SOHO_conv8_8x16_B, 3850, "SOHO h1, bottom",pc+1 )}, // 4252   GT: 3850   v17:5058 ---> delta:31.38% CHECKGTLNLMTL

              //conv4 CUBOID_8x16 + broadcast
             {tc(CLU_broadcast_8x16_T, 4821, "SOHO, top + broadcast"   )},    // 4419   GT:  4815
             {tc(CLU_broadcast_8x16_M, 4821, "SOHO, middle + broadcast")}, // 4405   GT:  4818
             //{tc(CLU_broadcast_8x16_B, 3849, "SOHO h1, bottom + broadcast",pc+1)}, // 4257   GT:  3843  v17:5144 ---> delta:33.85% CHECKGTLNLMTL

             //SOK
             {tc(SOK_conv8_8x16, 5162, "SOK, all tensors are the same")}, // 5335   GT: 5162   v17:7586 ---> delta:46.96% CHECKGTLNLMTL
             {tc(no_br, 4835, "No broadcast", pc, 60)}, //v17:7565 5302
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv8_H3_EXEC_16x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 4, 96, 28, 3, 96, 1, 0)};
    const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 5, 96, 28, 3, 96, 0, 0)};
    // const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    const DPUWorkload SOK_conv8{mk_SOK(28, 16, 96, 28, 16, 16)};
    DPUWorkload no_br{SOK_conv8};
    no_br.output_write_tiles = 1;
    no_br.isi_strategy = ISIStrategy::CLUSTERING;

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv8_T};
    CLU_broadcast_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_M{SOHO_conv8_M};
    CLU_broadcast_M.output_write_tiles = 6;

    // DPUWorkload CLU_broadcast_B{SOHO_conv8_B};
    // CLU_broadcast_B.output_write_tiles = 6;

    // no_fail=false;
    // ASSERT_FALSE(true);
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
             {tc(SOHO_conv8_T, 4746, "SOHO, top"   )},    // 4388   GT: 4746   v17:5644 ---> delta:18.92%  CHECKGTLNLMTL
             {tc(SOHO_conv8_M, 4748, "SOHO, middle")}, // 4434   GT: 4748   v17:5801 ---> delta:22.18%
             //{tc(SOHO_conv8_B, 3918, "SOHO h1, bottom")}, // 4225   GT: 3918   v17:5634 ---> delta:43.80%   CHECKGTLNLMTL

             //conv4 CUBOID_16x16 + broadcast
             {tc(CLU_broadcast_T, 4773, "SOHO, top + broadcast"  )},    // 4410   GT:  4773   v17:5659 ---> delta:18.56% CHECKGTLNLMTL
             {tc(CLU_broadcast_M, 4838, "SOHO, middle + broadcast")}, // 4455   GT:  4838   v17:5819 ---> delta:20.28%  CHECKGTLNLMTL
             //{tc(CLU_broadcast_B, 3881, "SOHO h1, bottom + broadcast")}, // 4253   GT:  3881   v17:5647 ---> delta:45.50%   CHECKGTLNLMTL

             //SOK
             {tc(SOK_conv8,      3756, "SOK, all tensors are the same")}, // 3987   GT: 3756   v17:4593 ---> delta:22.28% CHECKGTLNLMTL
             {tc(no_br, 3466, "SOK, all tensors are the same")}, //v17:4531 3956
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOK_Conv8_EXEC_16x16) {
    const DPUWorkload SOK_conv8{mk_SOK(28, 16, 96, 28, 16, 16)};

    // no_fail=false;
    const std::vector<GTestCase> tests{
            {tc(SOK_conv8, 3814, "SOK, all tensors are the same",
                pc)},  // 3987   GT: 3756   v17:4593 ---> delta:22.28% CHECKGTLNLMTL
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOK_Conv8_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOK_conv8{mk_SOK(28, 16, 96, 28, 16, 16)};

    // execution mode 8x16
    const DPUWorkload SOK_conv8_8x16{mod_execution(SOK_conv8, ExecutionMode::CUBOID_8x16)};

    const std::vector<GTestCase> tests{
            {tc(SOK_conv8_8x16, 5162, "SOK, all tensors are the same",
                pc)},  // 5335   GT: 5162   v17:7586 ---> delta:46.96% CHECKGTLNLMTL
    };

    executeTests(tests);
}
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv8_H2_EXEC_16x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 3, 96, 28, 2, 96, 1, 0)};
    const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 4, 96, 28, 2, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 3, 96, 28, 2, 96, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv8_T};
    CLU_broadcast_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_M{SOHO_conv8_M};
    CLU_broadcast_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_B{SOHO_conv8_B};
    CLU_broadcast_B.output_write_tiles = 6;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
             {tc(SOHO_conv8_T, 4748, "SOHO, top", pc)},    // 4291   GT: 4748   v17:5564 ---> delta:17.19%
             {tc(SOHO_conv8_M, 4745, "SOHO, middle",pc)}, // 4338   GT: 4745   v17:5743 ---> delta:21.03%
             {tc(SOHO_conv8_B, 4749, "SOHO, bottom", pc)}, // 4369   GT: 4749   v17:5699 ---> delta:20.00%  CHECKGTLNLMTL

             //conv4 CUBOID_16x16 + broadcast
             {tc(CLU_broadcast_T, 4756, "SOHO, top + broadcast" , pc)},    // 4317   GT:  4756   v17:5577 ---> delta:17.26%
             {tc(CLU_broadcast_M, 4762, "SOHO, middle + broadcast",pc)}, // 4361   GT:  4762   v17:5758 ---> delta:20.92%
             {tc(CLU_broadcast_B, 4765, "SOHO, bottom + broadcast", pc)}, // 4393   GT:  4765   v17:5711 ---> delta:19.85%  CHECKGTLNLMTL

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv8_H1_EXEC_16x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 2, 96, 28, 1, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 3, 96, 28, 1, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv8_T};
    CLU_broadcast_T.output_write_tiles = 6;

    // DPUWorkload CLU_broadcast_M{SOHO_conv8_M};
    // CLU_broadcast_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_B{SOHO_conv8_B};
    CLU_broadcast_B.output_write_tiles = 6;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
             {tc(SOHO_conv8_T, 3994, "SOHO, top")},    //GT:  
             //{tc(SOHO_conv8_M, 4745, "SOHO, middle",pc,22)}, 
             {tc(SOHO_conv8_B, 3918, "SOHO, bottom")}, 

             //conv4 CUBOID_16x16 + broadcast
             
             {tc(CLU_broadcast_T, 4013, "SOHO, top + broadcast" )},    //GT:  
             //{tc(CLU_broadcast_M, 4762, "SOHO, middle + broadcast",pc,21)}, 
             {tc(CLU_broadcast_B, 3881, "SOHO, bottom + broadcast")},

            // clang-format on
    };

    executeTests(tests);
}
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv8_H2_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 3, 96, 28, 2, 96, 1, 0)};
    const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 4, 96, 28, 2, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 3, 96, 28, 2, 96, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv8_8x16_T{mod_execution(SOHO_conv8_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv8_8x16_M{mod_execution(SOHO_conv8_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv8_8x16_B{mod_execution(SOHO_conv8_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv8_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv8_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv8_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 6;

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
             //SOH conv4 CUBOID_8x16 
             {tc(SOHO_conv8_8x16_T, 4747, "SOHO, top"   )},    // 4322   GT: 4747
             {tc(SOHO_conv8_8x16_M, 4746, "SOHO, middle")}, // 4373   GT: 4746
             {tc(SOHO_conv8_8x16_B, 4746, "SOHO, bottom")}, // 4416   GT: 4746

              //conv4 CUBOID_8x16 + broadcast
             {tc(CLU_broadcast_8x16_T, 4816, "SOHO, top + broadcast" ,pc)},    // 4328   GT:  4816
             {tc(CLU_broadcast_8x16_M, 4781, "SOHO, middle + broadcast")}, // 4373   GT:  4781
             {tc(CLU_broadcast_8x16_B, 4816, "SOHO, bottom + broadcast")}, // 4423   GT:  4816

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU40, SOHO_Conv8_H1_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 2, 96, 28, 1, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 3, 96, 28, 1, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv8_8x16_T{mod_execution(SOHO_conv8_T, ExecutionMode::CUBOID_8x16)};
    // const DPUWorkload SOHO_conv8_8x16_M{mod_execution(SOHO_conv8_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv8_8x16_B{mod_execution(SOHO_conv8_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv8_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 6;

    // DPUWorkload CLU_broadcast_8x16_M{SOHO_conv8_8x16_M};
    // CLU_broadcast_8x16_M.output_write_tiles = 6;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv8_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 6;

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
             //SOH conv4 CUBOID_8x16 
               {tc(SOHO_conv8_8x16_T, 3879, "SOHO, top"   )},    //GT: 
             //{tc(SOHO_conv8_8x16_M, 4746, "SOHO, middle")}, 
               {tc(SOHO_conv8_8x16_B, 3850, "SOHO, bottom",pc)},

              //conv4 CUBOID_8x16 + broadcast
               {tc(CLU_broadcast_8x16_T, 3946, "SOHO, top + broadcast" )},    //GT:  
             //{tc(CLU_broadcast_8x16_M, 4781, "SOHO, middle + broadcast")}, 
               {tc(CLU_broadcast_8x16_B, 3849, "SOHO, bottom + broadcast",pc,26)},

            // clang-format on
    };

    executeTests(tests);
}

class Regression_tests_CONV_EISXW_127644_Model_N_NPU40 : public Regression_tests_CONV_EISXW_127649_Model_E_NPU40 {
    // base wls, before split
    // const DPUWorkload wl_{
    //         // orig Layer
    //         VPUDevice::VPU_4_0,
    //         Operation::CONVOLUTION,
    //         {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // input dimensions
    //         {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // output dimensions
    //         {3, 3},                                        // kernels
    //         {1, 1},                                        // strides
    //         {1, 1, 1, 1},                                  // padding
    //         ExecutionMode::CUBOID_16x16,                   // execution mode
    //         ActivationFunction::NONE,                      // activation
    //         0.0F,                                          // act_sparsity
    //         0.0F,                                          // weight_sparsity
    //         {swz_def, swz_def},                            // input_swizzling
    //         {swz_def},                                     // output_swizzling
    //         1,                                             // output_write_tiles
    //         {0, 0, 0, 0},                                  // offsets
    //         ISIStrategy::CLUSTERING,                       // isi_strategy
    //         false,                                         // weight_sparsity_enabled
    // };

    // const DPUWorkload wl_halfK{
    //         // orig Layer
    //         VPUDevice::VPU_4_0,
    //         Operation::CONVOLUTION,
    //         {VPUTensor(16, 16, 160, 1, DataType::UINT8)},  // input dimensions
    //         {VPUTensor(16, 16, 80, 1, DataType::UINT8)},   // output dimensions
    //         {3, 3},                                        // kernels
    //         {1, 1},                                        // strides
    //         {1, 1, 1, 1},                                  // padding
    //         ExecutionMode::CUBOID_16x16,                   // execution mode
    //         ActivationFunction::NONE,                      // activation
    //         0.0F,                                          // act_sparsity
    //         0.0F,                                          // weight_sparsity
    //         {swz_def, swz_def},                            // input_swizzling
    //         {swz_def},                                     // output_swizzling
    //         1,                                             // output_write_tiles
    //         {0, 0, 0, 0},                                  // offsets
    //         ISIStrategy::CLUSTERING,                       // isi_strategy
    //         false,                                         // weight_sparsity_enabled
    // };

protected:
    static DPUWorkload mk_SOK(unsigned int in_c, unsigned int out_c) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(16, 16, in_c, 1, DataType::UINT8, Layout::ZXY)},   // input dimensions
                {VPUTensor(16, 16, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                       // kernels
                {1, 1},                                                       // strides
                {1, 1, 1, 1},                                                 // padding
                ExecutionMode::CUBOID_16x16,                                  // execution mode
                ActivationFunction::NONE,                                     // activation
                0.0F,                                                         // act_sparsity
                0.0F,                                                         // weight_sparsity
                {swz_def, swz_def},                                           // input_swizzling
                {swz_def},                                                    // output_swizzling
                4,                                                            // output_write_tiles
                {0, 0, 0, 0},                                                 // offsets
                ISIStrategy::SPLIT_OVER_K,                                    // isi_strategy
                false,                                                        //
        };
    }
    static DPUWorkload mk_HK(unsigned int in_w, unsigned int in_h, unsigned int in_c, unsigned int out_w,
                             unsigned int out_h, unsigned int out_c, unsigned int top_padd, unsigned int btm_padd) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                             // kernels
                {1, 1},                                                             // strides
                {top_padd, btm_padd, 1, 1},                                         // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                4,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::CLUSTERING,                                            // isi_strategy
                false,                                                              //
        };
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " in_C:" + std::to_string(wl.inputs[0].get_shape()[2]) + 
                              " out_C:" + std::to_string(wl.outputs[0].get_shape()[2]) + 
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }

    Regression_tests_CONV_EISXW_127644_Model_N_NPU40() {
    }
};

TEST_F(Regression_tests_CONV_EISXW_127644_Model_N_NPU40, Wl_in_ch_160_test_SOHO) {
    // execution mode 8x16
    const DPUWorkload SOHO_8x16_T{mod_execution(mk_SOHO(16, 5, 160, 16, 4, 160, 1, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_M{mod_execution(mk_SOHO(16, 6, 160, 16, 4, 160, 0, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_B{mod_execution(mk_SOHO(16, 5, 160, 16, 4, 160, 0, 1), ExecutionMode::CUBOID_8x16)};

    const DPUWorkload SOHO_halfk_8x16_T{
            mod_execution(mk_SOHO(16, 5, 160, 16, 4, 80, 1, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_halfk_8x16_M{
            mod_execution(mk_SOHO(16, 6, 160, 16, 4, 80, 0, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_halfk_8x16_B{
            mod_execution(mk_SOHO(16, 5, 160, 16, 4, 80, 0, 1), ExecutionMode::CUBOID_8x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(SOHO_8x16_T, 7382, "SOHO, top"   )},    // 7742   GT: 7382
             {tc(SOHO_8x16_M, 7384, "SOHO, middle")}, // 7757   GT: 7384
             {tc(SOHO_8x16_B, 7385, "SOHO, bottom")}, // 7834   GT: 7385

             {tc(SOHO_halfk_8x16_T, 3811, "SOHO half, top"   )},    // 3571   GT: 3811
             {tc(SOHO_halfk_8x16_M, 3810, "SOHO half, middle")}, // 3566   GT: 3810
             {tc(SOHO_halfk_8x16_B, 3810, "SOHO half, bottom")}, // 3627   GT: 3810
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127644_Model_N_NPU40, Wl_in_ch_160_test_SOHO_BROADCAST) {
    // execution mode 8x16
    const DPUWorkload SOHO_8x16_T{mod_execution(mk_HK(16, 5, 160, 16, 4, 160, 1, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_M{mod_execution(mk_HK(16, 6, 160, 16, 4, 160, 0, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_8x16_B{mod_execution(mk_HK(16, 5, 160, 16, 4, 160, 0, 1), ExecutionMode::CUBOID_8x16)};

    const DPUWorkload SOHO_halfk_8x16_T{mod_execution(mk_HK(16, 5, 160, 16, 4, 80, 1, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_halfk_8x16_M{mod_execution(mk_HK(16, 6, 160, 16, 4, 80, 0, 0), ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_halfk_8x16_B{mod_execution(mk_HK(16, 5, 160, 16, 4, 80, 0, 1), ExecutionMode::CUBOID_8x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(SOHO_8x16_T, 7382+100, "HK , top"   )},//   GT: 
             {tc(SOHO_8x16_M, 7384+100, "HK, middle")}, //   GT: 
             {tc(SOHO_8x16_B, 7385+100, "HK, bottom")}, //   GT: 

             {tc(SOHO_halfk_8x16_T, 3811+100, "HK half, top"   )}, //    GT: 
             {tc(SOHO_halfk_8x16_M, 3810+100, "HK half, middle")}, //    GT: 
             {tc(SOHO_halfk_8x16_B, 3810+100, "HK half, bottom")}, //    GT:
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127644_Model_N_NPU40, Wl_in_ch_160_test_SOK) {
    const DPUWorkload SOK_t123_ch48{mk_SOK(160, 48)};
    const DPUWorkload SOK_t4_ch16{mk_SOK(160, 16)};

    const DPUWorkload SOK_ch32{mk_SOK(160, 32)};
    const DPUWorkload SOK_ch64{mk_SOK(160, 64)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            {tc(SOK_ch64, 12290, "SOK, K 32")},     // ?  GT: 12290  v17:11194
            {tc(SOK_t123_ch48, 9406, "SOK, K48")},  // 9370   GT: 9406   v17:8456 ---> delta:-10.10%
            {tc(SOK_ch32, 6221, "SOK, K 32")},      // ?  GT: 6221   v17:5619
            {tc(SOK_t4_ch16, 3435, "SOK,K16")},     // 3960   GT: 3435   v17:4399 ---> delta:28.06%  (WHY so BIG for
                                                    // 16channels?) todo CHECKGTLNLMTL

    };

    executeTests(tests);
}

class Regression_EISXW_127594_Marina_M_10Th_june_ELTWISE_NPU40 : public Regression_Tests {
    // base wls, before split
    // DPUWorkload wl_h28{
    //        VPUDevice::VPU_4_0,
    //        Operation::ELTWISE,
    //        {VPUTensor(250, 28, 64, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
    //        {VPUTensor(250, 28, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
    //        {1, 1},                                                    // kernels
    //        {1, 1},                                                    // strides
    //        {0, 0, 0, 0},                                              // padding
    //        ExecutionMode::CUBOID_16x16,                                // execution mode
    //        ActivationFunction::NONE,                                  // activation
    //        0.0F,                                                      // act_sparsity
    //        0.0F,                                                      // weight_sparsity
    //        {swz_def, swz_def},                                        // input_swizzling
    //        {swz_def},                                                 // output_swizzling
    //        1,                                                         // output_write_tiles
    //        {0, 0, 0, 0},                                              // offsets
    //        ISIStrategy::CLUSTERING,                                   // isi_strategy
    //        false,                                                     //
    //};

    // DPUWorkload wl_h27{
    //         VPUDevice::VPU_4_0,
    //         Operation::ELTWISE,
    //         {VPUTensor(250, 27, 64, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
    //         {VPUTensor(250, 27, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
    //         {1, 1},                                                    // kernels
    //         {1, 1},                                                    // strides
    //         {0, 0, 0, 0},                                              // padding
    //         ExecutionMode::CUBOID_16x16,                                // execution mode
    //         ActivationFunction::NONE,                                  // activation
    //         0.0F,                                                      // act_sparsity
    //         0.0F,                                                      // weight_sparsity
    //         {swz_def, swz_def},                                        // input_swizzling
    //         {swz_def},                                                 // output_swizzling
    //         1,                                                         // output_write_tiles
    //         {0, 0, 0, 0},                                              // offsets
    //         ISIStrategy::CLUSTERING,                                   // isi_strategy
    //         false,                                                     //
    // };

protected:
    const DPUWorkload wl_h7_8x16{
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(250, 7, 64, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(250, 7, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                    // kernels
            {1, 1},                                                    // strides
            {0, 0, 0, 0},                                              // padding
            ExecutionMode::CUBOID_8x16,                                // execution mode
            ActivationFunction::NONE,                                  // activation
            0.0F,                                                      // act_sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            ISIStrategy::CLUSTERING,                                   // isi_strategy
            false,                                                     //
    };

    const DPUWorkload wl_h6_8x16{
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(250, 6, 64, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(250, 6, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                    // kernels
            {1, 1},                                                    // strides
            {0, 0, 0, 0},                                              // padding
            ExecutionMode::CUBOID_8x16,                                // execution mode
            ActivationFunction::NONE,                                  // activation
            0.0F,                                                      // act_sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            ISIStrategy::CLUSTERING,                                   // isi_strategy
            false,                                                     //
    };

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type +
                              " H=:" + std::to_string(wl.inputs[0].get_shape()[1]) + 
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_EISXW_127594_Marina_M_10Th_june_ELTWISE_NPU40() {
        show_wl_info = false;
    }
};

TEST_F(Regression_EISXW_127594_Marina_M_10Th_june_ELTWISE_NPU40, ELTWISE_Marina_Test) {
    // DPUWorkload wl_h7_8x16_B{wl_h7_8x16};
    // wl_h7_8x16_B.output_write_tiles = 4;

    // DPUWorkload wl_h6_8x16_B{wl_h6_8x16};
    // wl_h6_8x16_B.output_write_tiles = 4;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
           // {tc(wl_h7_16x16, 2665,   "SOHO ")},    // 3225   GTL: 2665  GTM:3197     
           // {tc(wl_h6_16x16, 2504,   "SOHO ")},    // 3168   GTL: 2504  GTM: 3132    

            //8x16 is teh allowed  mod here, FOR MTL based NETWORK is a huge delta!
            {tc(wl_h7_8x16, 2655,   "SOHO H7", pc )},    // 3319   GT: 2655  GTM :3240  CHECKGTLNLMTL
            {tc(wl_h6_8x16, 2448,   "SOHO H6", pc)},    // 2786   GT: 2448  => delta -20.35% GTM: 3158 CHECKGTLNLMTL

            // BROADCAST 
            // {tc(wl_h7_8x16_B, 0,   "SOHO H7 B", pc,50 )}, //2970
            // {tc(wl_h6_8x16_B, 0,   "SOHO H6 B", pc,50)}, //3209

           // {tc(wl_h7_4x16, 3498,   "SOHO ")},    // 4408   GT: 3498  GTM:3708
           // {tc(wl_h6_4x16, 3481,   "SOHO ")},    // 4289   GT: 3481  GTM:3644
            // clang-format on
    };

    executeTests(tests);
}

// Stride  2  CONV
class Regression_tests_CONV_STRIDE2_EISXW_117195_NPU40 : public Regression_Tests {
    // base wl, before split
    // DPUWorkload wl_ref{
    //         // orig Layer
    //         VPUDevice::VPU_4_0,
    //         Operation::CONVOLUTION,
    //         {VPUTensor(28, 28, 256, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
    //         {VPUTensor(14, 14, 256, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
    //         {3, 3},                                                     // kernels
    //         {2, 2},                                                     // strides
    //         {1, 0, 1, 0},                                               // padding
    //         ExecutionMode::CUBOID_16x16,                                // execution mode
    //         ActivationFunction::NONE,                                   // activation
    //         0.0F,                                                       // act_sparsity     // in ticket is: 0.6
    //         0.0F,                                                       // weight_sparsity  // in ticket is: 0.400662
    //         {swz_def, swz_def},                                         // input_swizzling
    //         {swz_def},                                                  // output_swizzling
    //         1,                                                          // output_write_tiles
    //         {0, 0, 0, 0},                                               // offsets
    //         ISIStrategy::CLUSTERING,                                    // isi_strategy
    //         false,                                                      // weight_sparsity_enabled
    // };

protected:
    /// with this function, starting from a SOHO base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param in_w: the input tensor width
    /// @param in_h: the input tensor height
    /// @param out_w: the output tensor width
    /// @param out_h: output tensor height
    /// @param top_padd: top padding for workload
    /// @param btm_padd: bottom padding for workload
    /// @param Tint: input tensor Dtype
    /// @param Tout: output tensor Dtype
    /// @param kw: kernel width
    /// @param kh: kernel height
    /// @param stride: wl stride
    /// @return a new DPUWorkload
    static DPUWorkload mk_SOHO(unsigned int in_w, unsigned int in_h, unsigned int in_c, unsigned int out_w,
                               unsigned int out_h, unsigned int out_c, unsigned int top_padd, unsigned int btm_padd) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                             // kernels
                {2, 2},                                                             // strides
                {top_padd, btm_padd, 1, 0},                                         // padding
                ExecutionMode::CUBOID_8x16,                                         // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                1,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::CLUSTERING,                                            // isi_strategy
                false,                                                              //
        };
    }

    static DPUWorkload mk_SOK(unsigned int in_w, unsigned int in_h, unsigned int in_c, unsigned int out_w,
                              unsigned int out_h, unsigned int out_c) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                             // kernels
                {2, 2},                                                             // strides
                {1, 0, 1, 0},                                                       // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                4,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::SPLIT_OVER_K,                                          // isi_strategy
                false,                                                              //
        };
    }

    const DPUWorkload SOHO_8x16_T{mk_SOHO(28, 8, 256, 14, 4, 256, 1, 0)};
    const DPUWorkload SOHO_8x16_M{mk_SOHO(28, 9, 256, 14, 4, 256, 0, 0)};
    const DPUWorkload SOHO_8x16_B{mk_SOHO(28, 5, 256, 14, 2, 256, 0, 0)};

    const DPUWorkload SOHO_16x16_T{mod_execution(SOHO_8x16_T, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload SOHO_16x16_M{mod_execution(SOHO_8x16_M, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload SOHO_16x16_B{mod_execution(SOHO_8x16_B, ExecutionMode::CUBOID_16x16)};

    const DPUWorkload SOK_16x16_K64{mk_SOK(28, 28, 256, 14, 14, 256 / 4)};
    const DPUWorkload SOK_16x16_K32{mk_SOK(28, 28, 256, 14, 14, 32)};
    const DPUWorkload SOK_16x16_K16{mk_SOK(28, 28, 256, 14, 14, 16)};

    const float act_sparsity = 0.6f;
    const float wt_sparsity = 0.400662f;

    /// @brief set sparsities for a wl, they are given as parameters
    ///
    /// @param wl: The DPU Workload, we set its sparsities
    /// @param input_sparsity_enable: activate/deactivate input sparsity, it's value could be true (that means input
    /// sparsity is activate) or false (that means input sparsity is deactivate)
    /// @param act_sparsity: value for input sparsity; should be [0, 1]
    /// @param weight_sparsity_enable: activate/deactivate weight sparsity , it's value could be true (that means
    /// weight sparsity is activate) or false (that means weight sparsity is deactivate)
    /// @param weight_sparsity: value for weight sparsity; should be [0, 1]
    ///
    /// @return the wl with new sparsities
    DPUWorkload sparsity_init(const DPUWorkload& wl, bool input_sparsity_enable, float act_spars,
                              bool weight_sparsity_enable, float weight_sparsity) {
        DPUWorkload wl_ref{wl};
        // input sparsity
        wl_ref.inputs[0].set_sparsity(input_sparsity_enable);
        wl_ref.act_sparsity = act_spars;

        // weight sparsity
        wl_ref.weight_sparsity_enabled = weight_sparsity_enable;
        wl_ref.weight_sparsity = weight_sparsity;

        return wl_ref;
    };

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_tests_CONV_STRIDE2_EISXW_117195_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload& wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " out_H:" + std::to_string(wl.outputs[0].get_shape()[1]) + 
                              " wt_spars_enabled:" +(wl.weight_sparsity_enabled ? "true" : "false") +
                              " act_spars_enabled:" +(wl.inputs[0].get_sparsity() ? "true" : "false") +
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }
};

TEST_F(Regression_tests_CONV_STRIDE2_EISXW_117195_NPU40, CONV_UINT8_STRIDE2_SOHO) {
    const std::vector<GTestCase> tests{
            // CUBOID_8x16
            {tc(SOHO_8x16_T, 18730, "SOHO, top H4")},      // 17250   GT: 18730
            {tc(SOHO_8x16_M, 18729, "SOHO, middle H4")},   // 17282   GT: 18729
            {tc(SOHO_8x16_B, 18726, "SOHO, bottom  H2")},  //  16911  GT: 18726

            // CUBOID_16x16
            {tc(SOHO_16x16_T, 18678, "SOHO, top H4")},     // 16901   GT: 18678
            {tc(SOHO_16x16_M, 18678, "SOHO, middle H4")},  // 16940   GT: 18678
            {tc(SOHO_16x16_B, 18673, "SOHO, bottom H2")},  //    GT: 18673
    };
    // EXPECT_TRUE(false);
    executeTests(tests);
}
TEST_F(Regression_tests_CONV_STRIDE2_EISXW_117195_NPU40, CONV_UINT8_STRIDE2_SOK) {
    const std::vector<GTestCase> tests{
            // SOK only 16x61
            {tc(SOK_16x16_K64, 19081, "SOK, K64 all the same")},  //   GT: 19735
            {tc(SOK_16x16_K32, 9601, "SOK, K32 all the same")},   //   GT: 9601
            {tc(SOK_16x16_K16, 4990, "SOK, K16 all the same")},   //   GT: 4990
    };
    // EXPECT_TRUE(false);
    executeTests(tests);
}

class Regression_tests_MEXP_C2_EISXW_126389_NPU40 : public Regression_Tests {
protected:
    /// with this function, starting from a SOHO base wl, you can create a new workload with the values you provide as
    /// parameters
    /// @param in_w: the input tensor width
    /// @param in_h: the input tensor height
    /// @param out_w: the output tensor width
    /// @param out_h: output tensor height
    /// @param top_padd: top padding for workload
    /// @param btm_padd: bottom padding for workload
    /// @param Tint: input tensor Dtype
    /// @param Tout: output tensor Dtype
    /// @param kw: kernel width
    /// @param kh: kernel height
    /// @param stride: wl stride
    /// @return a new DPUWorkload
    static DPUWorkload mk_SOHO(Operation op, unsigned int in_w, unsigned int in_h, unsigned int in_c,
                               unsigned int out_w, unsigned int out_h, unsigned int out_c) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                op,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                             // kernels
                {1, 1},                                                             // strides
                {0, 0, 0, 0},                                                       // padding
                ExecutionMode::CUBOID_8x16,                                         // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                1,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::CLUSTERING,                                            // isi_strategy
                false,                                                              //
        };
    }

    static DPUWorkload mk_SOK(Operation op, unsigned int in_w, unsigned int in_h, unsigned int in_c, unsigned int out_w,
                              unsigned int out_h, unsigned int out_c) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                op,
                {VPUTensor(in_w, in_h, in_c, 1, DataType::UINT8, Layout::ZXY)},     // input dimensions
                {VPUTensor(out_w, out_h, out_c, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {1, 1},                                                             // kernels
                {1, 1},                                                             // strides
                {0, 0, 0, 0},                                                       // padding
                ExecutionMode::CUBOID_16x16,                                        // execution mode
                ActivationFunction::NONE,                                           // activation
                0.0F,                                                               // act_sparsity
                0.0F,                                                               // weight_sparsity
                {swz_def, swz_def},                                                 // input_swizzling
                {swz_def},                                                          // output_swizzling
                4,                                                                  // output_write_tiles
                {0, 0, 0, 0},                                                       // offsets
                ISIStrategy::SPLIT_OVER_K,                                          // isi_strategy
                false,                                                              //
        };
    }

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_tests_MEXP_C2_EISXW_126389_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " Operation:" + Operation_ToText.at(static_cast<int>(wl.op)) + 
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }
};

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4634_4662_SOHO_test) {
    const DPUWorkload wl_SOHO_8x16{mk_SOHO(Operation::CONVOLUTION, 28, 7, 256, 28, 7, 128)};
    const DPUWorkload wl_SOHO_16x16{mod_execution(wl_SOHO_8x16, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_SOHO_4x16{mod_execution(wl_SOHO_8x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4634 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_SOHO_8x16, 3935, "SOHO 4634 _8x16")},   // v17:  GT:
             {tc(wl_SOHO_16x16, 3864, "SOHO 4634 _16x16")}, // v17:  GT:
             {tc(wl_SOHO_4x16, 3850, "SOHO 4634 _4x16")},   // v17:  GT:
            // clang-format on
    };
    // EXPECT_TRUE(false);
    executeTests(tests);
}
TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4634_4662_HK_test) {
    DPUWorkload wl_HK_8x16{mk_SOHO(Operation::CONVOLUTION, 28, 7, 256, 28, 7, 128)};
    wl_HK_8x16.output_write_tiles = 4;
    const DPUWorkload wl_HK_16x16{mod_execution(wl_HK_8x16, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_HK_4x16{mod_execution(wl_HK_8x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4634 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_HK_8x16,   4154, "HK 4634  _8x16 "  )},  // v17:  GT: 
             {tc(wl_HK_16x16,  3979, "HK 4634 _16x16 "  )}, // v17:  GT: 
             {tc(wl_HK_4x16,   3933, "HK 4634  _4x16 "  )},  // v17:  GT:
            // clang-format on
    };
    // EXPECT_TRUE(false);
    executeTests(tests);
}
TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4634_4662_SOK_test) {
    const DPUWorkload wl_SOK_16x16{mk_SOK(Operation::CONVOLUTION, 28, 28, 256, 28, 28, 32)};
    const DPUWorkload wl_SOK_8x16{mod_execution(wl_SOK_16x16, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload wl_SOK_4x16{mod_execution(wl_SOK_16x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4634 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_SOK_16x16,  3490, "SOK 4634 16x16", pc,36  )}, // v17:  GT:
             {tc(wl_SOK_8x16,  3435, "SOK 4634 8x16"  )},   // v17:  GT:
             {tc(wl_SOK_4x16,  3596, "SOK 4634 4x16"  )},   // v17:  GT:
            // clang-format on
    };
    // EXPECT_TRUE(false);
    executeTests(tests);
}

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4634_4662_SOHO_extra_K64_32_16_test) {
    const DPUWorkload SOHO_8x16{mk_SOHO(Operation::CONVOLUTION, 28, 7, 256, 28, 7, 128)};
    const DPUWorkload SOHO_16x16{mod_execution(SOHO_8x16, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload SOHO_4x16{mod_execution(SOHO_8x16, ExecutionMode::CUBOID_4x16)};

    // K64
    DPUWorkload SOHO_K64_8x16{SOHO_8x16};
    SOHO_K64_8x16.outputs[0] = VPUTensor(28, 7, 64, 1, DataType::UINT8);
    DPUWorkload SOHO_K64_16x16{SOHO_16x16};
    SOHO_K64_16x16.outputs[0] = VPUTensor(28, 7, 64, 1, DataType::UINT8);
    DPUWorkload SOHO_K64_4x16{SOHO_4x16};
    SOHO_K64_4x16.outputs[0] = VPUTensor(28, 7, 64, 1, DataType::UINT8);

    // K32
    DPUWorkload SOHO_K32_8x16{SOHO_8x16};
    SOHO_K32_8x16.outputs[0] = VPUTensor(28, 7, 32, 1, DataType::UINT8);
    DPUWorkload SOHO_K32_16x16{SOHO_16x16};
    SOHO_K32_16x16.outputs[0] = VPUTensor(28, 7, 32, 1, DataType::UINT8);
    DPUWorkload SOHO_K32_4x16{SOHO_4x16};
    SOHO_K32_4x16.outputs[0] = VPUTensor(28, 7, 32, 1, DataType::UINT8);

    // K16
    DPUWorkload SOHO_K16_8x16{std::move(SOHO_8x16)};
    SOHO_K16_8x16.outputs[0] = VPUTensor(28, 7, 16, 1, DataType::UINT8);
    DPUWorkload SOHO_K16_16x16{std::move(SOHO_16x16)};
    SOHO_K16_16x16.outputs[0] = VPUTensor(28, 7, 16, 1, DataType::UINT8);
    DPUWorkload SOHO_K16_4x16{std::move(SOHO_4x16)};
    SOHO_K16_4x16.outputs[0] = VPUTensor(28, 7, 16, 1, DataType::UINT8);

    std::cout << "*************************************************** wl_4634 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off

             {tc(SOHO_K64_8x16,  2044, "wl SOHO k64 8x16 ", pc, 29)},  // v17:  
             {tc(SOHO_K32_8x16,  1078, "wl SOHO k32 8x16 ")},  // v17:
             {tc(SOHO_K16_8x16,  848 , "wl SOHO k16 8x16 ")},  // v17:

             {tc(SOHO_K64_16x16, 2052, "wl SOHO k64 16x16 ")},  // v17:
             {tc(SOHO_K32_16x16, 1122, "wl SOHO k32 16x16 ")},  // v17:
             {tc(SOHO_K16_16x16, 691 , "wl SOHO k16 16x16 ")},  // v17:
             
             {tc(SOHO_K64_4x16,  1995, "wl SOHO k64 4x16 ")},
             {tc(SOHO_K32_4x16,  1095, "wl SOHO k32 4x16 ")},
             {tc(SOHO_K16_4x16,  946 , "wl SOHO k16 4x16 ")},

            // clang-format on
    };
    // EXPECT_TRUE(false);
    executeTests(tests);
}

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4648_SOHO_test) {
    const DPUWorkload wl_SOHO_8x16{mk_SOHO(Operation::CONVOLUTION, 28, 7, 128, 28, 7, 256)};
    const DPUWorkload wl_SOHO_16x16{mod_execution(wl_SOHO_8x16, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_SOHO_4x16{mod_execution(wl_SOHO_8x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4648 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_SOHO_16x16, 3946, "SOHO 4648 16x16")}, // v17:  GT:
             {tc(wl_SOHO_8x16,  3964, "SOHO 4648 8x16" )}, // v17:  GT:
             {tc(wl_SOHO_4x16,  3973, "SOHO 4648 4x16" )}, // v17:  GT:
            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4648_HK_test) {
    DPUWorkload wl_HK_8x16{mk_SOHO(Operation::CONVOLUTION, 28, 7, 128, 28, 7, 256)};
    wl_HK_8x16.output_write_tiles = 4;
    const DPUWorkload wl_HK_16x16{mod_execution(wl_HK_8x16, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_HK_4x16{mod_execution(wl_HK_8x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4648 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_HK_16x16,   4051, "HK 4648 16x16"  )}, // v17:  GT: 
             {tc(wl_HK_8x16,    4228, "HK 4648 8x16"  , pc,27 )}, // v17:  GT: 
             {tc(wl_HK_4x16,    4096, "HK 4648 4x16"   )}, // v17:  GT:
            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4648_SOK_test) {
    const DPUWorkload wl_SOK_16x16{mk_SOK(Operation::CONVOLUTION, 28, 28, 128, 28, 28, 64)};
    const DPUWorkload wl_SOK_8x16{mod_execution(wl_SOK_16x16, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload wl_SOK_4x16{mod_execution(wl_SOK_16x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4648 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_SOK_16x16,  3681, "SOK 4648 16x16" , pc, 26 )}, // v17:  GT:
             {tc(wl_SOK_8x16,   3498, "SOK 4648 8x16"   )}, // v17:  GT:
             {tc(wl_SOK_4x16,   3562, "SOK 4648 4x16"   )}, // v17:  GT:
            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, CONV_4676_SOK_SOHO_HK_test) {
    const DPUWorkload SOHO_T_M_8x16{
            // top and middle
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 9, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(14, 4, 512, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                       // kernels
            {2, 2},                                       // strides
            {0, 0, 0, 1},                                 // padding
            ExecutionMode::CUBOID_8x16,                   // execution mode
            ActivationFunction::NONE,                     // activation
            0.0F,                                         // act_sparsity
            0.0F,                                         // weight_sparsity
            {swz_def, swz_def},                           // input_swizzling
            {swz_def},                                    // output_swizzling
            1,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::CLUSTERING,                      // isi_strategy
            false,                                        // weight_sparsity_enabled
    };

    DPUWorkload HK_T_M_8x16{SOHO_T_M_8x16};
    HK_T_M_8x16.output_write_tiles = 4;

    const DPUWorkload SOHO_T_M_16x16{mod_execution(SOHO_T_M_8x16, ExecutionMode::CUBOID_16x16)};
    DPUWorkload HK_T_M_16x16{SOHO_T_M_16x16};
    HK_T_M_16x16.output_write_tiles = 4;

    const DPUWorkload SOHO_T_M_4x16{mod_execution(SOHO_T_M_8x16, ExecutionMode::CUBOID_4x16)};
    DPUWorkload HK_T_M_4x16{SOHO_T_M_4x16};
    HK_T_M_4x16.output_write_tiles = 4;

    //////////////////////////////////////////////////////////////

    DPUWorkload SOHO_B_8x16{SOHO_T_M_8x16};
    SOHO_B_8x16.inputs[0].set_shape({28, 4, 128, 1});
    SOHO_B_8x16.outputs[0].set_shape({14, 2, 512, 1});
    SOHO_B_8x16.padding = {0, 1, 0, 1};

    DPUWorkload HK_B_8x16{SOHO_B_8x16};
    HK_B_8x16.output_write_tiles = 4;

    const DPUWorkload SOHO_B_16x16{mod_execution(SOHO_B_8x16, ExecutionMode::CUBOID_16x16)};
    DPUWorkload HK_B_16x16{SOHO_B_16x16};
    HK_B_16x16.output_write_tiles = 4;

    const DPUWorkload SOHO_B_4x16{mod_execution(SOHO_B_8x16, ExecutionMode::CUBOID_4x16)};
    DPUWorkload HK_B_4x16{SOHO_B_4x16};
    HK_B_4x16.output_write_tiles = 4;

    //////////////////////////////////////////////////////////////

    const DPUWorkload SOK_16x16{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, 128, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(14, 14, 128, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                        // kernels
            {2, 2},                                        // strides
            {0, 1, 0, 1},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            4,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    const DPUWorkload SOK_8x16{mod_execution(SOK_16x16, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOK_4x16{mod_execution(SOK_16x16, ExecutionMode::CUBOID_4x16)};

    std::cout << "*************************************************** wl_4676  K3x3 s =2 "
                 "****************************************************\n ";
    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(SOK_16x16,  3681, "SOK 4676 16x16"  )}, // v17:  GT:
             {tc(SOK_8x16,   3498, "SOK 4676 8x16"   )}, // v17:  GT:
             {tc(SOK_4x16,   3562, "SOK 4676 4x16"   )}, // v17:  GT:


             {tc(SOHO_T_M_16x16, 18743, "SOHO _T_M 4676 16x16")}, // v17:  GT:
             {tc(SOHO_B_16x16,   18737, "SOHO _B   4676 16x16")}, // v17:  GT:

             {tc(SOHO_T_M_8x16,  18766, "SOHO _T_M 4676 8x16" )}, // v17:  GT:
             {tc(SOHO_B_8x16,    18762, "SOHO _B   4676 8x16" )}, // v17:  GT:

             {tc(SOHO_T_M_4x16,  18826, "SOHO _T_M 4676 4x16" )}, // v17:  GT:
             {tc(SOHO_B_4x16,    18823, "SOHO _B   4676 4x16" )}, // v17:  GT:



             {tc(HK_T_M_16x16, 18829, "HK _T_M 4676 16x16")}, // v17:  GT:
             {tc(HK_B_16x16,   18764, "HK _B   4676 16x16")}, // v17:  GT:

             {tc(HK_T_M_8x16,  18880, "HK _T_M 4676 8x16" )}, // v17:  GT:
             {tc(HK_B_8x16,    18781, "HK _B   4676 8x16" )}, // v17:  GT:

             {tc(HK_T_M_4x16,  18914, "HK _T_M 4676 4x16" )}, // v17:  GT:
             {tc(HK_B_4x16,    18821, "HK _B   4676 4x16" )}, // v17:  GT:

            // clang-format on
    };
}

TEST_F(Regression_tests_MEXP_C2_EISXW_126389_NPU40, ELTWISE_1662_test) {
    DPUWorkload wl_SOHO_8x16{mk_SOHO(Operation::ELTWISE, 28, 7, 256, 28, 7, 256)};
    DPUWorkload wl_HK_8x16{wl_SOHO_8x16};
    wl_HK_8x16.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             //CUBOID_8x16
             {tc(wl_SOHO_8x16, 1234, "SOHO "   )},          //1082 v17:1593  GT: 1226-1234
             {tc(wl_HK_8x16, 2155, "SOHO + broadcast"   )}, //1082 v17:1593  GT: 2077-2155

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

class Regression_tests_Big_CONV_EISXW_131119_NPU40 : public Regression_Tests {
    // conv4 base wl, before split
    // const DPUWorkload wl_{
    //        // orig Layer
    //        VPUDevice::VPU_4_0,
    //        Operation::CONVOLUTION,
    //        {VPUTensor(3000, 1, 80, 1, DataType::FLOAT16)},   // input dimensions
    //        {VPUTensor(3000, 1, 512, 1, DataType::FLOAT16)},  // output dimensions
    //        {3, 1},                                           // kernels
    //        {1, 1},                                           // strides
    //        {0, 0, 1, 1},                                     // padding
    //        ExecutionMode::CUBOID_16x16,                      // execution mode
    //        ActivationFunction::NONE,                         // activation
    //        0.0F,                                             // act_sparsity
    //        0.0F,                                             // weight_sparsity
    //        {swz_def, swz_def},                               // input_swizzling
    //        {swz_def},                                        // output_swizzling
    //        1,                                                // output_write_tiles
    //        {0, 0, 0, 0},                                     // offsets
    //        ISIStrategy::CLUSTERING,                          // isi_strategy
    //        false,                                            // weight_sparsity_enabled
    //};

protected:
    DPUWorkload wl_16x16{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(3000, 1, 80, 1, DataType::FLOAT16, Layout::ZXY)},   // input dimensions
            {VPUTensor(3000, 1, 128, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
            {3, 1},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 1, 1},                                                  // padding
            ExecutionMode::CUBOID_16x16,                                   // execution mode
            ActivationFunction::NONE,                                      // activation
            0.0F,                                                          // act_sparsity
            0.0F,                                                          // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            ISIStrategy::CLUSTERING,                                       // isi_strategy
            false,                                                         //
    };

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_tests_Big_CONV_EISXW_131119_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }
};

TEST_F(Regression_tests_Big_CONV_EISXW_131119_NPU40, Big_CONV_test) {
    DPUWorkload wl_8x16{mod_execution(wl_16x16, ExecutionMode::CUBOID_8x16)};
    DPUWorkload wl_4x16{mod_execution(wl_16x16, ExecutionMode::CUBOID_4x16)};

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_16x16,367050, "SOK no broadcast"   )},  //247920 v17:386686  GT:367050
             {tc(wl_8x16, 363839, "SOK no broadcast"   )},  //245905 v17:383467  GT:363839
             {tc(wl_4x16, 367203, "SOK no broadcast"   )},  //267827 v17:412267  GT:367203

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// wls taken as examples are from ticket: EISXW-91782
class Regression_tests_DifferentSwizz_NPU40 : public Regression_Tests {
protected:
    const VPUDevice dev{VPUDevice::VPU_4_0};
    // Layer 1 elm Float to int with Layout change!
    const DPUWorkload s1_elmws_c0{
            dev,
            Operation::ELTWISE,
            {VPUTensor(114, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(114, 3, 224, 1, DataType::UINT8, Layout::YZX)},    // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            ExecutionMode::CUBOID_16x16,                                  // execution mode
            ActivationFunction::RELU,                                     // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                         // input_swizzling
            {Swizzling::KEY_0},                                           // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            ISIStrategy::CLUSTERING,                                      // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };

    const DPUWorkload makeL1_Elmwise() const {
        DPUWorkload clone = s1_elmws_c0;
        {
            clone.inputs = {VPUTensor(115, 3, 224, 1, DataType::FLOAT16, Layout::ZXY)};
            clone.outputs = {VPUTensor(115, 3, 224, 1, DataType::UINT8, Layout::YZX)};
        }
        return clone;
    }
    const DPUWorkload s1_elmws_c1{makeL1_Elmwise()};

    // Layer 2 conv
    const DPUWorkload s2_conv_c0{
            dev,
            Operation::CONVOLUTION,
            {VPUTensor(224, 114, 3, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(112, 56, 64, 1, DataType::UINT8)},  // output dimensions
            {7, 7},                                        // kernels
            {2, 2},                                        // strides
            {3, 0, 3, 2},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::RELU,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {
                    Swizzling::KEY_0,
                    Swizzling::KEY_0,
            },  // input_swizzling
            {
                    Swizzling::KEY_5,
            },                        // output_swizzling
            1,                        // output_write_tiles
            {0, 0, 0, 0},             // offsets
            ISIStrategy::CLUSTERING,  // isi_strategy
            false,                    // weight_sparsity_enabled
    };

    const DPUWorkload makeL2_Conv7x7() const {
        DPUWorkload clone = s2_conv_c0;
        {
            clone.inputs = {VPUTensor(224, 115, 3, 1, DataType::UINT8)};
            // same output
            clone.padding = {0, 2, 3, 2};
        }
        return clone;
    }
    const DPUWorkload s2_conv_c1{makeL2_Conv7x7()};

    // Layer 3 maxpool
    // const DPUWorkload s3_maxp_c0{
    //        dev,
    //        Operation::MAXPOOL,
    //        {VPUTensor(112, 56, 64, 1, DataType::UINT8)},  // input dimensions
    //        {VPUTensor(56, 28, 64, 1, DataType::UINT8)},   // output dimensions
    //        {3, 3},                                        // kernels
    //        {2, 2},                                        // strides
    //        {1, 0, 1, 0},                                  // padding
    //        ExecutionMode::CUBOID_16x16,                   // execution mode
    //        ActivationFunction::RELU,                      // activation
    //        0.0F,                                          // act_sparsity
    //        0.0F,                                          // weight_sparsity
    //        {Swizzling::KEY_5, Swizzling::KEY_5},          // input_swizzling
    //        {Swizzling::KEY_5},                            // output_swizzling
    //        1,                                             // output_write_tiles
    //        {0, 0, 0, 0},                                  // offsets
    //        ISIStrategy::SPLIT_OVER_H,                     // isi_strategy
    //        false,                                         // weight_sparsity_enabled
    //};
    // const DPUWorkload makeL3_MaxPooling3x3() {
    //    DPUWorkload clone = s3_maxp_c0;
    //    {
    //        clone.inputs = {VPUTensor(112, 57, 64, 1, DataType::UINT8)};  // why 57?
    //        // same output
    //        clone.padding = {0, 0, 1, 0};
    //    }
    //    return clone;
    //}
    // const DPUWorkload s3_maxp_c1{makeL3_MaxPooling3x3()};

    // Layer 4 Conv
    const DPUWorkload s4_conv_c0{
            dev,
            Operation::CONVOLUTION,
            {VPUTensor(56, 28, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 28, 64, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                       // kernels
            {1, 1},                                       // strides
            {0, 0, 0, 0},                                 // padding
            ExecutionMode::CUBOID_16x16,                  // execution mode
            ActivationFunction::RELU,                     // activation
            0.0F,                                         // act_sparsity
            0.0F,                                         // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_0},         // input_swizzling
            {Swizzling::KEY_0},                           // output_swizzling
            1,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::SPLIT_OVER_H,                    // why not CLU? // isi_strategy
            false,                                        // weight_sparsity_enabled
    };
    const DPUWorkload s4_conv_c1{s4_conv_c0};

    // Layer 5 Elm to float
    const DPUWorkload s5_elmws_c0{
            dev,
            Operation::ELTWISE,
            {VPUTensor(56, 28, 64, 1, DataType::UINT8, Layout::ZXY)},    // input dimensions
            {VPUTensor(56, 28, 64, 1, DataType::FLOAT16, Layout::XYZ)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            ExecutionMode::CUBOID_8x16,                                  // execution mode
            ActivationFunction::RELU,                                    // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                        // input_swizzling
            {Swizzling::KEY_0},                                          // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            ISIStrategy::SPLIT_OVER_H,                                   // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };
    const DPUWorkload s5_elmws_c1{s5_elmws_c0};

    Regression_tests_DifferentSwizz_NPU40() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " Operation: \t" + Operation_ToText.at(static_cast<int>(wl.op)) +"\n";

        // clang-format on

        return message;
    }
};
// we do not have GT, for now tests are failing
// TEST_F(Regression_tests_DifferentSwizz_NPU40, DiffSwizz_DPU) {
//     const std::string modelFile{VPU_4_0_MODEL_PATH};
//     VPUCostModel test_model{modelFile};
//     EXPECT_TRUE(test_model.nn_initialized());
//
//     // EXPECT_EQ(1, 0);  // force fail, uncomment to have the log in tests
//
//     {
//         std::cout << "\n----------------------CLUSTER "
//                      "0---------------------------------------------------------------------------  ";
//
//         const std::vector<GTestCase> tests{
//                 // clang-format off
//              {tc(s1_elmws_c0,-1, "CLU0 wl1"   )},  //9748
//              {tc(s2_conv_c0, -1, "CLU0 wl2"   )},  //118658
//              //{tc(s3_maxp_c0, -1, "CLU0 wl3"   )},  //11432
//              {tc(s4_conv_c0, -1, "CLU0 wl4"   )},  //4324
//              {tc(s5_elmws_c0, -1,"CLU0 wl5"   )},  //28251
//
//                 // clang-format on
//         };
//
//         executeTests(tests);
//     }
//     {
//         std::cout << "\n----------------------CLUSTER "
//                      "1-----------------------------------------------------------------------------  ";
//
//         const std::vector<GTestCase> tests{
//                 // clang-format off
//              {tc(s1_elmws_c1, -1, "CLU1 wl1"   )},  //9815
//              {tc(s2_conv_c1, -1, "CLU1 wl2"   )},  //117425
//             // {tc(s3_maxp_c1, 25326, "CLU1 wl3"   )},  //11540
//              {tc(s4_conv_c1, -1, "CLU1 wl4"   )},  //4324
//              {tc(s5_elmws_c1, -1,"CLU1 wl5"   )},  //28251
//
//                 // clang-format on
//         };
//
//         executeTests(tests);
//     }
// }

class Regression_tests_ModelA : public Regression_Tests {
protected:
    DPUWorkload CONV75_SOHO_16x16{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(112, 28, 32, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(112, 28, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                       // input_swizzling
            {Swizzling::KEY_5},                                         // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::CLUSTERING,                                    // isi_strategy
            false,
    };

    DPUWorkload CONV75_SOK_16x16{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(112, 112, 32, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(112, 112, 16, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {1, 1},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            ExecutionMode::CUBOID_16x16,                                 // execution mode
            ActivationFunction::NONE,                                    // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                        // input_swizzling
            {Swizzling::KEY_5},                                          // output_swizzling
            4,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            ISIStrategy::SPLIT_OVER_K,                                   // isi_strategy
            false,
    };

    static DPUWorkload mk_CONV251_SOHO(unsigned int in_h, unsigned int out_h, unsigned int top_padd,
                                       unsigned int btm_padd) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(7, in_h, 160, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(7, out_h, 64, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                      // kernels
                {1, 1},                                                      // strides
                {top_padd, btm_padd, 1, 1},                                  // padding
                ExecutionMode::CUBOID_16x16,                                 // execution mode
                ActivationFunction::NONE,                                    // activation
                0.0F,                                                        // act_sparsity
                0.0F,                                                        // weight_sparsity
                {swz_def, swz_def},                                          // input_swizzling
                {swz_def},                                                   // output_swizzling
                1,                                                           // output_write_tiles
                {0, 0, 0, 0},                                                // offsets
                ISIStrategy::CLUSTERING,                                     // isi_strategy
                false,                                                       //
        };
    }

    DPUWorkload CONV251_SOK_16x16{
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(7, 7, 160, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(7, 7, 16, 1, DataType::UINT8, Layout::ZXY)},   // output dimensions
            {3, 3},                                                   // kernels
            {1, 1},                                                   // strides
            {1, 1, 1, 1},                                             // padding
            ExecutionMode::CUBOID_16x16,                              // execution mode
            ActivationFunction::NONE,                                 // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                     // input_swizzling
            {Swizzling::KEY_5},                                       // output_swizzling
            4,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            ISIStrategy::SPLIT_OVER_K,                                // isi_strategy
            false,
    };

    static DPUWorkload mk_GC105_SOHO(unsigned int in_h, unsigned int out_h, unsigned int ch, unsigned int top_padd,
                                     unsigned int btm_padd) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, in_h, ch, 1, DataType::UINT8, Layout::ZXY)},   // input dimensions
                {VPUTensor(56, out_h, ch, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                       // kernels
                {1, 1},                                                       // strides
                {top_padd, btm_padd, 1, 1},                                   // padding
                ExecutionMode::CUBOID_16x16,                                  // execution mode
                ActivationFunction::NONE,                                     // activation
                0.0F,                                                         // act_sparsity
                0.0F,                                                         // weight_sparsity
                {swz_def, swz_def},                                           // input_swizzling
                {swz_def},                                                    // output_swizzling
                1,                                                            // output_write_tiles
                {0, 0, 0, 0},                                                 // offsets
                ISIStrategy::CLUSTERING,                                      // isi_strategy
                false,                                                        //
        };
    }

    static DPUWorkload mk_GC105_SOK(unsigned int ch) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 56, ch, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(56, 56, ch, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                    // kernels
                {1, 1},                                                    // strides
                {1, 1, 1, 1},                                              // padding
                ExecutionMode::CUBOID_16x16,                               // execution mode
                ActivationFunction::NONE,                                  // activation
                0.0F,                                                      // act_sparsity
                0.0F,                                                      // weight_sparsity
                {swz_def, swz_def},                                        // input_swizzling
                {swz_def},                                                 // output_swizzling
                4,                                                         // output_write_tiles
                {0, 0, 0, 0},                                              // offsets
                ISIStrategy::SPLIT_OVER_K,                                 // isi_strategy
                false,                                                     //
        };
    }

    static DPUWorkload mk_GC124_SOHO(unsigned int in_h, unsigned int out_h, unsigned int ch, unsigned int top_padd,
                                     unsigned int btm_padd) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, in_h, ch, 1, DataType::UINT8, Layout::ZXY)},   // input dimensions
                {VPUTensor(28, out_h, ch, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                       // kernels
                {2, 2},                                                       // strides
                {top_padd, btm_padd, 0, 1},                                   // padding
                ExecutionMode::CUBOID_16x16,                                  // execution mode
                ActivationFunction::NONE,                                     // activation
                0.0F,                                                         // act_sparsity
                0.0F,                                                         // weight_sparsity
                {swz_def, swz_def},                                           // input_swizzling
                {swz_def},                                                    // output_swizzling
                1,                                                            // output_write_tiles
                {0, 0, 0, 0},                                                 // offsets
                ISIStrategy::CLUSTERING,                                      // isi_strategy
                false,                                                        //
        };
    }

    static DPUWorkload mk_GC124_SOK(unsigned int ch) {
        return DPUWorkload{
                VPUDevice::VPU_4_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(56, 56, ch, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(28, 28, ch, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                    // kernels
                {2, 2},                                                    // strides
                {0, 1, 0, 1},                                              // padding
                ExecutionMode::CUBOID_16x16,                               // execution mode
                ActivationFunction::NONE,                                  // activation
                0.0F,                                                      // act_sparsity
                0.0F,                                                      // weight_sparsity
                {swz_def, swz_def},                                        // input_swizzling
                {swz_def},                                                 // output_swizzling
                4,                                                         // output_write_tiles
                {0, 0, 0, 0},                                              // offsets
                ISIStrategy::SPLIT_OVER_K,                                 // isi_strategy
                false,                                                     //
        };
    }

    const DPUWorkload mod_execution(const DPUWorkload& wl, ExecutionMode em) {
        DPUWorkload wl_{wl};
        wl_.execution_order = em;
        return wl_;
    }

    Regression_tests_ModelA() {
        show_wl_info = false;
    }

    std::string test_message(const DPUWorkload &wl, std::string tensor_type) const override {
        // clang-format off
        std::string message = tensor_type + " tensor " +
                              " Op: " + Operation_ToText.at(static_cast<int>(wl.op)) +
                              " in_ch: " + std::to_string(wl.inputs[0].get_shape()[2]) +
                              " out_ch: " + std::to_string(wl.outputs[0].get_shape()[2]) +
                              " exec_order:" + ExecutionMode_ToText.at(static_cast<int>(wl.execution_order)) +"\n";

        // clang-format on

        return message;
    }
};

TEST_F(Regression_tests_ModelA, CONV75_test) {
    DPUWorkload wl_SOHO_16x16{CONV75_SOHO_16x16};
    DPUWorkload wl_SOHO_8x16{mod_execution(CONV75_SOHO_16x16, ExecutionMode::CUBOID_8x16)};
    DPUWorkload wl_SOHO_4x16{mod_execution(CONV75_SOHO_16x16, ExecutionMode::CUBOID_4x16)};

    DPUWorkload wl_SOK_16x16{CONV75_SOK_16x16};
    DPUWorkload wl_SOK_8x16{mod_execution(CONV75_SOK_16x16, ExecutionMode::CUBOID_8x16)};
    DPUWorkload wl_SOK_4x16{mod_execution(CONV75_SOK_16x16, ExecutionMode::CUBOID_4x16)};

    // SOHO broadcast
    DPUWorkload wl_SOHO_16x16_B{wl_SOHO_16x16};
    wl_SOHO_16x16_B.output_write_tiles = 4;

    DPUWorkload wl_SOHO_8x16_B{wl_SOHO_8x16};
    wl_SOHO_8x16_B.output_write_tiles = 4;

    DPUWorkload wl_SOHO_4x16_B{wl_SOHO_4x16};
    wl_SOHO_4x16_B.output_write_tiles = 4;

    // SOK no broadcast
    DPUWorkload wl_SOK_16x16_noB{wl_SOK_16x16};
    wl_SOK_16x16_noB.output_write_tiles = 1;
    wl_SOK_16x16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    DPUWorkload wl_SOK_8x16_noB{wl_SOK_8x16};
    wl_SOK_8x16_noB.output_write_tiles = 1;
    wl_SOK_8x16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    DPUWorkload wl_SOK_4x16_noB{wl_SOK_4x16};
    wl_SOK_4x16_noB.output_write_tiles = 1;
    wl_SOK_4x16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_SOHO_16x16,3734, "SOHO", pc, 48)}, 
             {tc(wl_SOHO_8x16, 3826, "SOHO", pc, 30)}, 
             {tc(wl_SOHO_4x16, 4279, "SOHO", pc, 32)}, 

             {tc(wl_SOK_16x16, 12814, "SOK", pc, 57)}, 
             {tc(wl_SOK_8x16, 12800, "SOK", pc, 48)}, 
             {tc(wl_SOK_4x16, 12880, "SOK", pc, 41)}, 

             {tc(wl_SOHO_16x16_B, 7996, "SOHO + broadcast" , pc,45 )}, 
             {tc(wl_SOHO_8x16_B, 7637, "SOHO + broadcast"   )}, 
             {tc(wl_SOHO_4x16_B, 7059, "SOHO + broadcast"   )}, 

              {tc(wl_SOK_16x16_noB, 4902, "SOK no broadcast"   )},
              {tc(wl_SOK_8x16_noB, 6462, "SOK no broadcast"   )},
              {tc(wl_SOK_4x16_noB, 8833, "SOK no broadcast"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_ModelA, CONV251_SOHO_ExecMode_16x16) {
    DPUWorkload wl_T{mk_CONV251_SOHO(3, 2, 1, 0)};
    DPUWorkload wl_M{mk_CONV251_SOHO(4, 2, 0, 0)};
    DPUWorkload wl_B{mk_CONV251_SOHO(2, 1, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_Br{wl_T};
    wl_T_Br.output_write_tiles = 4;

    DPUWorkload wl_M_Br{wl_M};
    wl_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T, 1593, "SOHO, top"   )}, 
             {tc(wl_M, 1652, "SOHO, middle"   )}, 
             {tc(wl_B, 1441, "SOHO, bottom"   )}, 

             {tc(wl_T_Br, 1649, "SOHO + broadcast, top"   )}, 
             {tc(wl_M_Br, 1612, "SOHO + broadcast, middle"   )}, 
             {tc(wl_B_Br, 1445, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_ModelA, CONV251_SOHO_ExecMode_8x16) {
    DPUWorkload wl_T{mod_execution(mk_CONV251_SOHO(3, 2, 1, 0), ExecutionMode::CUBOID_8x16)};
    DPUWorkload wl_M{mod_execution(mk_CONV251_SOHO(4, 2, 0, 0), ExecutionMode::CUBOID_8x16)};
    DPUWorkload wl_B{mod_execution(mk_CONV251_SOHO(2, 1, 0, 1), ExecutionMode::CUBOID_8x16)};

    // SOHO broadcast
    DPUWorkload wl_T_Br{wl_T};
    wl_T_Br.output_write_tiles = 4;

    DPUWorkload wl_M_Br{wl_M};
    wl_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T, 1639, "SOHO, top", 11   )}, 
             {tc(wl_M, 1585, "SOHO, middle"   )}, 
             {tc(wl_B, 1437, "SOHO, bottom"   )}, 

             {tc(wl_T_Br, 1600, "SOHO + broadcast, top"   )}, 
             {tc(wl_M_Br, 1603, "SOHO + broadcast, middle"   )}, 
             {tc(wl_B_Br, 1477, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_ModelA, CONV251_SOHO_ExecMode_4x16) {
    DPUWorkload wl_T{mod_execution(mk_CONV251_SOHO(3, 2, 1, 0), ExecutionMode::CUBOID_4x16)};
    DPUWorkload wl_M{mod_execution(mk_CONV251_SOHO(4, 2, 0, 0), ExecutionMode::CUBOID_4x16)};
    DPUWorkload wl_B{mod_execution(mk_CONV251_SOHO(2, 1, 0, 1), ExecutionMode::CUBOID_4x16)};

    // SOHO broadcast
    DPUWorkload wl_T_Br{wl_T};
    wl_T_Br.output_write_tiles = 4;

    DPUWorkload wl_M_Br{wl_M};
    wl_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T, 1583, "SOHO, top"   )}, 
             {tc(wl_M, 1630, "SOHO, middle"   )}, 
             {tc(wl_B, 1384, "SOHO, bottom"   )}, 

             {tc(wl_T_Br, 1631, "SOHO + broadcast, top"   )}, 
             {tc(wl_M_Br, 1599, "SOHO + broadcast, middle"   )}, 
             {tc(wl_B_Br, 1383, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

TEST_F(Regression_tests_ModelA, CONV251_SOK_test) {
    DPUWorkload wl_16x16{CONV251_SOK_16x16};
    DPUWorkload wl_8x16{mod_execution(wl_16x16, ExecutionMode::CUBOID_8x16)};
    DPUWorkload wl_4x16{mod_execution(wl_16x16, ExecutionMode::CUBOID_4x16)};

    // SOK no broadcast
    DPUWorkload wl_16x16_noB{wl_16x16};
    wl_16x16_noB.output_write_tiles = 1;
    wl_16x16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    DPUWorkload wl_8x16_noB{wl_8x16};
    wl_8x16_noB.output_write_tiles = 1;
    wl_8x16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    DPUWorkload wl_4x16_noB{wl_4x16};
    wl_4x16_noB.output_write_tiles = 1;
    wl_4x16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_16x16, 967, "SOK"   )}, 
             {tc(wl_8x16, 1072, "SOK"   )}, 
             {tc(wl_4x16, 1194, "SOK"   )}, 

              {tc(wl_16x16_noB, 967, "SOK no broadcast", 11)},
              {tc(wl_8x16_noB, 998, "SOK no broadcast"   )},
              {tc(wl_4x16_noB, 1144, "SOK no broadcast"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************* GC105 **************************************************

// ************************************** ch64 **************************************
TEST_F(Regression_tests_ModelA, GC105_SOHO_ch64_ExecMode_16x16) {
    DPUWorkload wl_T{mk_GC105_SOHO(15, 14, 64, 1, 0)};
    DPUWorkload wl_M{mk_GC105_SOHO(16, 14, 64, 0, 0)};
    DPUWorkload wl_B{mk_GC105_SOHO(15, 14, 64, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_Br{wl_T};
    wl_T_Br.output_write_tiles = 4;

    DPUWorkload wl_M_Br{wl_M};
    wl_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T, 4334, "SOHO, top"   )}, 
             {tc(wl_M, 4337, "SOHO, middle"   )}, 
             {tc(wl_B, 4404, "SOHO, bottom"   )}, 

             {tc(wl_T_Br, 4495, "SOHO + broadcast, top"   )}, 
             {tc(wl_M_Br, 4495, "SOHO + broadcast, middle"   )}, 
             {tc(wl_B_Br, 4495, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************** ch32 **************************************
TEST_F(Regression_tests_ModelA, GC105_SOHO_ch32_ExecMode_16x16) {
    DPUWorkload wl_T{mk_GC105_SOHO(15, 14, 32, 1, 0)};
    DPUWorkload wl_M{mk_GC105_SOHO(16, 14, 32, 0, 0)};
    DPUWorkload wl_B{mk_GC105_SOHO(15, 14, 32, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_Br{wl_T};
    wl_T_Br.output_write_tiles = 4;

    DPUWorkload wl_M_Br{wl_M};
    wl_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T, 1410, "SOHO, top"   )}, 
             {tc(wl_M, 1405, "SOHO, middle"   )}, 
             {tc(wl_B, 1399, "SOHO, bottom"   )}, 

             {tc(wl_T_Br, 1562, "SOHO + broadcast, top"   )}, 
             {tc(wl_M_Br, 1596, "SOHO + broadcast, middle"   )}, 
             {tc(wl_B_Br, 1584, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************** ch16 **************************************
TEST_F(Regression_tests_ModelA, GC105_SOHO_ch16_ExecMode_16x16) {
    DPUWorkload wl_T{mk_GC105_SOHO(15, 14, 16, 1, 0)};
    DPUWorkload wl_M{mk_GC105_SOHO(16, 14, 16, 0, 0)};
    DPUWorkload wl_B{mk_GC105_SOHO(15, 14, 16, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_Br{wl_T};
    wl_T_Br.output_write_tiles = 4;

    DPUWorkload wl_M_Br{wl_M};
    wl_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T, 1445, "SOHO, top"   )}, 
             {tc(wl_M, 1413, "SOHO, middle"   )}, 
             {tc(wl_B, 1391, "SOHO, bottom"   )}, 

             {tc(wl_T_Br, 1724, "SOHO + broadcast, top"   )}, 
             {tc(wl_M_Br, 1647, "SOHO + broadcast, middle"   )}, 
             {tc(wl_B_Br, 1636, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************** SOK **************************************
TEST_F(Regression_tests_ModelA, GC105_SOK_ExecMode_16x16) {
    DPUWorkload wl_ch32{mk_GC105_SOK(32)};
    DPUWorkload wl_ch16{mk_GC105_SOK(16)};

    // SOK no broadcast
    DPUWorkload wl_ch32_noB{wl_ch32};
    wl_ch32_noB.output_write_tiles = 1;
    wl_ch32_noB.isi_strategy = ISIStrategy::CLUSTERING;

    DPUWorkload wl_ch16_noB{wl_ch16};
    wl_ch16_noB.output_write_tiles = 1;
    wl_ch16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_ch32, 4686, "SOK"   )}, 
             {tc(wl_ch16, 4694, "SOK"   )}, 

              {tc(wl_ch32_noB, 4553, "SOK no broadcast"   )},
              {tc(wl_ch16_noB, 4543, "SOK no broadcast"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************* GC124 **************************************************

// ************************************** ch64 **************************************
TEST_F(Regression_tests_ModelA, GC124_SOHO_ch64_ExecMode_16x16) {
    DPUWorkload wl_T_M{mk_GC124_SOHO(15, 7, 64, 0, 0)};
    DPUWorkload wl_B{mk_GC124_SOHO(14, 7, 64, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_M_Br{wl_T_M};
    wl_T_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T_M, 1356, "SOHO, top or middle"   )}, 
             {tc(wl_B, 1361, "SOHO, bottom"   )}, 

             {tc(wl_T_M_Br, 1515, "SOHO + broadcast, top or middle"   )}, 
             {tc(wl_B_Br, 1471, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************** ch32 **************************************
TEST_F(Regression_tests_ModelA, GC124_SOHO_ch32_ExecMode_16x16) {
    DPUWorkload wl_T_M{mk_GC124_SOHO(15, 7, 32, 0, 0)};
    DPUWorkload wl_B{mk_GC124_SOHO(14, 7, 32, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_M_Br{wl_T_M};
    wl_T_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T_M, 748, "SOHO, top or middle"   )}, 
             {tc(wl_B, 788, "SOHO, bottom"   )}, 

             {tc(wl_T_M_Br, 816, "SOHO + broadcast, top or middle"   )}, 
             {tc(wl_B_Br, 811, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************** ch16 **************************************
TEST_F(Regression_tests_ModelA, GC124_SOHO_ch16_ExecMode_16x16) {
    DPUWorkload wl_T_M{mk_GC124_SOHO(15, 7, 16, 0, 0)};
    DPUWorkload wl_B{mk_GC124_SOHO(14, 7, 16, 0, 1)};

    // SOHO broadcast
    DPUWorkload wl_T_M_Br{wl_T_M};
    wl_T_M_Br.output_write_tiles = 4;

    DPUWorkload wl_B_Br{wl_B};
    wl_B_Br.output_write_tiles = 4;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_T_M, 749, "SOHO, top or middle"   )}, 
             {tc(wl_B, 780, "SOHO, bottom"   )}, 

             {tc(wl_T_M_Br, 891, "SOHO + broadcast, top or middle"   )}, 
             {tc(wl_B_Br, 890, "SOHO + broadcast, bottom"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}

// ************************************** SOK **************************************
TEST_F(Regression_tests_ModelA, GC124_SOK_ExecMode_16x16) {
    DPUWorkload wl_ch32{mk_GC124_SOK(32)};
    DPUWorkload wl_ch16{mk_GC124_SOK(16)};

    // SOK no broadcast
    DPUWorkload wl_ch32_noB{wl_ch32};
    wl_ch32_noB.output_write_tiles = 1;
    wl_ch32_noB.isi_strategy = ISIStrategy::CLUSTERING;

    DPUWorkload wl_ch16_noB{wl_ch16};
    wl_ch16_noB.output_write_tiles = 1;
    wl_ch16_noB.isi_strategy = ISIStrategy::CLUSTERING;

    const std::vector<GTestCase> tests{
            // clang-format off
             {tc(wl_ch32, 2134, "SOK"   )}, 
             {tc(wl_ch16, 2335, "SOK"   )}, 

              {tc(wl_ch32_noB, 2050, "SOK no broadcast"   )},
              {tc(wl_ch16_noB, 2039, "SOK no broadcast"   )},

            // clang-format on
    };

    // EXPECT_TRUE(false);

    executeTests(tests);
}
}  // namespace VPUNN_unit_tests