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
#include "common_helpers.h"
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

class Regression_tests_DW_CONV_EISXW117314_NPU27 : public Regression_Tests {
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
                VPUDevice::VPU_2_7,
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
                VPUDevice::VPU_2_7,
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
                2,                                                     // output_write_tiles
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

    Regression_tests_DW_CONV_EISXW117314_NPU27() {
        show_wl_info = false;
        // no_fail = 0;  // ALL tests will fail
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
TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_3x3) {
    const int p{1};
    const int k{3};
    // STRIDE=1
    DPUWorkload wl_CLU_top{mk_SOHO(56, 15, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    DPUWorkload wl_CLU_mid{mk_SOHO(56, 16, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    DPUWorkload wl_CLU_btm{mk_SOHO(56, 15, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
        //SOHO
            {tc(wls_CLU_top[ch64], 6044,  "top")},  //   GT:6044
            {tc(wls_CLU_top[ch32], 3014,  "top")},  //   GT:3014
            {tc(wls_CLU_top[ch16], 2754,  "top")},  //   GT:2754   

            {tc(wls_CLU_mid[ch64], 6265,  "middle")},  //   GT:6265
            {tc(wls_CLU_mid[ch32], 3140,  "middle")},  //   GT:3140
            {tc(wls_CLU_mid[ch16], 2754,  "middle")},  //   GT:2754
                                                
            {tc(wls_CLU_btm[ch64], 6298,  "bottom")},  //   GT:6298
            {tc(wls_CLU_btm[ch32], 3134,  "bottom")},  //   GT:3134
            {tc(wls_CLU_btm[ch16], 2754,  "bottom")},  //   GT:2754  

           //SOK
            {tc(wls_SOK[ch64], 21342,  "SOK, all tensors are the same")},   //   GT:21342
            {tc(wls_SOK[ch32], 10571,  "SOK, all tensors are the same")},   //   GT:10571
            {tc(wls_SOK[ch16], 9361,   "SOK, all tensors are the same")},   //   GT:9361

            // clang-format on
    };

    executeTests(tests);

    //****************************************************************************************************************************************************
    // STRIDE=2
    DPUWorkload wl_CLU_top_stride2{mk_SOHO(112, 28, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    DPUWorkload wl_CLU_mid_stride2{mk_SOHO(112, 29, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};
    DPUWorkload wl_CLU_btm_stride2{mk_SOHO(112, 28, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k, 2)};

    DPUWorkload wl_SOK_stride2{mk_SOK(112, 112, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
        //SOHO
            {tc(wls_CLU_top_stride2[ch64], 6055, "top")},  //   GT:6055
            {tc(wls_CLU_top_stride2[ch32], 3272, "top")},  //   GT:3272  
            {tc(wls_CLU_top_stride2[ch16], 3221, "top")},  //   GT:3221   
                                                                                      
            {tc(wls_CLU_mid_stride2[ch64], 6298, "middle")},  //   GT:6298
            {tc(wls_CLU_mid_stride2[ch32], 3339, "middle")},  //   GT:3339  
            {tc(wls_CLU_mid_stride2[ch16], 3221, "middle")},  //   GT:3221  
                                                                 
            {tc(wls_CLU_btm_stride2[ch64], 6055, "bottom")},  //    GT:6055
            {tc(wls_CLU_btm_stride2[ch32], 3337, "bottom")},  //    GT:3337   
            {tc(wls_CLU_btm_stride2[ch16], 3320, "bottom")},  //    GT:3320   
                                                                                   

            //SOK
            {tc(wls_SOK_stride2[ch64], 21617 ,"SOK, all tensors are the same")},    //   GT:21617
            {tc(wls_SOK_stride2[ch32], 11268,   "SOK, all tensors are the same")},  //   GT:11268
            {tc(wls_SOK_stride2[ch16], 10981,   "SOK, all tensors are the same")},  //   GT:10981

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_3x5) {
    DPUWorkload wl_SOK{mk_SOK(56, 58, 56, 56, DataType::UINT8, DataType::UINT8, 3, 5)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 34520,  "SOK, all tensors are the same")},  //    GT:34520
            {tc(wls_SOK[ch32], 17153,  "SOK, all tensors are the same")},  //    GT:17153
            {tc(wls_SOK[ch16], 15574,  "SOK, all tensors are the same")},  //    GT:15574

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_5x3) {
    DPUWorkload wl_SOK{mk_SOK(58, 56, 56, 56, DataType::UINT8, DataType::UINT8, 5, 3)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 34609,  "SOK, all tensors are the same")},  //   GT:34609
            {tc(wls_SOK[ch32], 17134,  "SOK, all tensors are the same")},  //   GT:17134 
            {tc(wls_SOK[ch16], 15514,  "SOK, all tensors are the same")},  //   GT:15514

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_3x7) {
    DPUWorkload wl_SOK{mk_SOK(56, 60, 56, 56, DataType::UINT8, DataType::UINT8, 3, 7)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 47882,  "SOK, all tensors are the same")},  //    GT:47882
            {tc(wls_SOK[ch32], 23697,  "SOK, all tensors are the same")},  //    GT:23697
            {tc(wls_SOK[ch16], 21804,  "SOK, all tensors are the same")},  //    GT:21804

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_7x3) {
    DPUWorkload wl_SOK{mk_SOK(60, 56, 56, 56, DataType::UINT8, DataType::UINT8, 7, 3)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
           //SOK
            {tc(wls_SOK[ch64], 48186,  "SOK, all tensors are the same")},  //    GT:48186
            {tc(wls_SOK[ch32], 23894,  "SOK, all tensors are the same")},  //    GT:23894
            {tc(wls_SOK[ch16], 21670,  "SOK, all tensors are the same")},  //    GT:21670

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, FLOAT16_kernel_3x3) {
    const int p{1};
    const int k{3};
    // STRIDE=1
    DPUWorkload wl_CLU_top{mk_SOHO(56, 15, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    DPUWorkload wl_CLU_mid{mk_SOHO(56, 16, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    DPUWorkload wl_CLU_btm{mk_SOHO(56, 15, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
        //SOHO 
            {tc(wls_CLU_top[ch64], 12143,  "top")},  //    GT:12143
            {tc(wls_CLU_top[ch32], 6194,  "top")},  //    GT:6194
            {tc(wls_CLU_top[ch16], 3070,  "top")},  //    GT:3070
                                                     
            {tc(wls_CLU_mid[ch64], 12580,  "middle")},  //    GT:12580
            {tc(wls_CLU_mid[ch32], 6427,  "middle")},  //    GT:6427
            {tc(wls_CLU_mid[ch16], 3203 ,  "middle")},  //    GT:3203 
                                                     
            {tc(wls_CLU_btm[ch64], 12326,  "bottom")},  //    GT:12326
            {tc(wls_CLU_btm[ch32], 6426,  "bottom")},  //    GT:6426
            {tc(wls_CLU_btm[ch16], 3188,  "bottom")},  //    GT:3188


           //SOK
            {tc(wls_SOK[ch64], 41897,  "SOK, all tensors are the same")},    //    GT:41897
            {tc(wls_SOK[ch32], 21441,    "SOK, all tensors are the same")},  //    GT:21441
            {tc(wls_SOK[ch16], 10472,    "SOK, all tensors are the same")},  //    GT:10472

            // clang-format on
    };

    executeTests(tests);

    //****************************************************************************************************************************************************8
    // STRIDE=2
    DPUWorkload wl_CLU_top_stride2{mk_SOHO(112, 28, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    DPUWorkload wl_CLU_mid_stride2{mk_SOHO(112, 29, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};
    DPUWorkload wl_CLU_btm_stride2{mk_SOHO(112, 28, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k, 2)};

    DPUWorkload wl_SOK_stride2{mk_SOK(112, 112, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
        //SOHO
            {tc(wls_CLU_top_stride2[ch64], 12190,  "top")},  //   GT:12190
            {tc(wls_CLU_top_stride2[ch32], 6218 ,  "top")},  //   GT:6218  
            {tc(wls_CLU_top_stride2[ch16], 3346 ,  "top")},  //   GT:3346    
                                                                                                             
            {tc(wls_CLU_mid_stride2[ch64], 12635,  "middle")},  //    GT:12635
            {tc(wls_CLU_mid_stride2[ch32], 6468 ,  "middle")},  //    GT:6468  
            {tc(wls_CLU_mid_stride2[ch16], 3394 ,  "middle")},  //    GT:3394  
                                                                                                                  
            {tc(wls_CLU_btm_stride2[ch64], 12640,  "bottom")},   //    GT:12640
            {tc(wls_CLU_btm_stride2[ch32], 6461 ,  "bottom")},  //    GT:6461  
            {tc(wls_CLU_btm_stride2[ch16], 3395  ,  "bottom")},  //    GT:3395  
                                                                                                                                         

         //SOK
          //  {tc(wls_SOK_stride2[ch64], no_gt,  "SOK, all tensors are the same")},  // ERROR_INPUT_TOO_BIG 
            {tc(wls_SOK_stride2[ch32], 21655,  "SOK, all tensors are the same")},  //    GT:21655
            {tc(wls_SOK_stride2[ch16], 11220,  "SOK, all tensors are the same")},  //    GT:11220

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_5x5) {
    const int p{2};
    const int k{5};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 16, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 18, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 16, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO     
             {tc(wls_CLU_top[ch64], 15880, "top")},    //   GT:15880
             {tc(wls_CLU_top[ch32], 7928,  "top")},     //  GT:7928
             {tc(wls_CLU_top[ch16], 7384,  "top")},     //  GT:7384
                                                             
             {tc(wls_CLU_mid[ch64], 16592, "middle")},     //    GT:16592
             {tc(wls_CLU_mid[ch32], 8198, "middle")},      //    GT:8198
             {tc(wls_CLU_mid[ch16], 7466, "middle")},      //    GT:7466
                                                            
             {tc(wls_CLU_btm[ch64], 16106, "bottom")},      //   GT:16106
             {tc(wls_CLU_btm[ch32], 8063, "bottom")},      //   GT:8063
             {tc(wls_CLU_btm[ch16], 7381, "bottom")},      //   GT:7381

            //SOK
            {tc(wls_SOK[ch64], 56996,  "SOK, all tensors are the same")},  //   GT:56996
            {tc(wls_SOK[ch32], 28228,  "SOK, all tensors are the same")},  //   GT:28228
            {tc(wls_SOK[ch16], 25870,  "SOK, all tensors are the same")},  //   GT:25870

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, FLOAT16_kernel_5x5) {
    const int p{2};
    const int k{5};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 16, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 18, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 16, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top[ch64], 31236, "top")},     //   GT:31236
             {tc(wls_CLU_top[ch32], 15903, "top")},     //   GT:15903
             {tc(wls_CLU_top[ch16], 7941, "top")},      //   GT:7941

             {tc(wls_CLU_mid[ch64], 32566, "middle")},      //   GT:32566
             {tc(wls_CLU_mid[ch32], 16679, "middle")},      //   GT:16679
             {tc(wls_CLU_mid[ch16], 8202, "middle")},       //   GT:8202

             {tc(wls_CLU_btm[ch64], 31976, "bottom")},      //   GT:31976
             {tc(wls_CLU_btm[ch32], 16251, "bottom")},      //   GT:16251
             {tc(wls_CLU_btm[ch16], 8114, "bottom")},       //   GT:8114


           //SOK
            {tc(wls_SOK[ch64], 111829,  "SOK, all tensors are the same", 11)},  //   GT:111829
            {tc(wls_SOK[ch32], 57144,  "SOK, all tensors are the same")},  //   GT:57144
            {tc(wls_SOK[ch16], 28155,  "SOK, all tensors are the same")},  //   GT:28155

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_6x6) {
    const int p{2};
    const int k{6};
    const DataType dt{DataType::UINT8};
    const DPUWorkload wl_CLU_top{mk_SOHO(57, 17, 56, 14, p, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(57, 19, 56, 14, 0, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(57, 17, 56, 14, 0, p, p, p, dt, dt, k, k)};  // s

    const DPUWorkload wl_SOK{mk_SOK(57, 57, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top[ch64],  23233, "top")},      //   GT:23233
             {tc(wls_CLU_top[ch32],  11362, "top")},      //   GT:11362
             {tc(wls_CLU_top[ch16],  10565, "top")},      //   GT:10565

             {tc(wls_CLU_mid[ch64], 24156, "middle")},      //   GT:24156
             {tc(wls_CLU_mid[ch32], 11672, "middle")},      //   GT:11672
             {tc(wls_CLU_mid[ch16], 10750, "middle")},      //   GT:10750

             {tc(wls_CLU_btm[ch64], 23673, "bottom")},  //   GT:23673 
             {tc(wls_CLU_btm[ch32], 11588, "bottom")},  //   GT:11588
             {tc(wls_CLU_btm[ch16], 10644, "bottom")},  //   GT:10644 


            //SOK
            {tc(wls_SOK[ch64], 83287,  "SOK, all tensors are the same")},  //   GT:83287
            {tc(wls_SOK[ch32], 40730,  "SOK, all tensors are the same")},  //   GT:40730
            {tc(wls_SOK[ch16], 37423,  "SOK, all tensors are the same")},  //   GT:37423

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, FLOAT16_kernel_6x6) {
    const int p{2};
    const int k{6};
    const DataType dt{DataType::FLOAT16};
    const DPUWorkload wl_CLU_top{mk_SOHO(57, 17, 56, 14, p, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(57, 19, 56, 14, 0, 0, p, p, dt, dt, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(57, 17, 56, 14, 0, p, p, p, dt, dt, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(57, 57, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top[ch64], 44804, "top")},  //      GT:44804
             {tc(wls_CLU_top[ch32], 22812, "top")},  //      GT:22812
             {tc(wls_CLU_top[ch16], 11471, "top")},  //      GT:11471

             {tc(wls_CLU_mid[ch64], 46757, "middle")},  //   GT:46757
             {tc(wls_CLU_mid[ch32], 23620, "middle")},  //   GT:23620
             {tc(wls_CLU_mid[ch16], 11823, "middle")},  //   GT:11823

             {tc(wls_CLU_btm[ch64], 45896, "bottom")},  //   GT: 45896
             {tc(wls_CLU_btm[ch32], 23184, "bottom")},  //   GT: 23184
             {tc(wls_CLU_btm[ch16], 11657, "bottom")},  //   GT: 11657

             //SOK
            {tc(wls_SOK[ch64], 160682,  "SOK, all tensors are the same")}, //  GT:160682
            {tc(wls_SOK[ch32], 82157,  "SOK, all tensors are the same")},  //  GT:82157
            {tc(wls_SOK[ch16], 40532,  "SOK, all tensors are the same")},  //  GT:40532

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_7x7) {
    const int p{3};
    const int k{7};
    // STRIDE=1
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 17, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 20, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 17, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO
             {tc(wls_CLU_top[ch64], 30375,  "top")},  //   GT:30375
             {tc(wls_CLU_top[ch32], 15191,  "top")},  //   GT:15191
             {tc(wls_CLU_top[ch16], 14314,  "top")},  //   GT:14314
                                                       
             {tc(wls_CLU_mid[ch64], 31570,  "middle")},  //   GT:31570
             {tc(wls_CLU_mid[ch32], 15787,  "middle")},  //   GT:15787
             {tc(wls_CLU_mid[ch16], 14546,  "middle")},  //   GT:14546
                                                     
             {tc(wls_CLU_btm[ch64], 30602,  "bottom")},  //   GT:30602
             {tc(wls_CLU_btm[ch32], 15363,  "bottom")},  //   GT:15363
             {tc(wls_CLU_btm[ch16], 14310,  "bottom")},  //   GT:14310
                                                            

            //SOK
            {tc(wls_SOK[ch64], 110502,  "SOK, all tensors are the same")},  //   GT:110502
            {tc(wls_SOK[ch32], 54629,  "SOK, all tensors are the same")},   //   GT:54629
            {tc(wls_SOK[ch16], 50701,  "SOK, all tensors are the same")},   //   GT:50701

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

    const DPUWorkload wl_SOK_stride2{mk_SOK(56 * 2, 56 * 2, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top_stride2[ch64], 31117, "top")},  //   GT:31117
             {tc(wls_CLU_top_stride2[ch32], 16939, "top")},  //   GT:16939  
             {tc(wls_CLU_top_stride2[ch16], 16886, "top")},  //   GT:16886
                                                                                                         
             {tc(wls_CLU_mid_stride2[ch64], 32280, "middle")},  //   GT:32280
             {tc(wls_CLU_mid_stride2[ch32], 17218, "middle")},  //   GT:17218 
             {tc(wls_CLU_mid_stride2[ch16], 17094, "middle")},  //   GT:17094
                                                                
             {tc(wls_CLU_btm_stride2[ch64], 32238, "bottom")},  //   GT:32238
             {tc(wls_CLU_btm_stride2[ch32], 17217, "bottom")},  //   GT:17217
             {tc(wls_CLU_btm_stride2[ch16], 17093, "bottom")},  //   GT:17093


             //SOK
            {tc(wls_SOK_stride2[ch64], 112578,  "SOK, all tensors are the same")},  //    GT:112578
            {tc(wls_SOK_stride2[ch32], 59945,  "SOK, all tensors are the same")},  //    GT:59945
            {tc(wls_SOK_stride2[ch16], 59614,  "SOK, all tensors are the same")},  //    GT:59614

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, FLOAT16_kernel_7x7) {
    const int p{3};
    const int k{7};
    // STRIDE=1
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 17, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 20, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 17, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top[ch64], 59688, "top")},  //  GT:59688
             {tc(wls_CLU_top[ch32], 30427, "top")},  //  GT:30427
             {tc(wls_CLU_top[ch16], 14144, "top")},  //  GT:14144
                                                      
             {tc(wls_CLU_mid[ch64], 62012, "middle")},  //    GT:62012
             {tc(wls_CLU_mid[ch32], 32027, "middle")},  //    GT:32027
             {tc(wls_CLU_mid[ch16], 15680, "middle")},  //    GT:15680
                                                 
             {tc(wls_CLU_btm[ch64], 60407, "bottom")},  //    GT:60407
             {tc(wls_CLU_btm[ch32], 30853, "bottom")},  //    GT:30853
             {tc(wls_CLU_btm[ch16], 15330, "bottom")},  //    GT:15330

            //SOK
            {tc(wls_SOK[ch64], 214272,  "SOK, all tensors are the same")},  //  GT:214272
            {tc(wls_SOK[ch32], 110296, "SOK, all tensors are the same")},  //   GT:110296
            {tc(wls_SOK[ch16], 54414, "SOK, all tensors are the same")},  //   GT:54414

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

    const DPUWorkload wl_SOK_stride2{mk_SOK(56 * 2, 56 * 2, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p, 2)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top_stride2{change_wl_channels_64_32_16(wl_CLU_top_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_mid_stride2{change_wl_channels_64_32_16(wl_CLU_mid_stride2)};
    std::array<DPUWorkload, 3> wls_CLU_btm_stride2{change_wl_channels_64_32_16(wl_CLU_btm_stride2)};
    std::array<DPUWorkload, 3> wls_SOK_stride2{change_wl_channels_64_32_16(wl_SOK_stride2)};

    const std::vector<GTestCase> tests_stride2{
            // clang-format off
         //SOHO 
             {tc(wls_CLU_top_stride2[ch64], 61139,  "top")},  //   GT:61139
             {tc(wls_CLU_top_stride2[ch32], 31238,  "top")},  //   GT:31238
             {tc(wls_CLU_top_stride2[ch16], 17010,  "top")},  //   GT:17010
                                                                                                       
             {tc(wls_CLU_mid_stride2[ch64], 63175, "middle")},  //  GT:63175
             {tc(wls_CLU_mid_stride2[ch32], 32807, "middle")},  //  GT:32807
             {tc(wls_CLU_mid_stride2[ch16], 17273, "middle")},  //  GT:17273
                                                                   
             {tc(wls_CLU_btm_stride2[ch64], 63345,  "bottom")},  //  GT:63345
             {tc(wls_CLU_btm_stride2[ch32], 32807,  "bottom")},  //  GT:32807
             {tc(wls_CLU_btm_stride2[ch16], 17278,  "bottom")},  //  GT:17278

             //SOK
          //  {tc(wls_SOK_stride2[ch64], no_gt,  "SOK, all tensors are the same")},  //  ERROR_INPUT_TOO_BIG ??????
            {tc(wls_SOK_stride2[ch32], 114043,  "SOK, all tensors are the same")},  //    GT:114043
            {tc(wls_SOK_stride2[ch16], 59863,  "SOK, all tensors are the same")},  //    GT:59863

            // clang-format on
    };

    executeTests(tests_stride2);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_9x9) {
    const int p{4};
    const int k{9};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 18, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 22, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 18, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO
             {tc(wls_CLU_top[ch64], 49420, "top")},  //  GT:49420
             {tc(wls_CLU_top[ch32], 24793, "top")},  //  GT:24793
             {tc(wls_CLU_top[ch16], 23279, "top")},  //  GT:23279
                                                      
             {tc(wls_CLU_mid[ch64], 52086, "middle")},  //   GT:52086
             {tc(wls_CLU_mid[ch32], 25713, "middle")},  //   GT:25713
             {tc(wls_CLU_mid[ch16], 23705, "middle")},  //   GT:23705
                                                     
             {tc(wls_CLU_btm[ch64], 49920, "bottom")},      //   GT:49920
             {tc(wls_CLU_btm[ch32], 24956, "bottom", 13)},  //   GT:24956
             {tc(wls_CLU_btm[ch16], 23274, "bottom")},      //   GT:23274

            //SOK
            {tc(wls_SOK[ch64], 178491, "SOK, all tensors are the same")},  //   GT:178491
            {tc(wls_SOK[ch32], 88348, "SOK, all tensors are the same")},  //   GT:88348
            {tc(wls_SOK[ch16], 82538, "SOK, all tensors are the same")},  //   GT:82538

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, FLOAT16_kernel_9x9) {
    const int p{4};
    const int k{9};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 18, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 22, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 18, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top[ch64], 96807, "top")},  //    GT:96807
             {tc(wls_CLU_top[ch32], 49420, "top")},  //    GT:49420
             {tc(wls_CLU_top[ch16], 24668, "top")},  //    GT:24668
                                                    
             {tc(wls_CLU_mid[ch64], 100920, "middle")}, //   GT:100920
             {tc(wls_CLU_mid[ch32], 52086, "middle")},  //    GT:52086
             {tc(wls_CLU_mid[ch16], 25518, "middle")},  //    GT:25518
                                                      
             {tc(wls_CLU_btm[ch64], 97752, "bottom")},  //    GT:97752
             {tc(wls_CLU_btm[ch32], 49920, "bottom")},  //    GT:49920
             {tc(wls_CLU_btm[ch16], 24897, "bottom")},  //    GT:24897

            //SOK
            {tc(wls_SOK[ch64], 347606,  "SOK, all tensors are the same")},  //   GT:347606
            {tc(wls_SOK[ch32], 178765, "SOK, all tensors are the same")},   //   GT:178765
            {tc(wls_SOK[ch16], 88279, "SOK, all tensors are the same")},    //   GT:88279

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, UINT8_kernel_11x11) {
    const int p{5};
    const int k{11};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 19, 56, 14, p, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 24, 56, 14, 0, 0, p, p, DataType::UINT8, DataType::UINT8, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 19, 56, 14, 0, p, p, p, DataType::UINT8, DataType::UINT8, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::UINT8, DataType::UINT8, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO  
             {tc(wls_CLU_top[ch64], 72390, "top")},  //    GT:72390
             {tc(wls_CLU_top[ch32], 36467, "top")},  //    GT:36467
             {tc(wls_CLU_top[ch16], 34673, "top")},  //    GT:34673
                                                       
             {tc(wls_CLU_mid[ch64], 75818, "middle")},  //    GT:75818
             {tc(wls_CLU_mid[ch32], 38112, "middle")},  //    GT:38112
             {tc(wls_CLU_mid[ch16], 35369, "middle")},  //    GT:35369
                                                 
             {tc(wls_CLU_btm[ch64], 72903, "bottom")},  //    GT:72903
             {tc(wls_CLU_btm[ch32], 36944, "bottom")},  //    GT:36944
             {tc(wls_CLU_btm[ch16], 34669, "bottom")},  //    GT:34669

            //SOK
            {tc(wls_SOK[ch64], 263757, "SOK, all tensors are the same")},  //   GT:263757
            {tc(wls_SOK[ch32], 130998, "SOK, all tensors are the same")},  //   GT:130998
            {tc(wls_SOK[ch16], 123055, "SOK, all tensors are the same")},  //   GT:123055

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_DW_CONV_EISXW117314_NPU27, FLOAT16_kernel_11x11) {
    const int p{5};
    const int k{11};
    const DPUWorkload wl_CLU_top{mk_SOHO(56, 19, 56, 14, p, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_mid{mk_SOHO(56, 24, 56, 14, 0, 0, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};
    const DPUWorkload wl_CLU_btm{mk_SOHO(56, 19, 56, 14, 0, p, p, p, DataType::FLOAT16, DataType::FLOAT16, k, k)};

    const DPUWorkload wl_SOK{mk_SOK(56, 56, 56, 56, DataType::FLOAT16, DataType::FLOAT16, k, k, p)};

    // vectors with workloads with channels 64, 32, 16
    std::array<DPUWorkload, 3> wls_CLU_top{change_wl_channels_64_32_16(wl_CLU_top)};
    std::array<DPUWorkload, 3> wls_CLU_mid{change_wl_channels_64_32_16(wl_CLU_mid)};
    std::array<DPUWorkload, 3> wls_CLU_btm{change_wl_channels_64_32_16(wl_CLU_btm)};
    std::array<DPUWorkload, 3> wls_SOK{change_wl_channels_64_32_16(wl_SOK)};

    // no_fail = 0;  // tests will fail
    const std::vector<GTestCase> tests{
            // clang-format off
         //SOHO
             {tc(wls_CLU_top[ch64], 141932, "top")},  //  GT:141932
             {tc(wls_CLU_top[ch32], 72664, "top")},  //  GT:72664
             {tc(wls_CLU_top[ch16], 36250, "top")},  //  GT:36250
                                                       
             {tc(wls_CLU_mid[ch64], 148677, "middle")},  //  GT:148677
             {tc(wls_CLU_mid[ch32], 77197, "middle")},  //  GT:77197
             {tc(wls_CLU_mid[ch16], 37913, "middle")},  //  GT:37913
                                                      
             {tc(wls_CLU_btm[ch64], 143745, "bottom")},  //  GT:143745
             {tc(wls_CLU_btm[ch32], 73573, "bottom")},  //  GT:73573
             {tc(wls_CLU_btm[ch16], 36900, "bottom")},  //  GT:36900


            //SOK
            {tc(wls_SOK[ch64], 512864,  "SOK, all tensors are the same")},  //   GT:512864
            {tc(wls_SOK[ch32], 263896, "SOK, all tensors are the same")},   //   GT:263896
            {tc(wls_SOK[ch16], 130799,  "SOK, all tensors are the same")},  //   GT:130799
            // clang-format on
    };

    executeTests(tests);
}

class Regression_tests_CONV_EISXW_127649_Model_E_NPU27 : public Regression_Tests {
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
                VPUDevice::VPU_2_7,
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
                VPUDevice::VPU_2_7,
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
                2,                                                                  // output_write_tiles
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

    Regression_tests_CONV_EISXW_127649_Model_E_NPU27() {
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

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv4_EXEC_8x16_SPARSITY) {
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
    CLU_broadcast_8x16_T.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv4_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv4_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 2;

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
             {tc(SOHO_conv4s_8x16_T, 7752, "SOHO, top + wt sparsity"   )}, //    GT: 7752
             {tc(SOHO_conv4s_8x16_M, 7756, "SOHO, middle + wt sparsity")}, //    GT: 7756
             {tc(SOHO_conv4s_8x16_B, 3958, "SOHO, bottom + wt sparsity")}, //    GT: 3958

             //conv4 CUBOID_8x16 + broadcast + sparsity
             {tc(CLU_broadcast_s_8x16_T, 8360, "SOHO, top + broadcast + wt sparsity"   )}, //  GT:8360 
             {tc(CLU_broadcast_s_8x16_M, 8359, "SOHO, middle + broadcast + wt sparsity")}, //  GT:8359 
             {tc(CLU_broadcast_s_8x16_B, 4437, "SOHO, bottom + broadcast + wt sparsity")}, //  GT:4437
            // clang-format on
    };

    executeTests(tests);
}
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv4_EXEC_16x16_SPARSITY) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 7, 64, 56, 6, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 8, 64, 56, 6, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 3, 64, 56, 2, 64, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv4_T};
    CLU_broadcast_T.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_M{SOHO_conv4_M};
    CLU_broadcast_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_B{SOHO_conv4_B};
    CLU_broadcast_B.output_write_tiles = 2;

    // execution mode 16x16 + sparsity
    DPUWorkload SOHO_conv4s_T{act_wt_sp(SOHO_conv4_T, sparsity)};
    DPUWorkload SOHO_conv4s_M{act_wt_sp(SOHO_conv4_M, sparsity)};
    DPUWorkload SOHO_conv4s_B{act_wt_sp(SOHO_conv4_B, sparsity)};

    // execution mode 16x16 + broadcast + sparsity
    DPUWorkload CLU_broadcast_s_T{act_wt_sp(CLU_broadcast_T, sparsity)};
    DPUWorkload CLU_broadcast_s_M{act_wt_sp(CLU_broadcast_M, sparsity)};
    DPUWorkload CLU_broadcast_s_B{act_wt_sp(CLU_broadcast_B, sparsity)};
    // EXPECT_TRUE(false);

    //  no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_16x16 + sparsity
             {tc(SOHO_conv4s_T, 8395, "SOHO, top + wt sparsity")},             //GT:8395
             {tc(SOHO_conv4s_M, 8400, "SOHO, middle + wt sparsity")},          //GT:8400
             {tc(SOHO_conv4s_B, 4755, "SOHO, bottom + wt sparsity", pc , 20)}, //GT:4755

             //conv4 CUBOID_16x16 + broadcast + sparsity
             {tc(CLU_broadcast_s_T, 8395, "SOHO, top + broadcast + wt sparsity")},            // GT: 8395
             {tc(CLU_broadcast_s_M, 8400, "SOHO, middle + broadcast + wt sparsity")},         // GT: 8400
             {tc(CLU_broadcast_s_B, 4755, "SOHO, bottom + broadcast + wt sparsity", pc, 18)}, // GT: 4755
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv4_H5_EXEC_16x16_SPARSITY) {
    // execution mode 16x16
    // const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 6, 64, 56, 5, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 7, 64, 56, 5, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 6, 64, 56, 5, 64, 0, 1)};

    // execution mode 16x16 + broadcast
    // DPUWorkload CLU_broadcast_T{SOHO_conv4_T};
    // CLU_broadcast_T.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_M{SOHO_conv4_M};
    CLU_broadcast_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_B{SOHO_conv4_B};
    CLU_broadcast_B.output_write_tiles = 2;

    // execution mode 16x16 + sparsity
    // DPUWorkload SOHO_conv4s_T{act_wt_sp(SOHO_conv4_T, sparsity)};
    DPUWorkload SOHO_conv4s_M{act_wt_sp(SOHO_conv4_M, sparsity)};
    DPUWorkload SOHO_conv4s_B{act_wt_sp(SOHO_conv4_B, sparsity)};

    // execution mode 16x16 + broadcast + sparsity
    //  DPUWorkload CLU_broadcast_s_T{act_wt_sp(CLU_broadcast_T, sparsity)};
    DPUWorkload CLU_broadcast_s_M{act_wt_sp(CLU_broadcast_M, sparsity)};
    DPUWorkload CLU_broadcast_s_B{act_wt_sp(CLU_broadcast_B, sparsity)};

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_16x16 + sparsity
             //{tc(SOHO_conv4s_T, 7366, "SOHO, top + wt sparsity" , pc, 14)},    //  GT:
             {tc(SOHO_conv4s_M, 8399, "SOHO, middle + wt sparsity")},          //  GT:8399
             {tc(SOHO_conv4s_B, 7409, "SOHO, bottom + wt sparsity", pc, 15)},  //  GT:7409

             //conv4 CUBOID_16x16 + broadcast + sparsity
            // {tc(CLU_broadcast_s_T, 7433, "SOHO, top + broadcast + wt sparsity"  , pc, 14)},    // GT:
             {tc(CLU_broadcast_s_M, 8399, "SOHO, middle + broadcast + wt sparsity")},           // GT:8399
             {tc(CLU_broadcast_s_B, 7409, "SOHO, bottom + broadcast + wt sparsity", pc, 16)},   // GT:7409
            // clang-format on

            ///
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv4_H5_EXEC_8x16_SPARSITY) {
    // execution mode 16x16
    // const DPUWorkload SOHO_conv4_T{mk_SOHO(56, 6, 64, 56, 5, 64, 1, 0)};
    const DPUWorkload SOHO_conv4_M{mk_SOHO(56, 7, 64, 56, 5, 64, 0, 0)};
    const DPUWorkload SOHO_conv4_B{mk_SOHO(56, 6, 64, 56, 5, 64, 0, 1)};

    // execution mode 8x16
    // const DPUWorkload SOHO_conv4_8x16_T{mod_execution(SOHO_conv4_T, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_M{mod_execution(SOHO_conv4_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv4_8x16_B{mod_execution(SOHO_conv4_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    // DPUWorkload CLU_broadcast_8x16_T{SOHO_conv4_8x16_T};
    // CLU_broadcast_8x16_T.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_8x16_M{SOHO_conv4_8x16_M};
    CLU_broadcast_8x16_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv4_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 2;

    // execution mode 8x16 + sparsity
    // DPUWorkload SOHO_conv4s_8x16_T{act_wt_sp(SOHO_conv4_8x16_T, sparsity)};
    DPUWorkload SOHO_conv4s_8x16_M{act_wt_sp(SOHO_conv4_8x16_M, sparsity)};
    DPUWorkload SOHO_conv4s_8x16_B{act_wt_sp(SOHO_conv4_8x16_B, sparsity)};

    // execution mode 8x16 + broadcast + sparsity
    // DPUWorkload CLU_broadcast_s_8x16_T{act_wt_sp(CLU_broadcast_8x16_T, sparsity)};
    DPUWorkload CLU_broadcast_s_8x16_M{act_wt_sp(CLU_broadcast_8x16_M, sparsity)};
    DPUWorkload CLU_broadcast_s_8x16_B{act_wt_sp(CLU_broadcast_8x16_B, sparsity)};

    //  no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
              //conv4 CUBOID_8x16 + sparsity
             //{tc(SOHO_conv4s_8x16_T, 7338, "SOHO, top + wt sparsity"   )}, // GT:
             {tc(SOHO_conv4s_8x16_M, 8359, "SOHO, middle + wt sparsity")}, // GT: 8359
             {tc(SOHO_conv4s_8x16_B, 7355, "SOHO, bottom + wt sparsity")}, // GT: 7355

             //conv4 CUBOID_8x16 + broadcast + sparsity
             //{tc(CLU_broadcast_s_8x16_T, 7403, "SOHO, top + broadcast + wt sparsity"   )}, // GT: 
             {tc(CLU_broadcast_s_8x16_M, 8359, "SOHO, middle + broadcast + wt sparsity")}, // GT:  8359
             {tc(CLU_broadcast_s_8x16_B, 7355, "SOHO, bottom + broadcast + wt sparsity")}, // GT:  7355
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv8_EXEC_8x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 4, 96, 28, 3, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 5, 96, 28, 3, 96, 0, 0)};
    // const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    // execution mode 8x16
    const DPUWorkload SOHO_conv8_8x16_T{mod_execution(SOHO_conv8_T, ExecutionMode::CUBOID_8x16)};
    // const DPUWorkload SOHO_conv8_8x16_M{mod_execution(SOHO_conv8_M, ExecutionMode::CUBOID_8x16)};
    // const DPUWorkload SOHO_conv8_8x16_B{mod_execution(SOHO_conv8_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    DPUWorkload CLU_broadcast_8x16_T{SOHO_conv8_8x16_T};
    CLU_broadcast_8x16_T.output_write_tiles = 2;

    // DPUWorkload CLU_broadcast_8x16_M{SOHO_conv8_8x16_M};
    // CLU_broadcast_8x16_M.output_write_tiles = 2;

    // DPUWorkload CLU_broadcast_8x16_B{SOHO_conv8_8x16_B};
    //  CLU_broadcast_8x16_B.output_write_tiles = 2;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
             //SOH conv4 CUBOID_8x16 
             {tc(SOHO_conv8_8x16_T, 4794, "SOHO, top "   )},          // GT:4794
             //{tc(SOHO_conv8_8x16_M, 4746, "SOHO, middle" )},          // GT:
             //{tc(SOHO_conv8_8x16_B, 3850, "SOHO h1, bottom",pc,32 )}, // GT:

              //conv4 CUBOID_8x16 + broadcast
             {tc(CLU_broadcast_8x16_T, 4794, "SOHO, top + broadcast"   )},            //  GT: 4794
             //{tc(CLU_broadcast_8x16_M, 4818, "SOHO, middle + broadcast")},            //  GT: 
             //{tc(CLU_broadcast_8x16_B, 3843, "SOHO h1, bottom + broadcast", pc, 34)}, //  GT:

            // clang-format on
    };

    executeTests(tests);
}

// TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOK_Conv8_EXEC_8x16) {
//     // execution mode 16x16
//     const DPUWorkload SOK_conv8{mk_SOK(28, 16, 96, 28, 16, 16)};
//
//     // execution mode 8x16
//     const DPUWorkload SOK_conv8_8x16{mod_execution(SOK_conv8, ExecutionMode::CUBOID_8x16)};
//
//     const std::vector<GTestCase> tests{
//             // clang-format off
//              {tc(SOK_conv8_8x16, 5162, "SOK, all tensors are the same", pc, 50)}, // GT:
//
//             // clang-format on
//     };
//
//     executeTests(tests);
// }
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv8_EXEC_16x16) {
    // execution mode 16x16
    const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 4, 96, 28, 3, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 5, 96, 28, 3, 96, 0, 0)};
    // const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    // execution mode 16x16 + broadcast
    DPUWorkload CLU_broadcast_T{SOHO_conv8_T};
    CLU_broadcast_T.output_write_tiles = 2;

    //  DPUWorkload CLU_broadcast_M{SOHO_conv8_M};
    // CLU_broadcast_M.output_write_tiles = 2;

    //  DPUWorkload CLU_broadcast_B{SOHO_conv8_B};
    // CLU_broadcast_B.output_write_tiles = 2;

    //  no_fail=false;
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
             {tc(SOHO_conv8_T, 5835, "SOHO, top"   , pc, 20)},    //    GT:5835
            // {tc(SOHO_conv8_M, 4748, "SOHO, middle", pc, 23)},    //    GT:
            // {tc(SOHO_conv8_B, 3918, "SOHO h1, bottom", pc, 45)}, //    GT:

             //conv4 CUBOID_16x16 + broadcast
             {tc(CLU_broadcast_T, 5835, "SOHO, top + broadcast"  , pc, 20)},     // GT:5835
            // {tc(CLU_broadcast_M, 4838, "SOHO, middle + broadcast", pc, 21)},    // GT:
            // {tc(CLU_broadcast_B, 3881, "SOHO h1, bottom + broadcast", pc, 46)}, // GT:

            // clang-format on
    };

    executeTests(tests);
}

// TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOK_Conv8_EXEC_16x16) {
//     const DPUWorkload SOK_conv8{mk_SOK(28, 16, 96, 28, 16, 16)};
//
//      no_fail=false;
//     const std::vector<GTestCase> tests{
//             // clang-format off
//              {tc(SOK_conv8,    3756, "SOK, all tensors are the same", pc, 25)}, // GT:
//
//             // clang-format on
//     };
//
//     executeTests(tests);
// }
TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv8_H2_EXEC_16x16) {
    // execution mode 16x16
    // const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 3, 96, 28, 2, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 4, 96, 28, 2, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 3, 96, 28, 2, 96, 0, 1)};

    // execution mode 16x16 + broadcast
    // DPUWorkload CLU_broadcast_T{SOHO_conv8_T};
    // CLU_broadcast_T.output_write_tiles = 2;

    // DPUWorkload CLU_broadcast_M{SOHO_conv8_M};
    // CLU_broadcast_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_B{SOHO_conv8_B};
    CLU_broadcast_B.output_write_tiles = 2;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
            // {tc(SOHO_conv8_T, 4748, "SOHO, top", pc, 20)},    // GT: 
           //  {tc(SOHO_conv8_M, 4745, "SOHO, middle",pc,22)},   // GT: 
             {tc(SOHO_conv8_B, 5982, "SOHO, bottom", pc, 21)}, // GT: 5982

             //conv4 CUBOID_16x16 + broadcast
            // {tc(CLU_broadcast_T, 4756, "SOHO, top + broadcast" , pc, 20)},    // GT: 
             //{tc(CLU_broadcast_M, 4762, "SOHO, middle + broadcast",pc,21)},    // GT: 
             {tc(CLU_broadcast_B, 5982, "SOHO, bottom + broadcast", pc, 20)},  // GT: 5982

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv8_H1_EXEC_16x16) {
    // execution mode 16x16
    // const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 2, 96, 28, 1, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 3, 96, 28, 1, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    // execution mode 16x16 + broadcast
    // DPUWorkload CLU_broadcast_T{SOHO_conv8_T};
    // CLU_broadcast_T.output_write_tiles = 2;

    // DPUWorkload CLU_broadcast_M{SOHO_conv8_M};
    // CLU_broadcast_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_B{SOHO_conv8_B};
    CLU_broadcast_B.output_write_tiles = 2;

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            //SOH conv4 CUBOID_16x16    
            // {tc(SOHO_conv8_T, 4748, "SOHO, top", pc, 20)},    // GT: 
           //  {tc(SOHO_conv8_M, 4745, "SOHO, middle",pc,22)},   // GT: 
             {tc(SOHO_conv8_B, 5315, "SOHO, bottom", pc, 21)}, // GT: 5315

             //conv4 CUBOID_16x16 + broadcast
            // {tc(CLU_broadcast_T, 4756, "SOHO, top + broadcast" , pc, 20)},    // GT: 
             //{tc(CLU_broadcast_M, 4762, "SOHO, middle + broadcast",pc,21)},    // GT: 
             {tc(CLU_broadcast_B, 5206, "SOHO, bottom + broadcast", pc, 20)},  // GT: 5206

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv8_H2_EXEC_8x16) {
    // execution mode 16x16
    // const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 3, 96, 28, 2, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 4, 96, 28, 2, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 3, 96, 28, 2, 96, 0, 1)};

    // execution mode 8x16
    // const DPUWorkload SOHO_conv8_8x16_T{mod_execution(SOHO_conv8_T, ExecutionMode::CUBOID_8x16)};
    // const DPUWorkload SOHO_conv8_8x16_M{mod_execution(SOHO_conv8_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv8_8x16_B{mod_execution(SOHO_conv8_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    // DPUWorkload CLU_broadcast_8x16_T{SOHO_conv8_8x16_T};
    // CLU_broadcast_8x16_T.output_write_tiles = 2;

    // DPUWorkload CLU_broadcast_8x16_M{SOHO_conv8_8x16_M};
    // CLU_broadcast_8x16_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv8_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 2;

    // no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
             //SOH conv4 CUBOID_8x16 
            // {tc(SOHO_conv8_8x16_T, 4747, "SOHO, top"   )},    //  GT:
            // {tc(SOHO_conv8_8x16_M, 4746, "SOHO, middle")},    //  GT:
             {tc(SOHO_conv8_8x16_B, 4791, "SOHO, bottom")},    //  GT:4791

              //conv4 CUBOID_8x16 + broadcast
            // {tc(CLU_broadcast_8x16_T, 4816, "SOHO, top + broadcast" )},   // GT:
             //{tc(CLU_broadcast_8x16_M, 4781, "SOHO, middle + broadcast")}, // GT:
             {tc(CLU_broadcast_8x16_B, 4791, "SOHO, bottom + broadcast")}, // GT:4791

            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127649_Model_E_NPU27, SOHO_Conv8_H1_EXEC_8x16) {
    // execution mode 16x16
    // const DPUWorkload SOHO_conv8_T{mk_SOHO(28, 2, 96, 28, 1, 96, 1, 0)};
    // const DPUWorkload SOHO_conv8_M{mk_SOHO(28, 3, 96, 28, 1, 96, 0, 0)};
    const DPUWorkload SOHO_conv8_B{mk_SOHO(28, 2, 96, 28, 1, 96, 0, 1)};

    // execution mode 8x16
    // const DPUWorkload SOHO_conv8_8x16_T{mod_execution(SOHO_conv8_T, ExecutionMode::CUBOID_8x16)};
    // const DPUWorkload SOHO_conv8_8x16_M{mod_execution(SOHO_conv8_M, ExecutionMode::CUBOID_8x16)};
    const DPUWorkload SOHO_conv8_8x16_B{mod_execution(SOHO_conv8_B, ExecutionMode::CUBOID_8x16)};

    // execution mode 8x16 + broadcast
    // DPUWorkload CLU_broadcast_8x16_T{SOHO_conv8_8x16_T};
    // CLU_broadcast_8x16_T.output_write_tiles = 2;

    // DPUWorkload CLU_broadcast_8x16_M{SOHO_conv8_8x16_M};
    // CLU_broadcast_8x16_M.output_write_tiles = 2;

    DPUWorkload CLU_broadcast_8x16_B{SOHO_conv8_8x16_B};
    CLU_broadcast_8x16_B.output_write_tiles = 2;

    //  no_fail = false;
    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            // clang-format off
             //SOH conv4 CUBOID_8x16 
            // {tc(SOHO_conv8_8x16_T, 4747, "SOHO, top"   )},    //  GT:
            // {tc(SOHO_conv8_8x16_M, 4746, "SOHO, middle")},    //  GT:
             {tc(SOHO_conv8_8x16_B, 4259, "SOHO, bottom H1", pc, 20)},    //  GT:4259

              //conv4 CUBOID_8x16 + broadcast
            // {tc(CLU_broadcast_8x16_T, 4816, "SOHO, top + broadcast" )},   // GT:
             //{tc(CLU_broadcast_8x16_M, 4781, "SOHO, middle + broadcast")}, // GT:
             {tc(CLU_broadcast_8x16_B, 4259, "SOHO, bottom H1 + broadcast",pc,21)}, // GT:4259 (but no Brdcst)

            // clang-format on
    };

    executeTests(tests);
}

class Regression_tests_CONV_EISXW_127644_Model_N_NPU27 : public Regression_tests_CONV_EISXW_127649_Model_E_NPU27 {
protected:
    static DPUWorkload mk_SOK(unsigned int in_c, unsigned int out_c) {
        return DPUWorkload{
                VPUDevice::VPU_2_7,
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
                2,                                                            // output_write_tiles
                {0, 0, 0, 0},                                                 // offsets
                ISIStrategy::SPLIT_OVER_K,                                    // isi_strategy
                false,                                                        //
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

    Regression_tests_CONV_EISXW_127644_Model_N_NPU27() {
    }
};

TEST_F(Regression_tests_CONV_EISXW_127644_Model_N_NPU27, Wl_in_ch_160_test_SOHO) {
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
             {tc(SOHO_8x16_T, 7857, "SOHO, top"   )}, //   GT: 7857
             {tc(SOHO_8x16_M, 7895, "SOHO, middle")}, //   GT: 7895
             {tc(SOHO_8x16_B, 7876, "SOHO, bottom")}, //   GT: 7876

             {tc(SOHO_halfk_8x16_T, 3845, "SOHO, top"   )}, //   GT: 3845
             {tc(SOHO_halfk_8x16_M, 3845, "SOHO, middle")}, //   GT: 3845
             {tc(SOHO_halfk_8x16_B, 3845, "SOHO, bottom")}, //   GT: 3845
            // clang-format on
    };

    executeTests(tests);
}

TEST_F(Regression_tests_CONV_EISXW_127644_Model_N_NPU27, Wl_in_ch_160_test_SOK) {
    const DPUWorkload SOK_t123_ch48{mk_SOK(160, 48)};
    const DPUWorkload SOK_t4_ch16{mk_SOK(160, 16)};

    const DPUWorkload SOK_ch32{mk_SOK(160, 32)};
    const DPUWorkload SOK_ch64{mk_SOK(160, 64)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            {tc(SOK_t4_ch16,   4612,  "SOK,K16", pc, 30)},    //  GT:  4612 
            {tc(SOK_ch32,      6149,  "SOK, K 32")},          //  GT:  6149 
            {tc(SOK_t123_ch48, 9050,  "SOK, K48", 11)},       //  GT:  9050 
            {tc(SOK_ch64,      12029, "SOK, K 32")},          //  GT:  12029
            // clang-format on
    };

    executeTests(tests);
}

class Regression_EISXW_127594_Marina_M_10Th_june_ELTWISE_NPU27 : public Regression_Tests {
protected:
    const DPUWorkload wl_h7_8x16{
            VPUDevice::VPU_2_7,
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
            VPUDevice::VPU_2_7,
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

    Regression_EISXW_127594_Marina_M_10Th_june_ELTWISE_NPU27() {
        show_wl_info = false;
    }
};

TEST_F(Regression_EISXW_127594_Marina_M_10Th_june_ELTWISE_NPU27, ELTWISE_SOHO_H27_Test) {
    const DPUWorkload wl_h7_16x16{mod_execution(wl_h7_8x16, ExecutionMode::CUBOID_16x16)};
    const DPUWorkload wl_h6_16x16{mod_execution(wl_h6_8x16, ExecutionMode::CUBOID_16x16)};

    const DPUWorkload wl_h7_4x16{mod_execution(wl_h7_8x16, ExecutionMode::CUBOID_4x16)};
    const DPUWorkload wl_h6_4x16{mod_execution(wl_h6_8x16, ExecutionMode::CUBOID_4x16)};

    // no_fail = false;
    const std::vector<GTestCase> tests{
            // clang-format off
            {tc(wl_h7_16x16, 3231,   "SOHO ")},    // GT:3231
            {tc(wl_h6_16x16, 3154,   "SOHO ")},    // GT:3154

            //only 8x16 is allowed at splitting fro element-wise
            {tc(wl_h7_8x16, 3240,   "SOHO (real situation) ", pc )},   // GT:3240
            {tc(wl_h6_8x16, 3158,   "SOHO (real situation)", pc+1)},    // GT:3158

            {tc(wl_h7_4x16, 3467,   "SOHO ", 20)},    // GT:3467
            {tc(wl_h6_4x16, 3466,   "SOHO ", 20)},    // GT:3466
            // clang-format on
    };

    executeTests(tests);
}
}  // namespace VPUNN_unit_tests