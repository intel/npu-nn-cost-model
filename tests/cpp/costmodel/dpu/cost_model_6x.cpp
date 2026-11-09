// Copyright © 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "costmodel/cost_model.h"
#include "logger/logger.h"

///@todo: do more specified tests for NPU6x, like for NPU5x, currently it's just copy of NPU5x

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

//@todo: need revision for npu6 operations & execution mode
//@todo: should the tests be for same data types, or better to check the ones that are not the same as in npu5?
class TestCostModelNPU6x : public TestCostModel {
public:
protected:
    const VPUDevice device{VPUDevice::NPU_6_0};
    DPUWorkload wl_conv{device,
                        Operation::CONVOLUTION,
                        {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                        {VPUTensor(56, 56, 32, 1, DataType::INT8)},  // output dimensions
                        {3, 3},                                      // kernels
                        {1, 1},                                      // strides
                        {1, 1, 1, 1},                                // padding
                        ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM{device,
                          Operation::CM_CONVOLUTION,
                          {VPUTensor(56, 56, 15, 1, DataType::INT8)},  // input dimensions
                          {VPUTensor(56, 56, 32, 1, DataType::INT8)},  // output dimensions
                          {3, 3},                                      // kernels
                          {1, 1},                                      // strides
                          {1, 1, 1, 1},                                // padding
                          ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP{device,
                        Operation::MAXPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // output dimensions
                        {3, 3},                                      // kernels
                        {1, 1},                                      // strides
                        {1, 1, 1, 1},                                // padding
                        ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP{device,
                        Operation::AVEPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // output dimensions
                        {3, 3},                                      // kernels
                        {1, 1},                                      // strides
                        {1, 1, 1, 1},                                // padding
                        ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv{device,
                           Operation::DW_CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // output dimensions
                           {3, 3},                                      // kernels
                           {1, 1},                                      // strides
                           {1, 1, 1, 1},                                // padding
                           ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT{device,
                       Operation::ELTWISE,
                       {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                       {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // output dimensions
                       {1, 1},                                      // kernels
                       {1, 1},                                      // strides
                       {0, 0, 0, 0},                                // padding
                       ExecutionMode::CUBOID_8x16};

    DPUWorkload wl_ELT_MUL{device,
                           Operation::ELTWISE_MUL,
                           {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // output dimensions
                           {1, 1},                                      // kernels
                           {1, 1},                                      // strides
                           {0, 0, 0, 0},                                // padding
                           ExecutionMode::CUBOID_8x16};

    DPUWorkload wl_LYR_NORM{device,
                            Operation::LAYER_NORM,
                            {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // input dimensions
                            {VPUTensor(56, 56, 16, 1, DataType::INT8)},  // output dimensions
                            {1, 1},                                      // kernels
                            {1, 1},                                      // strides
                            {0, 0, 0, 0},                                // padding
                            ExecutionMode::CUBOID_16x16};

    const std::vector<DPUWorkload> wl_list{wl_conv, wl_convCM, wl_MAXP, wl_AVGP, wl_DW_conv, wl_ELT};  // only supported

    DPUWorkload wl_conv_FP{device,
                           Operation::CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 32, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM_FP{device,
                             Operation::CM_CONVOLUTION,
                             {VPUTensor(56, 56, 15, 1, DataType::FLOAT16)},  // input dimensions
                             {VPUTensor(56, 56, 32, 1, DataType::FLOAT16)},  // output dimensions
                             {3, 3},                                         // kernels
                             {1, 1},                                         // strides
                             {1, 1, 1, 1},                                   // padding
                             ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP_FP{device,
                           Operation::MAXPOOL,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP_FP{device,
                           Operation::AVEPOOL,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv_FP{device,
                              Operation::DW_CONVOLUTION,
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                              {3, 3},                                         // kernels
                              {1, 1},                                         // strides
                              {1, 1, 1, 1},                                   // padding
                              ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT_FP{device,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                          {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                          {1, 1},                                         // kernels
                          {1, 1},                                         // strides
                          {0, 0, 0, 0},                                   // padding
                          ExecutionMode::CUBOID_8x16};
    DPUWorkload wl_ELT_MUL_FP{device,
                              Operation::ELTWISE_MUL,
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                              {1, 1},                                         // kernels
                              {1, 1},                                         // strides
                              {0, 0, 0, 0},                                   // padding
                              ExecutionMode::CUBOID_8x16};

    DPUWorkload wl_LYR_NORM_FP{device,
                               Operation::LAYER_NORM,
                               {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                               {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                               {1, 1},                                         // kernels
                               {1, 1},                                         // strides
                               {0, 0, 0, 0},                                   // padding
                               ExecutionMode::CUBOID_16x16};
    const std::vector<DPUWorkload> wl_list_FP{wl_conv_FP, wl_convCM_FP,  wl_MAXP_FP,
                                              wl_AVGP_FP, wl_DW_conv_FP, wl_ELT_FP};

    const std::vector<DPUWorkload> wl_list_unsupported_op{wl_ELT_MUL, wl_LYR_NORM, wl_ELT_MUL_FP, wl_LYR_NORM_FP};

    //   const float w_sparsity_level{0.69f};                                //< to be used for lists
    //   std::vector<DPUWorkload> wl_list_sparse{wl_conv, wl_ELT};           // only supported
    //   std::vector<DPUWorkload> wl_list_FP_sparse{wl_conv_FP, wl_ELT_FP};  // only supported

    void basicTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        CyclesInterfaceType cost_cyc{crt_model.DPU(wl, errInfo)};
        //  ASSERT_NO_THROW(cost_cyc) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;

        utest_log::debug()
                << details
                << "-X-------------------------------------------------------------------------------------------"
                   "-\n";
    }

    ///  test for unsupported operations
    void basicTest_unsupported(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;

        utest_log::debug()
                << details
                << "-X-------------------------------------------------------------------------------------------"
                   "-\n";
    }

private:
};

TEST_F(TestCostModelNPU6x, NPU6_Loaded) {
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    // is loaded OK
    EXPECT_TRUE(crt_model.nn_initialized());
}

TEST_F(TestCostModelNPU6x, All_Operations_INT16) {
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    {
        DPUWorkload wl{wl_conv};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list) {
        basicTest(wl, crt_model, "All int8:");
    }
}

TEST_F(TestCostModelNPU6x, All_Operations_FP16) {
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    {
        DPUWorkload wl{wl_conv_FP};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list_FP) {
        basicTest(wl, crt_model, "All FP16:");
    }
}

TEST_F(TestCostModelNPU6x, Unsupported_Operations) {
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    for (const auto& wl : wl_list_unsupported_op) {
        basicTest_unsupported(wl, crt_model, "Unsupported");
    }
}

TEST_F(TestCostModelNPU6x, All_Operations_INT2_Weights) {
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    {
        wl_conv_FP.weight_type = DataType::INT2;
        DPUWorkload wl{wl_conv_FP};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list_FP) {
        auto new_wl = DPUWorkload(wl);
        new_wl.weight_type = DataType::INT2;

        basicTest(new_wl, crt_model, "All FP16w2:");
    }

    for (const auto& wl : wl_list) {
        auto new_wl = DPUWorkload(wl);
        new_wl.weight_type = DataType::INT2;

        basicTest(new_wl, crt_model, "All INT8w2:");
    }
}

TEST_F(TestCostModelNPU6x, FP32_output_basic) {
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    const DPUWorkload wl_ref_less{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::FLOAT32)},                  // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };

    const DPUWorkload wl_ref2{
            device,
            Operation::ELTWISE,
            {VPUTensor(64, 64, 48, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 64, 48, 1, DataType::FLOAT32)},                   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_8x16,                                      // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };

    {
        DPUWorkload wl{wl_ref_less};
        basicTest(wl, crt_model, " COnv should fit in memory.");
    }
    {
        DPUWorkload wl{wl_ref2};
        basicTest(wl, crt_model, "Elemwise should fit into memory.");
    }

    auto checkSame = [&crt_model](const DPUWorkload& w1, const DPUWorkload& w2, std::string info = "",
                                  const int thresh = 0) {
        DPUWorkload wl{w1};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        DPUWorkload wl_2{w2};

        unsigned cost_cyc2{};
        ASSERT_NO_THROW(cost_cyc2 = crt_model.DPU(wl_2, errInfo)) << info << wl_2;

        EXPECT_NEAR(cost_cyc, cost_cyc2, thresh) << info << wl << wl_2;
    };

    auto checkLess = [&crt_model](const DPUWorkload& w1, const DPUWorkload& w2, std::string info = "") {
        DPUWorkload wl{w1};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        DPUWorkload wl_2{w2};

        unsigned cost_cyc2{};
        ASSERT_NO_THROW(cost_cyc2 = crt_model.DPU(wl_2, errInfo)) << info << wl_2;

        EXPECT_LE(cost_cyc, cost_cyc2) << info << wl << wl_2;
    };

    const DPUWorkload wl_ref_less_16{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::FLOAT16)},                  // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };

    const DPUWorkload wl_ref2_16{
            device,
            Operation::ELTWISE,
            {VPUTensor(64, 64, 48, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 64, 48, 1, DataType::FLOAT16)},                   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_8x16,                                      // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };

    {
        checkSame(wl_ref_less, wl_ref_less_16, "Test 1, convs with FP32 and FP16 output should be equal", 500);
        checkLess(wl_ref2_16, wl_ref2, "Test 2, elemnwise with FP16 should be faster than FP32");
    }
}

TEST_F(TestCostModelNPU6x, CONV_Error_Invalid_Input_Config_Investigation_Test) {
    DPUWorkload wl1{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(150, 38, 12, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(150, 38, 80, 1, DataType::UINT8)},  // output dimensions
            {1, 1},                                        // kernels
            {1, 1},                                        // strides
            {0, 0, 0, 0},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::RELU,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},          // input_swizzling
            {Swizzling::KEY_0},                            // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
    };
    wl1.input_autopad = true;
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};
    std::string errInfo;
    unsigned cost_cyc{};
    EXPECT_FALSE(Cycles::isErrorCode(cost_cyc = crt_model.DPU(wl1, errInfo))) << wl1;
}

/// this test verify the differences between the cost of the new NN that have support for DW_CONV with more than 64
/// channels and the extrapolation formula that is used only for situations where NN does not have support
TEST_F(TestCostModelNPU6x, L1_DW_CONV_more_than_64_channels_60NN_vs_extrapolation_test) {
    auto mk_wl = [&](int channels) {
        return DPUWorkload{
                VPUDevice::NPU_6_0,
                Operation::DW_CONVOLUTION,
                {VPUTensor(10, 6, static_cast<unsigned int>(channels), 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(10, 6, static_cast<unsigned int>(channels), 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                                                       // kernels
                {1, 1},                                                                       // strides
                {0, 0, 0, 0},                                                                 // padding
                ExecutionMode::CUBOID_16x16,                                                  // execution mode
                ActivationFunction::NONE,                                                     // activation
                0.0F,                                                                         // act_sparsity
                0.0F,                                                                         // weight_sparsity
                {swz_def, swz_def},                                                           // input_swizzling
                {swz_def},                                                                    // output_swizzling
                1,                                                                            // output_write_tiles
                {0, 0, 0, 0},                                                                 // offsets
                ISIStrategy::CLUSTERING,                                                      // isi_strategy
                false,                                                                        // weight_sparsity_enabled
        };
    };
    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

    struct TestInput {
        DPUWorkload wl;
    };

    struct TestCase {
        TestInput t_in;
    };

    using TestsVector_ = std::vector<TestCase>;

    // tolerance for cost difference between NN and extrapolated
    const unsigned int cost_tolerance = 40;  // cycles

    const DPUWorkload wl_64ch = mk_wl(64);
    const unsigned int cost_wl64{crt_model.DPU(wl_64ch)};  // 279 cycles
    ASSERT_FALSE(Cycles::isErrorCode(cost_wl64));

    const DPUWorkload wl_32ch = mk_wl(32);
    const unsigned int cost_wl32{crt_model.DPU(wl_32ch)};  // cycles
    ASSERT_FALSE(Cycles::isErrorCode(cost_wl32));

    auto testChOptions = [&](TestsVector_ tests) {
        for (auto& t : tests) {
            std::string errInfo;
            const unsigned int cost_cyc{crt_model.DPU(t.t_in.wl, errInfo)};
            const unsigned int extrapolatedCost = Cycles::extrapolate_cost(t.t_in.wl.inputs[0].channels(), 32 /*ch*/,
                                                                           cost_wl32, 64 /*ch*/, cost_wl64);

            EXPECT_FALSE(Cycles::isErrorCode(cost_cyc)) << " Cost " << cost_cyc << " " << t.t_in.wl;

            // cost from NN should be near to extrapolated cost [ extrapolated_cost - 40, extrapolated_cost + 40 ]
            EXPECT_GE(cost_cyc, extrapolatedCost - cost_tolerance) << t.t_in.wl << " Cost " << cost_cyc;
            EXPECT_LE(cost_cyc, extrapolatedCost + cost_tolerance) << t.t_in.wl << " Cost " << cost_cyc;
        }
    };

    // clang-format off
    TestsVector_ tests = {
//      channels   
        {{mk_wl(96) }}, // NN 419 , Extrapolated 434 , diff 15 , diff percent 3.4%
        {{mk_wl(128)}}, // NN 504 , Extrapolated 524 , diff 20 , diff percent 3.8%
        {{mk_wl(160)}}, // NN 584 , Extrapolated 614 , diff 30 , diff percent 4.8%
        {{mk_wl(192)}}, // NN 669 , Extrapolated 704 , diff 35 , diff percent 4.9%
        {{mk_wl(224)}}, // NN 768 , Extrapolated 794 , diff 26 , diff percent 3.2%
        {{mk_wl(256)}}, // NN 871 , Extrapolated 884 , diff 13 , diff percent 1.4%
    };
    testChOptions(std::move(tests));
    // clang-format on
}

/// investigation, cost model does not return INVALID_INPUT_CONFIGURATION error for this test, but the cost from NN is
/// very small, cost = 0
/// this test could be very fragile due to cost, because costs<0 are considered invalid, so NN could any time return an
/// error here
TEST_F(TestCostModelNPU6x, MAXPOOL_Error_Invalid_Input_Config_Investigation_Test) {
    DPUWorkload wl1{
            VPUDevice::NPU_6_0,
            Operation::MAXPOOL,
            {VPUTensor(2, 32, 16, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(2, 32, 16, 1, DataType::UINT8, Layout::YXZ)},  // output dimensions
            {1, 1},                                                   // kernels
            {1, 1},                                                   // strides
            {0, 0, 0, 0},                                             // padding
            ExecutionMode::CUBOID_16x16,                              // execution mode
            ActivationFunction::NONE,                                 // activation
            0.0F,                                                     // act_sparsity
            0.0F,                                                     // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                     // input_swizzling
            {Swizzling::KEY_0},                                       // output_swizzling
            1,                                                        // output_write_tiles
            {0, 0, 0, 0},                                             // offsets
            ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                    // weight_sparsity_enabled
    };

    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};
    std::string errInfo;
    unsigned cost_cyc{};
    EXPECT_FALSE(Cycles::isErrorCode(cost_cyc = crt_model.DPU(wl1, errInfo))) << "Cost:" << cost_cyc << std::endl
                                                                              << wl1;  // NN cost 0
}
TEST_F(TestCostModelNPU6x, CM_CONV_supports_all_exec_modes_Test) {
    // tiled workloads
    auto mkWl = [&](const ExecutionMode exec) -> DPUWorkload {
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::CM_CONVOLUTION,
                {VPUTensor(56, 56, 15, 1, DataType::UINT8)},  // input dimensions
                {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                       // kernels
                {1, 1},                                       // strides
                {1, 1, 1, 1},                                 // padding
                exec,                                         // execution mode
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
        return wl;
    };

    struct TestInput {
        DPUWorkload wl;
    };

    struct TestExpectation {
        bool is_error_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector_ = std::vector<TestCase>;

    auto check_if_error_code = [&](const TestsVector_& tests) {
        VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};
        unsigned cost{};
        for (const auto& t : tests) {
            ASSERT_NO_THROW(cost = crt_model.DPU(t.t_in.wl)) << t.t_in.wl;
            EXPECT_EQ(Cycles::isErrorCode(cost), t.t_exp.is_error_expected) << Cycles::toErrorText(cost);
        }
    };

    TestsVector_ tests = {{{{mkWl(ExecutionMode::CUBOID_16x16)}}, {false}},
                          {{{mkWl(ExecutionMode::CUBOID_8x16)}}, {false}},
                          {{{mkWl(ExecutionMode::CUBOID_4x16)}}, {false}}};

    check_if_error_code(tests);
}

TEST_F(TestCostModelNPU6x, NoErrorInLog_Test) {
    // This was originally added to observe the exact error message produced by the cost model
    // for this specific workload, rather than to validate that a particular error code is returned.
    //
    // The wl is expected to exceed CMX for the current NPU6 configuration, which typically triggers
    // ERROR_INPUT_TOO_BIG.
    //
    // If the wl starts fitting in CMX due model/config updates, the failure path can no longer be exercised.

    const DPUWorkload wl{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(130, 86, 256, 1, DataType::UINT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(129, 85, 64, 1, DataType::UINT8, Layout::ZXY, false)},   // output dimensions
            {2, 2},                                                             // kernels
            {1, 1},                                                             // strides
            {0, 0, 0, 0},                                                       // padding
            ExecutionMode::CUBOID_8x16,                                         // execution mode
            ActivationFunction::NONE,                                           // activation
            0.0F,                                                               // act_sparsity
            0.0F,                                                               // weight_sparsity
            {Swizzling::KEY_0, Swizzling::KEY_0},                               // input_swizzling
            {Swizzling::KEY_0},                                                 // output_swizzling
            1,                                                                  // output_write_tiles
            {0, 0, 0, 0},                                                       // offsets
            ISIStrategy::CLUSTERING,                                            // isi_strategy
            false,                                                              // weight_sparsity_enabled
            HaloWorkload{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            SEPModeInfo{true,             // sep_enabled
                        {130, 86, 1, 1},  // sep_w/h/c/b
                        {130, 86, 256, 1},
                        false},

            // weights dtype (CSV: input_1_datatype)
            DataType::BFLOAT16,
            "",     // layer_info
            false,  // weightless_operation / in_place_input1
            false,  // in_place_output_memory
            false,  // superdense_memory
            false,  // input_autopad
            false,  // output_autopad
            CostSourceHint::AUTO,
            ProfilingServiceBackend::__size,
            MPEEngine::SCL,
            false  // reduce_minmax_op
    };

    // Ensure we're still exercising the intended error path scenario (WL exceeds CMX).
    // If not, this test no longer serves its investigative purpose
    VPUNN::DPU_OperationValidator validator;
    const auto mem = validator.compute_wl_memory(wl);
    const auto& config = validator.get_config(wl.device);
    const int available_cmx = config.get_cmx_size(wl.device);

    EXPECT_TRUE(mem.cmx > available_cmx) << "Workload no longer exceeds CMX; skipping. Update test data if needed.\n"
                                         << mem << "Available CMX: " << available_cmx << " B\n";

    VPUCostModel crt_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};
    std::string errInfo;
    const auto cost_cyc = crt_model.DPU(wl, errInfo);

    EXPECT_TRUE(Cycles::isErrorCode(cost_cyc))
            << "Cost:" << cost_cyc << "\n"
            << "DPU error: " << Cycles::toErrorText(cost_cyc) << "\n"
            << errInfo << "\n"
            << wl << "\n";  // cost is 4294967295  - confirmed ERROR_INPUT_TOO_BIG
                            // available memory: 2097152B = 2MB, necessary memory: 4098432B = ~4MB
}

TEST_F(TestCostModelNPU6x, SanitizeWorkload_ConvToCMConversionAndChannelAlignment_Test) {
    /// Tests for VPUCostModel::sanitize_workload, covering:
    ///  - compressConv_replace_by_CM_CONV_VPU27: CONV with IC<16, no input_autopad -> CM_CONVOLUTION
    ///  - channels_preserving_operations_consistency_check: align input channels to output for channel-preserving ops
    VPUCostModel model{VPUNN::VPUDevice::__size, ""};  // no NN needed, sanitize_workload is independent
    SanityReport result;

    // Helper to build a minimal workload for a given operation and in/out channel counts
    auto make_wl = [&](Operation op, unsigned int in_ch, unsigned int out_ch,
                       ExecutionMode exec = ExecutionMode::CUBOID_16x16) {
        DPUWorkload wl{device,
                       op,
                       {VPUTensor(16, 16, in_ch, 1, DataType::UINT8)},
                       {VPUTensor(16, 16, out_ch, 1, DataType::UINT8)},
                       {1, 1},
                       {1, 1},
                       {0, 0, 0, 0},
                       exec};
        return wl;
    };

    {  // CONV IC<16, no autopad -> becomes CM_CONV
        auto wl = make_wl(Operation::CONVOLUTION, 4, 64);
        model.sanitize_workload(wl, result);
        EXPECT_EQ(wl.op, Operation::CM_CONVOLUTION)
                << "CONV with IC<16 and no input_autopad should become CM_CONVOLUTION";
    }
    {  // CONV IC<16, input_autopad=true -> stays CONV
        auto wl = make_wl(Operation::CONVOLUTION, 4, 64);
        wl.input_autopad = true;
        model.sanitize_workload(wl, result);
        EXPECT_EQ(wl.op, Operation::CONVOLUTION) << "CONV with IC<16 but input_autopad=true should remain CONVOLUTION";
    }
    {  // CONV IC>=16 -> stays CONV
        auto wl = make_wl(Operation::CONVOLUTION, 16, 64);
        model.sanitize_workload(wl, result);
        EXPECT_EQ(wl.op, Operation::CONVOLUTION) << "CONV with IC>=16 should remain CONVOLUTION";
    }

    {  // Channel-preserving ops with mismatched in/out channels -> input aligned to output
        const std::vector<Operation> ch_preserving_ops{Operation::ELTWISE, Operation::DW_CONVOLUTION,
                                                       Operation::MAXPOOL, Operation::AVEPOOL};

        for (const auto op : ch_preserving_ops) {
            auto wl = make_wl(op, 30, 64,
                              (op == Operation::ELTWISE) ? ExecutionMode::CUBOID_8x16 : ExecutionMode::CUBOID_16x16);
            const auto original_op = wl.op;
            model.sanitize_workload(wl, result);
            EXPECT_EQ(wl.inputs[0].channels(), 64u) << "Input channels should be aligned to output for "
                                                    << Operation_ToText.at(static_cast<int>(original_op));
        }
    }
    {  // Channel-preserving op with output_autopad=true -> input NOT aligned
        auto wl = make_wl(Operation::ELTWISE, 30, 64, ExecutionMode::CUBOID_8x16);
        wl.output_autopad = true;
        model.sanitize_workload(wl, result);
        EXPECT_EQ(wl.inputs[0].channels(), 30u) << "Input channels should NOT change when output_autopad is true";
    }
    {  // Channel-preserving op with output channels < 16 -> input NOT aligned
        auto wl = make_wl(Operation::MAXPOOL, 8, 8);
        model.sanitize_workload(wl, result);
        EXPECT_EQ(wl.inputs[0].channels(), 8u) << "Input channels should NOT change when output channels < 16";
    }
    {  // Channel-preserving op with matching channels -> no mutation
        auto wl = make_wl(Operation::DW_CONVOLUTION, 32, 32);
        model.sanitize_workload(wl, result);
        EXPECT_EQ(wl.inputs[0].channels(), 32u)
                << "Input channels should remain unchanged when already matching output";
    }
}

TEST_F(TestCostModelNPU6x, DPU_DcimWithSclExecutionModeReturnsConfigurationError) {
    VPUCostModel model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};
    auto wl = wl_conv;
    wl.mpe_engine = MPEEngine::DCIM;
    const auto original_execution_order = wl.execution_order;
    std::string errInfo;

    const auto cycles = model.DPU(wl, errInfo);

    EXPECT_EQ(cycles, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    EXPECT_EQ(wl.execution_order, original_execution_order);
}

TEST_F(TestCostModelNPU6x, DISABLED_Instantiation_time_test) {
    const std::string cache_file_default{
            (std::filesystem::path{NPU_6_0_MODEL_PATH}).replace_extension("cachebin").string()};

    const int iterations = 10;
    {
        std::vector<std::unique_ptr<VPUCostModel>> ptrs;
        ptrs.reserve(iterations);
        const auto t0 = VPUNN::tick();
        for (int i = 1; i <= iterations; ++i) {
            // VPUCostModel crt_model{NPU_6_0_MODEL_PATH, false, 0, 1, ""};
            ptrs.push_back(std::make_unique<VPUCostModel>(VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH, false, i, 1, ""));
        }

        // Total latency in ms
        const auto one_latency = VPUNN::tock(t0);

        EXPECT_LE(one_latency, 0.0) << "NO CACHE: Cost model instantiation took too long: " << one_latency / iterations
                                    << " ms\n";
    }

    {
        std::vector<std::unique_ptr<VPUCostModel>> ptrs;
        ptrs.reserve(iterations);
        const auto t0 = VPUNN::tick();
        for (int i = 1; i <= iterations; ++i) {
            // VPUCostModel crt_model{NPU_6_0_MODEL_PATH, false, 0, 1, cache_file_default};
            ptrs.push_back(std::make_unique<VPUCostModel>(VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH, false, i, 1,
                                                          cache_file_default));
        }

        // Total latency in ms
        const auto one_latency = VPUNN::tock(t0);

        EXPECT_LE(one_latency, 0.0) << "CACHE from file: Cost model instantiation took too long: "
                                    << one_latency / iterations << " ms\n";
    }
    // load to memeory the cache file, to verify that the latency improvement is not only due to file reading time, but
    // also from the fact that NN can load from memory much faster than from file
    {
        const auto cache_buffer = read_a_file(cache_file_default);
        const auto model_buffer = read_a_file(NPU_6_0_MODEL_PATH);

        std::vector<std::unique_ptr<VPUCostModel>> ptrs;
        ptrs.reserve(iterations);
        const auto t0 = VPUNN::tick();
        for (int i = 1; i <= iterations; ++i) {
            // VPUCostModel crt_model{NPU_6_0_MODEL_PATH, false, 0, 1, cache_file_default};
            ptrs.push_back(std::make_unique<VPUCostModel>(VPUDevice::NPU_6_0,
                                                          model_buffer.data(),  // const char* model_data
                                                          model_buffer.size(),  // size_t model_data_length
                                                          false,                // bool copy_model_data
                                                          false,                // bool profile
                                                          i,                    // unsigned int cache_size
                                                          1,                    // unsigned int batch_size
                                                          cache_buffer.data(),  // const char* dpu_cache_data
                                                          cache_buffer.size()   // size_t dpu_cache_data_length
                                                          ));
        }

        // Total latency in ms
        const auto one_latency = VPUNN::tock(t0);

        EXPECT_LE(one_latency, 0.0) << "CACHE from memory: Cost model instantiation took too long: "
                                    << one_latency / iterations << " ms\n";
    }
}

}  // namespace VPUNN_unit_tests
