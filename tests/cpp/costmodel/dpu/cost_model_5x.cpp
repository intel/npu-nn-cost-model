// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "costmodel/cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestCostModelNPU5x : public TestCostModel {
public:
protected:
    const VPUDevice device{VPUDevice::NPU_5_0};
    DPUWorkload wl_conv{device,
                        Operation::CONVOLUTION,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM{device,
                          Operation::CM_CONVOLUTION,
                          {VPUTensor(56, 56, 15, 1, DataType::UINT8)},  // input dimensions
                          {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                          {3, 3},                                       // kernels
                          {1, 1},                                       // strides
                          {1, 1, 1, 1},                                 // padding
                          ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP{device,
                        Operation::MAXPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP{device,
                        Operation::AVEPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv{device,
                           Operation::DW_CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                           {3, 3},                                       // kernels
                           {1, 1},                                       // strides
                           {1, 1, 1, 1},                                 // padding
                           ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT{device,
                       Operation::ELTWISE,
                       {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                       {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                       {1, 1},                                       // kernels
                       {1, 1},                                       // strides
                       {0, 0, 0, 0},                                 // padding
                       ExecutionMode::CUBOID_8x16};

    DPUWorkload wl_ELT_MUL{device,
                           Operation::ELTWISE_MUL,
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                           {1, 1},                                       // kernels
                           {1, 1},                                       // strides
                           {0, 0, 0, 0},                                 // padding
                           ExecutionMode::CUBOID_8x16};

    DPUWorkload wl_LYR_NORM{device,
                            Operation::LAYER_NORM,
                            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                            {1, 1},                                       // kernels
                            {1, 1},                                       // strides
                            {0, 0, 0, 0},                                 // padding
                            ExecutionMode::CUBOID_16x16};

    const std::vector<DPUWorkload> wl_list{wl_conv,    wl_convCM, wl_MAXP,    wl_AVGP,
                                           wl_DW_conv, wl_ELT,    wl_ELT_MUL, wl_LYR_NORM};

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
    const std::vector<DPUWorkload> wl_list_FP{wl_conv_FP,    wl_convCM_FP, wl_MAXP_FP,    wl_AVGP_FP,
                                              wl_DW_conv_FP, wl_ELT_FP,    wl_ELT_MUL_FP, wl_LYR_NORM_FP};

    //   const float w_sparsity_level{0.69f};                                //< to be used for lists
    //   std::vector<DPUWorkload> wl_list_sparse{wl_conv, wl_ELT};           // only supported
    //   std::vector<DPUWorkload> wl_list_FP_sparse{wl_conv_FP, wl_ELT_FP};  // only supported

    void basicTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

private:
};

TEST_F(TestCostModelNPU5x, NPU5_1_Loaded) {
    VPUCostModel crt_model{NPU_5_0_MODEL_PATH};

    // is loaded OK
    EXPECT_TRUE(crt_model.nn_initialized());
}

TEST_F(TestCostModelNPU5x, All_Operations_INT8) {
    VPUCostModel crt_model{NPU_5_0_MODEL_PATH};

    {
        DPUWorkload wl{wl_conv};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list) {
        basicTest(wl, crt_model, "All int8:");
    }
}
TEST_F(TestCostModelNPU5x, All_Operations_FP16) {
    VPUCostModel crt_model{NPU_5_0_MODEL_PATH};

    {
        DPUWorkload wl{wl_conv_FP};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list_FP) {
        basicTest(wl, crt_model, "All FP16:");
    }
}

TEST_F(TestCostModelNPU5x, All_Operations_INT2_Weights) {
    VPUCostModel crt_model{NPU_5_0_MODEL_PATH};

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

TEST_F(TestCostModelNPU5x, FP32_output_basic) {
    VPUCostModel crt_model{NPU_5_0_MODEL_PATH};

    const DPUWorkload wl_ref_less{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::FLOAT32)},             // output dimensions
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
            ExecutionMode::CUBOID_8x16,                                     // execution mode
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

    auto checkSame = [&crt_model](const DPUWorkload& w1, const DPUWorkload& w2, std::string info = "", const int thresh = 0) {
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
            {VPUTensor(21, 21, 512, 1, DataType::FLOAT16)},             // output dimensions
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
            ExecutionMode::CUBOID_8x16,                                     // execution mode
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

/// Test Data Types
TEST_F(TestCostModel, Acceptance_DataTypes_Test) {
    class Builder {
    public:
        static DPUWorkload makeWL_with_new_DataTypes(VPUDevice device, DataType Tin, DataType Tout) {
            return DPUWorkload{
                    device,
                    Operation::CONVOLUTION,
                    {VPUTensor(15, 50, 64, 1, Tin)},   // input dimensions
                    {VPUTensor(15, 50, 64, 1, Tout)},  // output dimensions
                    {1, 1},                            // kernels
                    {1, 1},                            // strides
                    {0, 0, 0, 0},                      // padding
                    ExecutionMode::CUBOID_16x16,       // execution mode
                    ActivationFunction::NONE,          // activation
                    0.F,                               // act sparsity
                    0.F,                               // weight_sparsity
                    {swz_def, swz_def},                // input_swizzling
                    {swz_def},                         // output_swizzling
                    1,                                 // owtiles
                    {0, 0, 0, 0},                      // offsets,
                    ISIStrategy::CLUSTERING,           // isi_strategy
                    false,                             // weight_sparsity_enabled
            };
        }
    };
    using DataTypePairs = std::vector<std::pair<DataType, DataType>>;
    struct TestIn {
        VPUDevice device;
        const std::string& model_path;
        DataTypePairs invalid_dtypes;
        DataTypePairs valid_dtypes;

        std::vector<DPUWorkload> workloads_invalid_dtypes() const {
            return build_wl_dtype(invalid_dtypes);
        }

        std::vector<DPUWorkload> workloads_valid_dtypes() const {
            return build_wl_dtype(valid_dtypes);
        }

    private:
        std::vector<DPUWorkload> build_wl_dtype(const DataTypePairs& pairs) const {
            std::vector<DPUWorkload> wls_dtype;
            for (const std::pair<DataType, DataType>& dtype_pair : pairs) {
                wls_dtype.push_back(Builder::makeWL_with_new_DataTypes(device, dtype_pair.first, dtype_pair.second));
            }
            return wls_dtype;
        }
    };

    using TestVector = std::vector<TestIn>;
    TestVector tst = {{VPUDevice::NPU_5_0,  // device
                       NPU_5_0_MODEL_PATH,  // model_path
                       {                    // invalid dtypes
                        {DataType::INT4, DataType::INT4},
                        {DataType::INT2, DataType::INT2},
                        {DataType::INT1, DataType::INT1}},
                       {// valid dtypes
                        {DataType::FLOAT16, DataType::HF8},
                        {DataType::BFLOAT16, DataType::UINT8}}},
    };

    for (auto& test_in : tst) {
        VPUNN::VPUCostModel model_file{test_in.model_path};
        // invalid data type cases
        for (const auto& wl : test_in.workloads_invalid_dtypes()) {
            auto cycles = model_file.DPU(wl);
            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << "\n" << wl;
        }

        // valid data types cases
        for (const auto& wl : test_in.workloads_valid_dtypes()) {
            auto cycles = model_file.DPU(wl);
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << "\n" << wl;
        }
    }
}

TEST_F(TestCostModel, Basic_NPU50_vs_VPU40_DPU) {
    {  // 50 and 40
        VPUNN::VPUCostModel model_4_0{VPU_4_0_MODEL_PATH};
        EXPECT_TRUE(model_4_0.nn_initialized());
        VPUNN::VPUCostModel model_5_0{NPU_5_0_MODEL_PATH};
        EXPECT_TRUE(model_5_0.nn_initialized());

        // Operation::CONVOLUTION,
        auto cycles_40 = model_4_0.DPU(wl_glob_40);
        auto cycles_50 = model_5_0.DPU(wl_glob_50);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_40));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_50));

        // 50 must be faster, more MACs
        EXPECT_GT(cycles_40, cycles_50) << wl_glob_40 << wl_glob_50;
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        VPUNN::VPUCostModel model_2_7{model_path};
        EXPECT_FALSE(model_2_7.nn_initialized());
        VPUNN::VPUCostModel model_4_0{model_path};
        EXPECT_FALSE(model_4_0.nn_initialized());
        VPUNN::VPUCostModel model_5_0{model_path};
        EXPECT_FALSE(model_5_0.nn_initialized());

        auto cycles_27 = model_2_7.DPU(wl_glob_27);
        auto cycles_40 = model_4_0.DPU(wl_glob_40);
        auto cycles_50 = model_5_0.DPU(wl_glob_50);

        EXPECT_EQ(cycles_27, 3445);  // theoretical, but at 1300MHz
        EXPECT_EQ(cycles_40, 3445);  // theoretical, but at 1700MHz
        EXPECT_EQ(cycles_50, 1723);  // theoretical, but at 1950MHz with 4k macs
    }
}

}  // namespace VPUNN_unit_tests