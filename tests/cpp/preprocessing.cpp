// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "inference/preprocessing.h"
#include <gtest/gtest.h>
#include "vpu/compatibility/types01.h"
#include "vpu/compatibility/types11.h"

namespace VPUNN_unit_tests {

class TestPreprocessingLatest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl = {VPUNN::VPUDevice::VPU_2_7,
                                   VPUNN::Operation::CONVOLUTION,
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                   {3, 3},                                                     // kernels
                                   {1, 1},                                                     // strides
                                   {1, 1},                                                     // padding
                                   VPUNN::ExecutionMode::CUBOID_16x16};

private:
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessingLatest, SingleWLTestPreprocessing) {
    // Instantiate the preprocessing class
    auto pp = VPUNN::PreprocessingLatest<float>();
    // Transform a single workload
    std::vector<float> result = pp.transform(wl);
}

// Demonstrate some basic assertions.
TEST_F(TestPreprocessingLatest, MultipleWLTestPreprocessing) {
    // Instantiate the preprocessing class
    auto pp = VPUNN::PreprocessingLatest<float>();
    // Transform a batch of them
    const std::vector<VPUNN::DPUWorkload> wl_lst = {wl, wl, wl};
    std::vector<float> result = pp.transform(wl_lst);

    for (unsigned int batch_idx = 0; batch_idx < 3; batch_idx++) {
        std::vector<float> batch_result = pp.transform(wl_lst[batch_idx]);

        EXPECT_EQ(batch_result.size() * 3, result.size());
        for (unsigned int idx = 0; idx < batch_result.size(); idx++) {
            EXPECT_EQ(batch_result[idx], result[idx + batch_idx * pp.output_size()]);
        }
    }
}

// Demonstrate basic creation and size
TEST_F(TestPreprocessingLatest, CreationAndSize) {
    auto pp = VPUNN::PreprocessingLatest<float>();
    EXPECT_EQ(pp.output_size(), 102);

    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, 102);
}

// The latest fast implementation might have also a stable versioned implementation and their outputs should be the same
TEST_F(TestPreprocessingLatest, TestLatestIsEqualToSomeVersion) {
    // Instantiate the preprocessing class
    auto pp = VPUNN::PreprocessingLatest<float>();

    // const VPUNN::RuntimeProcessingFactory factory;

    //    auto pp_equiv = VPUNN::Preprocessing_Interface11<float>();
    auto pp_equiv = VPUNN::PreprocessingLatest<float>();  // not equal to any
    // factory.make_preprocessing(pp.implements_also_interface());

    EXPECT_EQ(pp.interface_version(), pp_equiv.interface_version());
    EXPECT_EQ(pp.implements_also_interface(), pp_equiv.interface_version());

    {
        size_t filled = 0;
        std::vector<float> resultL = pp.generate_descriptor(wl, filled);
        size_t filled_equiv = 0;
        std::vector<float> result_equiv = pp_equiv.generate_descriptor(wl, filled_equiv);

        ASSERT_EQ(resultL.size(), result_equiv.size());
        EXPECT_EQ(filled, filled_equiv) << "Actual filled data must match";

        for (size_t i = 0; i < resultL.size(); ++i) {
            EXPECT_EQ(resultL[i], result_equiv[i]) << "!= at elem : " << i;
        }
    }
    // todo:  test should be done on multiple workloads
    {
        const VPUNN::DPUWorkload tst_wl = {
                VPUNN::VPUDevice::VPU_2_7,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::XYZ,
                                  true)},  // input dimensions
                {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::XYZ,
                                  true)},                            // output dimensions
                {3, 3},                                              // kernels
                {1, 1},                                              // strides
                {1, 1},                                              // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                  //
                VPUNN::ActivationFunction::NONE,                     //
                0.7F,                                                // act sparsity
                0.3F,                                                // weight_sparsity
                {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},  // swiz in
                {VPUNN::Swizzling::KEY_0},                           // swiz out
                1,                                                   // owtiles
                {0, 0, 0, 0},                                        // offsets,
                VPUNN::ISIStrategy::SPLIT_OVER_H,                    //
        };

        size_t filled = 0;
        std::vector<float> resultL = pp.generate_descriptor(tst_wl, filled);
        size_t filled_equiv = 0;
        std::vector<float> result_equiv = pp_equiv.generate_descriptor(tst_wl, filled_equiv);

        ASSERT_EQ(resultL.size(), result_equiv.size());
        EXPECT_EQ(filled, filled_equiv) << "Actual filled data must match";

        for (size_t i = 0; i < resultL.size(); ++i) {
            EXPECT_EQ(resultL[i], result_equiv[i]) << "!= at elem : " << i;
        }
    }
}

class TestPreprocessing_Interface01 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl = {VPUNN::VPUDevice::VPU_2_7,
                                   VPUNN::Operation::CONVOLUTION,
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                   {3, 3},                                                     // kernels
                                   {1, 1},                                                     // strides
                                   {1, 1},                                                     // padding
                                   VPUNN::ExecutionMode::CUBOID_16x16};

private:
};
/// Test cases covering the creation of the Preprocesing mode
TEST_F(TestPreprocessing_Interface01, CreationTest) {
    auto pp = VPUNN::Preprocessing_Interface01<float>();

    ASSERT_EQ(pp.output_size(), 71);

    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    ASSERT_EQ(data_written, 67);
}

TEST_F(TestPreprocessing_Interface01, TransformBad) {
    auto pp = VPUNN::Preprocessing_Interface01<float>();
    const VPUNN::DPUWorkload wl_b = {VPUNN::VPUDevice::VPU_2_7,
                                     VPUNN::Operation::CONVOLUTION,
                                     {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                     {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                     {3, 3},                                                     // kernels
                                     {1, 1},                                                     // strides
                                     {1, 1},                                                     // padding
                                     VPUNN::ExecutionMode::CUBOID_16x16};

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
    const VPUNN::DPUWorkload wl = {VPUNN::VPUDevice::VPU_2_7,
                                   VPUNN::Operation::CONVOLUTION,
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                   {3, 3},                                                     // kernels
                                   {1, 1},                                                     // strides
                                   {1, 1},                                                     // padding
                                   VPUNN::ExecutionMode::CUBOID_16x16};

private:
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface10, SingleWLTestPreprocessing) {
    // Instantiate the preprocessing class
    auto pp = VPUNN::Preprocessing_Interface10<float>();
    // Transform a single workload
    std::vector<float> result = pp.transform(wl);
}

// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface10, CreationAndSize) {
    auto pp = VPUNN::Preprocessing_Interface10<float>();
    EXPECT_EQ(pp.output_size(), 67);

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)VPUNN::NNVersions::VERSION_10_ENUMS_SAME);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, 67);
}

class TestPreprocessing_Interface11 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::XYZ, true)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::XYZ, true)},  // output dimensions
            {3, 3},                                                                               // kernels
            {1, 1},                                                                               // strides
            {1, 1},                                                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                                   //
            VPUNN::ActivationFunction::NONE,                                                      //
            0.7F,                                                                                 // act sparsity
            0.3F,                                                                                 // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},                                   // swiz in
            {VPUNN::Swizzling::KEY_0},                                                            // swiz out
            1,                                                                                    // owtiles
            {0, 0, 0, 0},                                                                         // offsets,
            VPUNN::ISIStrategy::SPLIT_OVER_H,                                                     //
    };

private:
};

// Demonstrate some basic assertions.
TEST_F(TestPreprocessing_Interface11, SingleWLTestPreprocessing) {
    // Instantiate the preprocessing class
    auto pp = VPUNN::Preprocessing_Interface11<float>();
    // Transform a single workload
    std::vector<float> result = pp.transform(wl);
    EXPECT_EQ(result.size(), 93);
}

// Demonstrate basic creation and size
TEST_F(TestPreprocessing_Interface11, CreationAndSize) {
    auto pp = VPUNN::Preprocessing_Interface11<float>();
    EXPECT_EQ(pp.output_size(), 93);  // less , should be 91!!!

    EXPECT_EQ(pp.interface_version(), pp.getInterfaceVersion()) << " dynamic and static version must be equal";
    EXPECT_EQ(pp.interface_version(), (int)VPUNN::NNVersions::VERSION_11_VPU27_BETA);
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl, data_written);
    EXPECT_EQ(data_written, 93);  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
}

TEST_F(TestPreprocessing_Interface11, DescriptorContentTest) {
    const VPUNN::DPUWorkload wl2 = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::ZXY, false)},  // input dimensions
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::ZXY, true)},  // output dimensions
            {1, 1},                                                                                 // kernels
            {1, 1},                                                                                 // strides
            {0, 0, 0, 0},                                                                           // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                                                      //
            VPUNN::ActivationFunction::NONE,                                                        //
            0.F,                                                                                    // act sparsity
            0.F,                                                                                    // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},                                     // swiz in
            {VPUNN::Swizzling::KEY_0},                                                              // swiz out
            1,                                                                                      // owtiles
            {0, 0, 0, 0},                                                                           // offsets,
            VPUNN::ISIStrategy::CLUSTERING,                                                         //
    };

    auto pp = VPUNN::Preprocessing_Interface11<float>();
    size_t data_written = 0;
    std::vector<float> result = pp.generate_descriptor(wl2, data_written);
    EXPECT_EQ(data_written, pp.output_size());  // this is the actual filled dimension

    EXPECT_EQ(result.size(), data_written);
    std::cout << "\n descriptor\n";

    for (auto x : result) {
        std::cout << " " << std::lround(x) << " ";
    }
    std::cout << "\n" << wl2 << "\n";

    const int dev_idx = 0;
    EXPECT_EQ(std::lround(result[dev_idx + (int)wl2.device]), 1);  // as long as no enum change in specific interface

    const int op_idx = dev_idx + 4;
    EXPECT_EQ(std::lround(result[op_idx + (int)wl2.op]), 1);  // as long as no enum change in specific interface

    const int in_0_idx = 10;
    const int in_1_idx = in_0_idx + 16;
    const int out_0_idx = in_1_idx + 16;
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

TEST_F(TestPreprocessing_Interface11, DescriptorContentTest_FLOAT_INT) {
    const VPUNN::DPUWorkload wl_int_int = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                            //
            VPUNN::ActivationFunction::NONE,                              //
            0.F,                                                          // act sparsity
            0.F,                                                          // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},           // swiz in
            {VPUNN::Swizzling::KEY_0},                                    // swiz out
            1,                                                            // owtiles
            {0, 0, 0, 0},                                                 // offsets,
            VPUNN::ISIStrategy::CLUSTERING,                               //
    };
    const VPUNN::DPUWorkload wl_float_float = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              //
            VPUNN::ActivationFunction::NONE,                                //
            0.F,                                                            // act sparsity
            0.F,                                                            // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},             // swiz in
            {VPUNN::Swizzling::KEY_0},                                      // swiz out
            1,                                                              // owtiles
            {0, 0, 0, 0},                                                   // offsets,
            VPUNN::ISIStrategy::CLUSTERING,                                 //
    };
    const VPUNN::DPUWorkload wl_float_int = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              //
            VPUNN::ActivationFunction::NONE,                                //
            0.F,                                                            // act sparsity
            0.F,                                                            // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},             // swiz in
            {VPUNN::Swizzling::KEY_0},                                      // swiz out
            1,                                                              // owtiles
            {0, 0, 0, 0},                                                   // offsets,
            VPUNN::ISIStrategy::CLUSTERING,                                 //
    };
    const VPUNN::DPUWorkload wl_int_float = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::UINT8)},    // input dimensions
            {VPUNN::VPUTensor(15, 1972, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              //
            VPUNN::ActivationFunction::NONE,                                //
            0.F,                                                            // act sparsity
            0.F,                                                            // weight_sparsity
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},             // swiz in
            {VPUNN::Swizzling::KEY_0},                                      // swiz out
            1,                                                              // owtiles
            {0, 0, 0, 0},                                                   // offsets,
            VPUNN::ISIStrategy::CLUSTERING,                                 //
    };

    auto show_desc = [](const std::vector<float>& result, std::string prefix) {
        std::cout << prefix;
        for (auto x : result) {
            std::cout << " " << std::lround(x) << " ";
        }
    };

    auto pp = VPUNN::Preprocessing_Interface11<float>();
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

    const int dtype_idx = 4;
    const int in_0_idx = 10;
    const int in_1_idx = in_0_idx + 16;
    const int out_0_idx = in_1_idx + 16;

    {
        auto& r = result_i_i;
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 0]), 1);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 2]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 0]), 1);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 2]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 0]), 1);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 2]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 3]), 0);
    }
    {
        auto& r = result_i_f;
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 0]), 1);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 2]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 0]), 1);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 2]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 0]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 2]), 1);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 3]), 0);
    }
    {
        auto& r = result_f_i;
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 0]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 2]), 1);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 0]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 2]), 1);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 0]), 1);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 2]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 3]), 0);
    }
    {
        auto& r = result_f_f;
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 0]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 2]), 1);
        EXPECT_EQ(std::lround(r[in_0_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 0]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 2]), 1);
        EXPECT_EQ(std::lround(r[in_1_idx + dtype_idx + 3]), 0);

        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 0]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 1]), 0);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 2]), 1);
        EXPECT_EQ(std::lround(r[out_0_idx + dtype_idx + 3]), 0);
    }
}

}  // namespace VPUNN_unit_tests
