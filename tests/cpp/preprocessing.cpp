// Copyright © 2022 Intel Corporation
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

namespace VPUNN_unit_tests {

class TestPreprocessingLatest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl = {VPUNN::VPUDevice::VPU_2_7,
                                   VPUNN::Operation::CONVOLUTION,
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                   //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
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

// Demonstrate basic creation and zize
TEST_F(TestPreprocessingLatest, CreationAndSize) {
    auto pp = VPUNN::PreprocessingLatest<float>();
    EXPECT_EQ(pp.output_size(), 67);

    size_t data_written = 0;
    std::vector<float> result = pp.transform(wl, data_written);
    EXPECT_EQ(data_written, 67);
}

// The latest fast implementation might have also a stable versioned implementation and their outputs should be the same
TEST_F(TestPreprocessingLatest, TestLatestIsEqualToSomeVersion) {
    // Instantiate the preprocessing class
    auto pp = VPUNN::PreprocessingLatest<float>();
    size_t filled = 0;
    std::vector<float> resultL = pp.transform(wl, filled);

    auto pp_equiv = VPUNN::Preprocessing_Interface10<float>();
    EXPECT_EQ(pp.implements_also_interface(), pp_equiv.getInterfaceVersion());

    size_t filled_equiv = 0;
    std::vector<float> result_equiv = pp_equiv.transform(wl, filled_equiv);

    ASSERT_EQ(resultL.size(), result_equiv.size());
    EXPECT_EQ(filled, filled_equiv) << "Actual filled data must match";

    for (size_t i = 0; i < resultL.size(); ++i) {
        EXPECT_EQ(resultL[i], result_equiv[i]) << "!= at elem : " << i;
    }
    // todo:  test should be done on multiple workloads
}

class TestPreprocessing_Interface01 : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    const VPUNN::DPUWorkload wl = {VPUNN::VPUDevice::VPU_2_7,
                                   VPUNN::Operation::CONVOLUTION,
                                   {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                   //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
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
    std::vector<float> result = pp.transform(wl, data_written);
    ASSERT_EQ(data_written, 67);
}

TEST_F(TestPreprocessing_Interface01, TransformBad) {
    auto pp = VPUNN::Preprocessing_Interface01<float>();
    const VPUNN::DPUWorkload wl_b = {VPUNN::VPUDevice::VPU_2_7,
                                     VPUNN::Operation::CONVOLUTION,
                                     {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                     //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
                                     {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                     {3, 3},                                                     // kernels
                                     {1, 1},                                                     // strides
                                     {1, 1},                                                     // padding
                                     VPUNN::ExecutionMode::CUBOID_16x16};

    ASSERT_EQ(pp.output_size(), 71);

    size_t data_written = 0;
    std::vector<float> result = pp.transform(wl_b, data_written);
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
                                   //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
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
    std::vector<float> result = pp.transform(wl, data_written);
    EXPECT_EQ(data_written, 67);
}

}  // namespace VPUNN_unit_tests
