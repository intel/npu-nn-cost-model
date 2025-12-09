#include "vpu/shave/VPUEM_cost_function.h"
#include "vpu/shave/VPUEM_models.h"
#include "vpu/shave/shave_devices.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "vpu/types.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

struct ShvCollectionTestData {
    std::string name_;
    std::vector<int> expected_cycles_;

    ShvCollectionTestData(const std::string& name, const std::vector<int> & expected_cycles) : name_(name), expected_cycles_(expected_cycles) {}
};

class VPUEM50ShvCollectionTest : public ::testing::TestWithParam<ShvCollectionTestData> {
protected:
    std::vector<SHAVEWorkload> test_vectors = {{
                                                       "test_op",
                                                       VPUDevice::NPU_5_0,
                                                       {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::NPU_5_0,
                                                       {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::NPU_5_0,
                                                       {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::NPU_5_0,
                                                       {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::NPU_5_0,
                                                       {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
                                               }};

    void SetUp() override {
    }
    const ShaveInstanceHolder_NPU50 ih;

public:
private:
};

TEST_P(VPUEM50ShvCollectionTest, ReturnCyclesTest) {
    const ShvCollectionTestData operation_param = GetParam();

    const DeviceShaveContainer& list = ih.getContainer();
    
    const auto& shaveOp = list.getShaveExecutor(operation_param.name_);

    for (size_t idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(shaveOp.dpuCycles(test_vectors[idx], 1950, 1114), operation_param.expected_cycles_[idx]);
                
    }
}

TEST_P(VPUEM50ShvCollectionTest, ReturnCyclesNominalFreqTest) {
    const ShvCollectionTestData operation_param = GetParam();

    const DeviceShaveContainer& list = ih.getContainer();

    const auto& shaveOp = list.getShaveExecutor(operation_param.name_);

    for (size_t idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(shaveOp.dpuCycles(test_vectors[idx]), operation_param.expected_cycles_[idx]);
    }
}

INSTANTIATE_TEST_SUITE_P(VPUEM50ShvCollectionTests, VPUEM50ShvCollectionTest,
                         ::testing::Values(ShvCollectionTestData{"vpuem.sigmoid", {7526, 1920, 707, 2382, 6275}},
                                           ShvCollectionTestData{"vpuem.add", {7190, 1818, 679, 2252, 5997}},
                                           ShvCollectionTestData{"vpuem.softmax", {7526, 1920, 707, 2382, 6275}},
                                           ShvCollectionTestData{"vpuem.gelu", {38238, 8890, 2641, 11271, 31711}},
                                           ShvCollectionTestData{"vpuem.hswish", {13812, 3353, 1111, 4208, 11482}},
                                           ShvCollectionTestData{"vpuem.log", {7526, 1920, 707, 2382, 6275}},
                                           ShvCollectionTestData{"vpuem.mul", {7190, 1818, 679, 2252, 5997}},
                                           ShvCollectionTestData{"vpuem.swish", {12461, 3103, 1118, 3859, 10381}},
                                           ShvCollectionTestData{"vpuem.tanh", {7526, 1920, 707, 2382, 6275}}
                                           ));

}  // namespace VPUNN_unit_tests