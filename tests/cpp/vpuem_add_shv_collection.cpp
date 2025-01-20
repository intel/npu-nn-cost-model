#include <gtest/gtest.h>
#include "vpu/types.h"
#include "vpu/shave/shave_devices.h"
#include "vpu/shave/VPUEM_cost_function.h"
#include "vpu/shave/VPUEM_models.h"

#include <fstream>
#include <iostream>
#include <string>



namespace VPUNN_unit_tests {
using namespace VPUNN;

struct ShvCollectionTestData {
    std::string name_;
    std::vector<int> expected_cycles_;

    ShvCollectionTestData(const std::string& name, const std::vector<int> & expected_cycles) : name_(name), expected_cycles_(expected_cycles) {}
};

class VPUEMaddShvCollectionTest : public ::testing::TestWithParam<ShvCollectionTestData> {
protected:
    std::vector<SHAVEWorkload> test_vectors = {
        {"test_op",
        VPUDevice::VPU_2_7,
        {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
        {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
        },
        {"test_op",
        VPUDevice::VPU_2_7,
        {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
        {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
        },
        {"test_op",
        VPUDevice::VPU_2_7,
        {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
        {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
        },
        {"test_op",
        VPUDevice::VPU_2_7,
        {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
        {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
        },
        {"test_op",
        VPUDevice::VPU_2_7,
        {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
        {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
        }};

    void SetUp() override {
    }
    const ShaveInstanceHolder_VPU27 ih;


public:
private:
};

TEST_P(VPUEMaddShvCollectionTest, ReturnCyclesTest) {
    const ShvCollectionTestData operation_param = GetParam();
    const DeviceShaveContainer& list = ih.getContainer();
    const auto& shaveOp = list.getShaveExecutor(operation_param.name_);
    for (size_t idx = 0; idx < test_vectors.size(); idx++) {
        
        EXPECT_EQ(shaveOp.dpuCycles(test_vectors[idx], 1400, 1050), operation_param.expected_cycles_[idx]);
    }
}

INSTANTIATE_TEST_SUITE_P(VPUEMaddShvCollectionTests, VPUEMaddShvCollectionTest,
                         ::testing::Values(ShvCollectionTestData{"vpuem.add", {13498, 3202, 5072, 4034, 11210}}));
                                           



}  // namespace VPUNN_unit_tests