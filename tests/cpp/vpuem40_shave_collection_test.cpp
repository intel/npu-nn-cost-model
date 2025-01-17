#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "vpu/shave/VPUEM_cost_function.h"
#include "vpu/shave/VPUEM_models.h"
#include "vpu/shave/shave_devices.h"
#include "vpu/types.h"



namespace VPUNN_unit_tests {
using namespace VPUNN;

struct ShvCollectionTestData {
    std::string name_;
    std::vector<int> expected_cycles_;

    ShvCollectionTestData(const std::string& name, const std::vector<int> & expected_cycles) : name_(name), expected_cycles_(expected_cycles) {}
};

class VPUEM40ShvCollectionTest : public ::testing::TestWithParam<ShvCollectionTestData> {
protected:
    std::vector<SHAVEWorkload> test_vectors = {{
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY)},
                                               },
                                               {
                                                       "test_op",
                                                       VPUDevice::VPU_4_0,
                                                       {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
                                                       {VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)},
                                               }};

    void SetUp() override {
    }
    const ShaveInstanceHolder_NPU40 ih;

public:
private:
};

TEST_P(VPUEM40ShvCollectionTest, ReturnCyclesTest) {
    const ShvCollectionTestData operation_param = GetParam();

    const DeviceShaveContainer& list = ih.getContainer();
    
    const auto& shaveOp = list.getShaveExecutor(operation_param.name_);
    std::cout << shaveOp.toString() << std::endl;
    for (size_t idx = 0; idx < test_vectors.size(); idx++) {  
        EXPECT_EQ(shaveOp.dpuCycles(test_vectors[idx], 1700, 971), operation_param.expected_cycles_[idx]);
                
    }
}

TEST_P(VPUEM40ShvCollectionTest, ReturnCyclesTestNominalFreq) {
    const ShvCollectionTestData operation_param = GetParam();

    const DeviceShaveContainer& list = ih.getContainer();

    const auto& shaveOp = list.getShaveExecutor(operation_param.name_);
    std::cout << shaveOp.toString() << std::endl;
    for (size_t idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(shaveOp.dpuCycles(test_vectors[idx]), operation_param.expected_cycles_[idx]);
    }
}

INSTANTIATE_TEST_SUITE_P(VPUEM40ShvCollectionTests, VPUEM40ShvCollectionTest,
                         ::testing::Values(ShvCollectionTestData{"vpuem.sigmoid", {7528, 1920, 707, 2382, 6276}}, 
                                           ShvCollectionTestData{"vpuem.add", {7192, 1819, 679, 2253, 5998}},
                                           ShvCollectionTestData{"vpuem.softmax", {7528, 1920, 707, 2382, 6276}},
                                           ShvCollectionTestData{"vpuem.gelu", {38245, 8892, 2641, 11273, 31716}},
                                           ShvCollectionTestData{"vpuem.hswish", {13815, 3354, 1111, 4208, 11485}},
                                           ShvCollectionTestData{"vpuem.log", {7528, 1920, 707, 2382, 6276}},
                                           ShvCollectionTestData{"vpuem.mul", {7192, 1819, 679, 2253, 5998}},
                                           ShvCollectionTestData{"vpuem.swish", {12463, 3104, 1118, 3860, 10383}},
                                           ShvCollectionTestData{"vpuem.tanh", {7528, 1920, 707, 2382, 6276}}
                                           ));

}  // namespace VPUNN_unit_tests