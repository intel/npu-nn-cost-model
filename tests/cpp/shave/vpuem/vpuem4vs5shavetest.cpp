#include "vpu/shave/VPUEM_cost_function.h"
#include "vpu/shave/VPUEM_models.h"
#include "vpu/shave/shave_devices.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "vpu/types.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

struct ShvNameTestData {
    std::string name_;

    ShvNameTestData(const std::string& name)
            : name_(name){
    }
};

class VPUEM4vs5ShaveTest : public ::testing::TestWithParam<ShvNameTestData> {
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

    const ShaveInstanceHolder_NPU40 ih4;
    const ShaveInstanceHolder_NPU50 ih_npu5;

public:
private:
};

TEST_P(VPUEM4vs5ShaveTest, ReturnCyclesTest) {
    const ShvNameTestData operation_param = GetParam();

    const DeviceShaveContainer& list4 = ih4.getContainer();
    const DeviceShaveContainer& list_npu5 = ih_npu5.getContainer();

    const auto& shaveOp4 = list4.getShaveExecutor(operation_param.name_);
    const auto& shaveOp_npu5 = list_npu5.getShaveExecutor(operation_param.name_);

    for (size_t idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_NEAR(shaveOp4.dpuCycles(test_vectors[idx]), shaveOp_npu5.dpuCycles(test_vectors[idx]),
                    0.001 * std::max(shaveOp4.dpuCycles(test_vectors[idx]), shaveOp_npu5.dpuCycles(test_vectors[idx])));
    }
}



INSTANTIATE_TEST_SUITE_P(VPUEM4vs5ShaveTests, VPUEM4vs5ShaveTest,
                         ::testing::Values(ShvNameTestData{"vpuem.sigmoid"}, ShvNameTestData{"vpuem.add"},
                                           ShvNameTestData{"vpuem.softmax"}, ShvNameTestData{"vpuem.gelu"},
                                           ShvNameTestData{"vpuem.hswish"}, ShvNameTestData{"vpuem.log"},
                                           ShvNameTestData{"vpuem.mul"}, ShvNameTestData{"vpuem.swish"},
                                           ShvNameTestData{"vpuem.tanh"}));

}  // namespace VPUNN_unit_tests