#include "vpu/shave/VPUEM_models.h"

#include <gtest/gtest.h>
#include "vpu/types.h"

#include <iostream>
#include <fstream>
#include <vector>

namespace VPUNN_unit_tests{
using namespace VPUNN;

class VPUEMSoftmaxShvTest : public ::testing::Test {
public:
protected:
    std::vector<VPUTensor> test_vectors = {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY),
                                           VPUTensor(15, 32, 15, 1, DataType::FLOAT16, Layout::ZXY),
                                           VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY),
                                           VPUTensor(17, 32, 17, 1, DataType::FLOAT16, Layout::ZXY),
                                           VPUTensor(29, 32, 29, 1, DataType::FLOAT16, Layout::ZXY)};

    const std::vector<CostFunctionSoftmaxDescriptor> costFunctionSoftmaxData = {
            {false, 128,
             {{8,283,57, 158, 245, 29, 46, 120, 244, 26, 44, 119}, 
              {4, 0, 0, 0, 151, 160, 160, 0, 20, 52, 45, 0}}},
            {false, 512,
             {{8, 284, 57, 208, 236, 29, 47, 160, 239, 31, 46, 163},
              {8, 137, 29, 72, 128, 16, 18, 63, 127, 15, 18, 62}}}
    };

    const VPUNN::VPUEMSoftmaxModel vpuem_softmax{VPUNN::DataType::FLOAT16, costFunctionSoftmaxData, 1300, 975};

    const std::vector<CostFunctionSoftmaxDescriptor> costFunctionReduceMinData = {
            {false, 128, {{8, 178, 11, 71, 167, 8, 13, 60, 166, 7, 13, 59}}},
            {false, 512, {{8, 186, 11, 132, 173, 8, 14, 119, 172, 8, 14, 118}}}
    };

    const VPUNN::VPUEMSoftmaxModel vpuem_reducemin{VPUNN::DataType::FLOAT16, costFunctionReduceMinData, 1300, 975};

    const std::vector<CostFunctionSoftmaxDescriptor> costFunctionMVNData = {
            {true, 128, {{8, 252, 68, 100, 266, 41, 32, 114, 0, 0, 0, 0}}},
            {true, 512, {{8, 333, 78, 259, 307, 36, 38, 233, 0, 0, 0, 0}}}};

    const VPUNN::VPUEMSoftmaxModel vpuem_mvn{VPUNN::DataType::FLOAT16, costFunctionMVNData, 1300, 975};

    
    const std::vector<CostFunctionSoftmaxDescriptor> costFunctionMVNFusedData = {
            {true, 128, {{8, 270, 93, 134, 265, 47, 38, 129, 0, 0, 0, 0}}},
            {true, 512, {{8, 2172, 57, 2096, 2151, 22, 27, 2075, 0, 0, 0, 0}}}};

    const VPUNN::VPUEMSoftmaxModel vpuem_mvn_fused{VPUNN::DataType::FLOAT16, costFunctionMVNFusedData, 1300, 975};
   
     const std::vector<CostFunctionSoftmaxDescriptor> costFunctionMVNcwData = {
            {false, 128, {{4, 303, 22, 176, 254, 46, 38, 127, 253, 37, 30, 126}}},
            {false, 512, {{8, 220, 49, 147, 199, 20, 25, 126, 200, 21, 25, 127}}}};

    const VPUNN::VPUEMSoftmaxModel vpuem_mvn_cw{VPUNN::DataType::FLOAT16, costFunctionMVNcwData, 1300, 975};

    void SetUp() override {
    }

private:
};

TEST_F(VPUEMSoftmaxShvTest, ReturnsSoftmaxShaveCycles) {
    
    std::vector<int> expected_cycles = {29467, 6834, 2107, 8658, 24390};
    for (long unsigned int idx = 0; idx <  test_vectors.size(); idx++) {
        EXPECT_EQ(vpuem_softmax.getShaveCycles(1, 1, test_vectors[idx].size()) , expected_cycles[idx]);

    }
}

TEST_F(VPUEMSoftmaxShvTest, ReturnsReduceMinShaveCycles) {
    std::vector<int> expected_cycles = {114795, 75947, 18795, 50699, 146731};
    for (long unsigned int idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(vpuem_reducemin.getShaveCycles(test_vectors[idx].height(),
                                                 test_vectors[idx].height() * test_vectors[idx].width(),
                                                 test_vectors[idx].channels()), expected_cycles[idx]);
    }
}

TEST_F(VPUEMSoftmaxShvTest, ReturnsMVNShaveCycles) {
    std::vector<int> expected_cycles = {35368, 4332, 2428, 9600, 24116};
    for (long unsigned int idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(vpuem_mvn.getShaveCycles(test_vectors[idx].height(),
                                           test_vectors[idx].height() * test_vectors[idx].width(),
                                           test_vectors[idx].channels()),
                  expected_cycles[idx]);
    }
}

TEST_F(VPUEMSoftmaxShvTest, ReturnsMVNFusedShaveCycles) {
    std::vector<int> expected_cycles = {48288, 5850, 3246, 13052, 32902};
    for (long unsigned int idx = 0; idx < test_vectors.size(); idx++) {
        EXPECT_EQ(vpuem_mvn_fused.getShaveCycles(test_vectors[idx].height(),
                                           test_vectors[idx].height() * test_vectors[idx].width(),
                                           test_vectors[idx].channels()),
                  expected_cycles[idx]);
    }
}


}  // namespace VPUNN_unit_tests