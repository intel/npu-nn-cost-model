#include "vpu/shave/VPUEM_piecewise_calc_subblk_size.h"   

#include <gtest/gtest.h>
#include "vpu/types.h"

#include <fstream>
#include <iostream>
#include <vector>


namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUEMPiecewiseCalcSUbblkTest : public ::testing::Test {
public:
    VPUEMCalcSubblk vpuemCalculator = VPUEMCalcSubblk(true, 32, 128);
protected:

    const std::vector<SHAVEWorkload> test_workloads{{
            "sigmoid",
            VPUDevice::VPU_2_7,
            {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
            {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
    },

    {"sigmoid",
      VPUDevice::VPU_2_7,
      {VPUTensor(9, 32, 9, 1, DataType::FLOAT16, Layout::ZXY)},
      {VPUTensor(9, 32, 9, 1, DataType::FLOAT16, Layout::ZXY)},
    },

    {"sigmoid",
      VPUDevice::VPU_2_7,
      {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
      {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
    },

    {"sigmoid",
      VPUDevice::VPU_2_7,
     {VPUTensor(512, 32, 512, 1, DataType::FLOAT16, Layout::ZXY)},
     {VPUTensor(512, 32, 512, 1, DataType::FLOAT16, Layout::ZXY)},
    },
    {"add",
      VPUDevice::VPU_2_7,
      {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY),
       VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
      {VPUTensor(8, 32, 8, 1, DataType::FLOAT16, Layout::ZXY)},
    },
    {"add",
      VPUDevice::VPU_2_7,
      {VPUTensor(9, 32, 9, 1, DataType::FLOAT16, Layout::ZXY),
       VPUTensor(9, 32, 9, 1, DataType::FLOAT16, Layout::ZXY)},
      {VPUTensor(9, 32, 9, 1, DataType::FLOAT16, Layout::ZXY)},
    },
    {"add",
      VPUDevice::VPU_2_7,
      {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY),
       VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
      {VPUTensor(32, 32, 32, 1, DataType::FLOAT16, Layout::ZXY)},
    },
    {"add",
      VPUDevice::VPU_2_7,
      {VPUTensor(512, 32, 512, 1, DataType::FLOAT16, Layout::ZXY),
       VPUTensor(512, 32, 512, 1, DataType::FLOAT16, Layout::ZXY)},
      {VPUTensor(512, 32, 512, 1, DataType::FLOAT16, Layout::ZXY)},
    }
   };

   

    void SetUp() override {
    }

private:
};

TEST_F(VPUEMPiecewiseCalcSUbblkTest, ReturnsSubblkUnits) {
    std::vector<int> expected_blocks = {16, 21, 128, 32768, 16, 21, 128, 32768};
    for (long unsigned int idx = 0; idx < test_workloads.size(); idx++) {
        auto vpuemTuple =
                vpuemCalculator.calc_dsp_block_unit(test_workloads[idx].get_inputs(), test_workloads[idx].get_outputs());
        EXPECT_EQ(std::get<0>(vpuemTuple), std::list<int>({1, 1, expected_blocks[idx]}));
    }
}


TEST_F(VPUEMPiecewiseCalcSUbblkTest, ReturnsIDim) {
    std::vector<int> expected_idim = {128, 128, 256, 256, 128, 128, 256, 256};
    for (long unsigned int idx = 0; idx < test_workloads.size(); idx++) {
        auto vpuemTuple = vpuemCalculator.calc_dsp_block_unit(test_workloads[idx].get_inputs(),
                                                              test_workloads[idx].get_outputs());
        VPUEM_Subblk_Tensor idim = VPUEM_Subblk_Tensor{{1, 1, expected_idim[idx]}, DataType::FLOAT16, Layout::ZXY};
        EXPECT_EQ(std::get<1>(vpuemTuple).front().get_shape(), idim.get_shape());
    }
}

TEST_F(VPUEMPiecewiseCalcSUbblkTest, ReturnsIDimLast) {
    std::vector<int> expected_idim = {128, 128, 256, 256, 128, 128, 256, 256};
    for (long unsigned int idx = 0; idx < test_workloads.size(); idx++) {
        auto vpuemTuple = vpuemCalculator.calc_dsp_block_unit(test_workloads[idx].get_inputs(),
                                                              test_workloads[idx].get_outputs());
        VPUEM_Subblk_Tensor idim = VPUEM_Subblk_Tensor{{1, 1, expected_idim[idx]}, DataType::FLOAT16, Layout::ZXY};
        EXPECT_EQ(std::get<2>(vpuemTuple).front().get_shape(), idim.get_shape());
    }
}

TEST_F(VPUEMPiecewiseCalcSUbblkTest, ReturnsODim) {
    std::vector<int> expected_odim = {128, 128, 256, 256, 128, 128, 256, 256};
    for (long unsigned int idx = 0; idx < test_workloads.size(); idx++) {
        auto vpuemTuple = vpuemCalculator.calc_dsp_block_unit(test_workloads[idx].get_inputs(),
                                                              test_workloads[idx].get_outputs());
        VPUEM_Subblk_Tensor idim = VPUEM_Subblk_Tensor{{1, 1, expected_odim[idx]}, DataType::FLOAT16, Layout::ZXY};
        EXPECT_EQ(std::get<3>(vpuemTuple).front().get_shape(), idim.get_shape());
    }
}

TEST_F(VPUEMPiecewiseCalcSUbblkTest, ReturnsODimLast) {
    std::vector<int> expected_odim = {128, 128, 256, 256, 128, 128, 256, 256};
    for (long unsigned int idx = 0; idx < test_workloads.size(); idx++) {
        auto vpuemTuple = vpuemCalculator.calc_dsp_block_unit(test_workloads[idx].get_inputs(),
                                                              test_workloads[idx].get_outputs());
        VPUEM_Subblk_Tensor idim = VPUEM_Subblk_Tensor{{1, 1, expected_odim[idx]}, DataType::FLOAT16, Layout::ZXY};
        EXPECT_EQ(std::get<4>(vpuemTuple).front().get_shape(), idim.get_shape());
    }
}

}  // namespace VPUNN_unit_tests