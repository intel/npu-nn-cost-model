// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
// #include "vpu_cost_model.h"
#include "dmann_cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDMANNCostModelNPU4x : public TestDMANNCostModel {
public:
protected:
	void SetUp() override {
        TestDMANNCostModel::SetUp();
	}
};

class TestDMA_TH_CostModelNPU4x : public TestDMA_TH_CostModel {
public:
protected:
    DMANNWorkload_NPU40 wl_glob_40{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device

            8192,  // int src_width;
            8192,  // int dst_width;

            0,  // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;

            //
    };
};

TEST_F(TestDMANNCostModelNPU4x, SmokeTestDMA_40) {
    const std::string model_path = VPU_DMA_4_0_MODEL_PATH;
    ASSERT_NO_THROW(DMACostModel<DMANNWorkload_NPU40> x{model_path});
    DMACostModel<DMANNWorkload_NPU40> dma_model(model_path);
    ASSERT_TRUE(dma_model.nn_initialized());

    {
        DMANNWorkload_NPU40 wl{wl_glob_40};
        auto dma_cycles = dma_model.computeCycles(wl);

        // std::cout << wl << Cycles::toErrorText(dma_cycles);
        //  value chosen by hand to detect problems without changing the DMANN
        EXPECT_EQ(dma_cycles, 455 /*@1700MHZ*/) << wl << Cycles::toErrorText(dma_cycles);
    }
    {
        DMANNWorkload_NPU40 wl{
                VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device
                65535,                      // int src_width;
                65535,                      // int dst_width;
                0,                          // int num_dim;
                {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                Num_DMA_Engine::Num_Engine_1,
                MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
        };
        auto dma_cycles = dma_model.computeCycles(wl);

        // std::cout << wl << Cycles::toErrorText(dma_cycles);
        //  value chosen by hand to detect problems without changing the DMANN
        EXPECT_EQ(dma_cycles, 2041 /*@1700MHz*/) << wl << Cycles::toErrorText(dma_cycles);
    }
}

TEST_F(TestDMANNCostModelNPU4x, SerializerTestDMA) {
    const std::string model_path = VPU_DMA_4_0_MODEL_PATH;
    ASSERT_NO_THROW(DMACostModel<DMANNWorkload_NPU40> x{model_path});
    DMACostModel<DMANNWorkload_NPU40> dma_model(model_path);
    ASSERT_TRUE(dma_model.nn_initialized());

    DMANNWorkload_NPU40 wl{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device
            65535,                      // int src_width;
            65535,                      // int dst_width;
            0,                          // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
    };
    auto dma_cycles = dma_model.computeCycles(wl);

    EXPECT_EQ(dma_cycles, 2041 /*@1700MHz*/) << wl << Cycles::toErrorText(dma_cycles);
}

// this test is for post process for DMA interface 02
TEST_F(TestDMANNCostModelNPU4x, DMA_PostProcessing_Test) {
    DMANNWorkload_NPU40 wl{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device
            65535,                      // int src_width;
            65535,                      // int dst_width;
            0,                          // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
    };

    ConvertFromDirectCycleToDPUCyc<DMANNWorkload_NPU40> pp_DirectCycToDPUCyc_converter;

    float nn_output = 222.9F;
    CyclesInterfaceType cyc_direct = pp_DirectCycToDPUCyc_converter.process(wl, nn_output);

    EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cyc_direct)) << "CASE ConvertFromDirectCycleToDPUCyc " << cyc_direct;
    EXPECT_FALSE(pp_DirectCycToDPUCyc_converter.is_NN_value_invalid(nn_output))
            << "CASE ConvertFromDirectCycleToDPUCyc ";
}

TEST_F(TestDMANNCostModelNPU4x, SweepGT_DMATime_40) {
    class TestCase {
    public:
        const DMANNWorkload_NPU40 t_in;  // wl
        CyclesInterfaceType t_exp;       // expected time (GT)
    };

    auto mkwl = [](const int bytes, MemoryLocation in_loc, MemoryLocation out_loc) {
        const DMAWorkload dmaOld_{
                VPUDevice::VPU_4_0,                            // device
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // input dimensions WHCB
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // output dimensions
                in_loc,                                        // src
                out_loc,                                       // dst
                1,                                             // owt
        };
        const DMANNWorkload_NPU40 dmaNN = DMAWorkloadTransformer::create_workload<DMANNWorkload_NPU40>(dmaOld_);
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return dmaNN;
    };
    const float ctoDPU{1700.0f / 975.0f};
    const int c{1};

    const std::vector<TestCase> tc{
            //// expected times are in VPU clock
            // 38  CMX2CMX
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},        //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(3, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},        //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(6, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(12, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},       //
            {mkwl(24, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(48, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(96, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},       //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},      //
            {mkwl(192, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},      //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},      //
            {mkwl(384, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},      //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},      //
            {mkwl(768, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},      //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},     //
            {mkwl(1536, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},     //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX), 253 * c},     //
            {mkwl(3072, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},     //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::CMX), 253 * c},     //
            {mkwl(6144, MemoryLocation::CMX, MemoryLocation::CMX), 304 * c},     //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::CMX), 304 * c},     //
            {mkwl(12288, MemoryLocation::CMX, MemoryLocation::CMX), 355 * c},    //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::CMX), 456 * c},    //
            {mkwl(24576, MemoryLocation::CMX, MemoryLocation::CMX), 557 * c},    //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::CMX), 709 * c},    //
            {mkwl(49152, MemoryLocation::CMX, MemoryLocation::CMX), 961 * c},    //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::CMX), 1214 * c},   //
            {mkwl(98304, MemoryLocation::CMX, MemoryLocation::CMX), 1720 * c},   //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::CMX), 2226 * c},  //
            {mkwl(196608, MemoryLocation::CMX, MemoryLocation::CMX), 3288 * c},  //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::CMX), 4299 * c},  //
            {mkwl(393216, MemoryLocation::CMX, MemoryLocation::CMX), 6373 * c},  //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::CMX), 8396 * c},  //

            ///
            // 40 CMX2DDR
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},          //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},          //
            {mkwl(3, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},          //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},          //
            {mkwl(6, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},          //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},          //
            {mkwl(12, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(24, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},         //
            {mkwl(48, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},         //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},         //
            {mkwl(96, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(192, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},        //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(384, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},        //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(768, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},       //
            {mkwl(1536, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(3072, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(6144, MemoryLocation::CMX, MemoryLocation::DRAM), 506 * c},       //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::DRAM), 355 * c},       //
            {mkwl(12288, MemoryLocation::CMX, MemoryLocation::DRAM), 506 * c},      //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::DRAM), 506 * c},      //
            {mkwl(24576, MemoryLocation::CMX, MemoryLocation::DRAM), 658 * c},      //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::DRAM), 810 * c},      //
            {mkwl(49152, MemoryLocation::CMX, MemoryLocation::DRAM), 1113 * c},     //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::DRAM), 1417 * c},     //
            {mkwl(98304, MemoryLocation::CMX, MemoryLocation::DRAM), 2023 * c},     //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::DRAM), 2681 * c},    //
            {mkwl(196608, MemoryLocation::CMX, MemoryLocation::DRAM), 3945 * c},    //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::DRAM), 5210 * c},    //
            {mkwl(393216, MemoryLocation::CMX, MemoryLocation::DRAM), 7688 * c},    //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::DRAM), 10166 * c},   //
            {mkwl(786432, MemoryLocation::CMX, MemoryLocation::DRAM), 15172 * c},   //
            {mkwl(1048576, MemoryLocation::CMX, MemoryLocation::DRAM), 20078 * c},  //
            ///

            // 40  DDR2CMX
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::CMX), 304 * c},          //
            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},          //
            {mkwl(3, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},          //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},          //
            {mkwl(6, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},          //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},          //
            {mkwl(12, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},         //
            {mkwl(24, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(48, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},         //
            {mkwl(96, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},         //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},        //
            {mkwl(192, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},        //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},        //
            {mkwl(384, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},        //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},        //
            {mkwl(768, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},        //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},       //
            {mkwl(1536, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},       //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},       //
            {mkwl(3072, MemoryLocation::DRAM, MemoryLocation::CMX), 506 * c},       //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::CMX), 506 * c},       //
            {mkwl(6144, MemoryLocation::DRAM, MemoryLocation::CMX), 759 * c},       //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::CMX), 709 * c},       //
            {mkwl(12288, MemoryLocation::DRAM, MemoryLocation::CMX), 709 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::CMX), 860 * c},      //
            {mkwl(24576, MemoryLocation::DRAM, MemoryLocation::CMX), 1012 * c},     //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::CMX), 1063 * c},     //
            {mkwl(49152, MemoryLocation::DRAM, MemoryLocation::CMX), 1467 * c},     //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::CMX), 1771 * c},     //
            {mkwl(98304, MemoryLocation::DRAM, MemoryLocation::CMX), 2327 * c},     //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::CMX), 2782 * c},    //
            {mkwl(196608, MemoryLocation::DRAM, MemoryLocation::CMX), 3692 * c},    //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::CMX), 4805 * c},    //
            {mkwl(393216, MemoryLocation::DRAM, MemoryLocation::CMX), 6878 * c},    //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::CMX), 9255 * c},    //
            {mkwl(786432, MemoryLocation::DRAM, MemoryLocation::CMX), 13605 * c},   //
            {mkwl(1048576, MemoryLocation::DRAM, MemoryLocation::CMX), 18106 * c},  //
                                                                                    ///

            // 43  DDR2DDR
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},          //
            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},          //
            {mkwl(3, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},          //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},          //
            {mkwl(6, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},          //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},          //
            {mkwl(12, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(24, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},         //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},         //
            {mkwl(48, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(96, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},         //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},        //
            {mkwl(192, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},        //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},        //
            {mkwl(384, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},        //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},        //
            {mkwl(768, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},        //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},       //
            {mkwl(1536, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},       //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},       //
            {mkwl(3072, MemoryLocation::DRAM, MemoryLocation::DRAM), 658 * c},       //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::DRAM), 506 * c},       //
            {mkwl(6144, MemoryLocation::DRAM, MemoryLocation::DRAM), 607 * c},       //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::DRAM), 658 * c},       //
            {mkwl(12288, MemoryLocation::DRAM, MemoryLocation::DRAM), 709 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::DRAM), 810 * c},      //
            {mkwl(24576, MemoryLocation::DRAM, MemoryLocation::DRAM), 961 * c},      //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::DRAM), 1164 * c},     //
            {mkwl(49152, MemoryLocation::DRAM, MemoryLocation::DRAM), 1518 * c},     //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::DRAM), 1973 * c},     //
            {mkwl(98304, MemoryLocation::DRAM, MemoryLocation::DRAM), 2782 * c},     //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::DRAM), 3541 * c},    //
            {mkwl(196608, MemoryLocation::DRAM, MemoryLocation::DRAM), 5412 * c},    //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::DRAM), 7384 * c},    //
            {mkwl(393216, MemoryLocation::DRAM, MemoryLocation::DRAM), 10671 * c},   //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::DRAM), 13908 * c},   //
            {mkwl(786432, MemoryLocation::DRAM, MemoryLocation::DRAM), 21089 * c},   //
            {mkwl(1048576, MemoryLocation::DRAM, MemoryLocation::DRAM), 28574 * c},  //
            {mkwl(1572864, MemoryLocation::DRAM, MemoryLocation::DRAM), 40914 * c},  //
            {mkwl(2097152, MemoryLocation::DRAM, MemoryLocation::DRAM), 57300 * c},  //
            {mkwl(3145728, MemoryLocation::DRAM, MemoryLocation::DRAM), 84356 * c},  //
                                                                                     ///

    };

    DMACostModel<DMANNWorkload_NPU40> model_4_0{VPU_DMA_4_0_MODEL_PATH};
    EXPECT_TRUE(model_4_0.nn_initialized());

    // const std::string modelFile{VPU_2_7_MODEL_PATH};
    // VPUCostModel old_model{modelFile};

    constexpr int abs_err_negligible{200};  // cycles
    constexpr float rel_err_max{1.10f};     // rel to GT  HUGE for now!!!!!!!!!!!!!!!!!!

    constexpr bool force_fail{false};                       // forces one fail
    constexpr bool ignore_relative_errors_failures{false};  // no t allowing this failures

    auto runTest = [=, &model_4_0](const TestCase& tc) {
        const DMANNWorkload_NPU40 dmaNN_{tc.t_in};

        std::string info{};
        const auto cycles_ = model_4_0.computeCycles(dmaNN_, info);

        if (Cycles::isErrorCode(cycles_)) {
            // const auto = model_4_0.computeBandwidthMsg(dmaNN_);
            ASSERT_FALSE(Cycles::isErrorCode(cycles_))
                    << "Error info:" << info << "\n Error at inference not expected: " << Cycles::toErrorText(cycles_)
                    << " , \n " << dmaNN_;
        }

        CyclesInterfaceType exp{CyclesInterfaceType(ctoDPU * tc.t_exp)};

        const auto delta{std::abs((long long)cycles_ - (long long)exp)};
        const auto rel_err{(float)delta / (float)exp};

        if (delta > abs_err_negligible) {
            if (dmaNN_.getAccessedBytes() >= 512 /*bytes*/) {
                // look at relative error
                if (!ignore_relative_errors_failures) {
                    ASSERT_LE(rel_err, rel_err_max)
                            << "Bytes accessed:" << dmaNN_.getAccessedBytes() << "\n"
                            << dmaNN_ << " Inferred time: " << cycles_ << ", GT " << exp << "\n";
                }
            } else if (dmaNN_.getAccessedBytes() >= 5 /*bytes*/) {
                // delta should not be more than 150%
                ASSERT_LE(delta, (3 * exp) / 2) << "Bytes accessed:" << dmaNN_.getAccessedBytes() << "\n"
                                                << dmaNN_ << " INferred time: " << cycles_ << ", GT " << exp << "\n";
            } else {  // small , very small
                // delta  , 8x
                ASSERT_LE(delta, (8 * exp)) << "Bytes accessed:" << dmaNN_.getAccessedBytes() << "\n"
                                            << dmaNN_ << " INferred time: " << cycles_ << ", GT " << exp << "\n";
            }
        }

        std::cout << "\nBYTES:, " << dmaNN_.getAccessedBytes() << " , " << (int)dmaNN_.transfer_direction << ":"
                  << MemoryDirection_ToText.at((int)dmaNN_.transfer_direction) << " ,gt:  " << exp
                  << " , inferred: " << cycles_ << " , delta: " << delta << " ,rel err:  " << rel_err << " ,  "
                  << (rel_err > rel_err_max ? "     TOO BIG%!" : "");
    };

    EXPECT_TRUE(!force_fail);

    std::cout << "\nMax Relative error allowed: " << rel_err_max;
    std::cout << "\nBYTES,   , Direction    ,  GT  , inferred  , delta  ,  err \n";

    for (const auto& t : tc) {
        runTest(t);
    }
}

// Legacy is not supported!
TEST_F(TestDMANNCostModelNPU4x, SweepGT_DMATime_40_Theoretical) {
    class TestCase {
    public:
        DMAWorkload t_in;           // wl
        CyclesInterfaceType t_exp;  // expected time (GT)
    };

    auto mkwl = [](const int bytes, MemoryLocation in_loc, MemoryLocation out_loc) {
        const DMAWorkload dmaOld_{
                VPUDevice::VPU_4_0,                            // device
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // input dimensions WHCB
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // output dimensions
                in_loc,                                        // src
                out_loc,                                       // dst
                1,                                             // owt
        };
        return dmaOld_;
    };
    const float ctoDPU{1700.0f / 975.0f};
    const int c{1};

    const std::vector<TestCase> tc{
            //// expected times are in VPU clock
            // 38  CMX2CMX
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},        //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(3, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},        //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(6, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},        //
            {mkwl(12, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},       //
            {mkwl(24, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(48, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},       //
            {mkwl(96, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},       //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},      //
            {mkwl(192, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},      //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},      //
            {mkwl(384, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},      //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::CMX), 152 * c},      //
            {mkwl(768, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},      //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},     //
            {mkwl(1536, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},     //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX), 253 * c},     //
            {mkwl(3072, MemoryLocation::CMX, MemoryLocation::CMX), 203 * c},     //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::CMX), 253 * c},     //
            {mkwl(6144, MemoryLocation::CMX, MemoryLocation::CMX), 304 * c},     //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::CMX), 304 * c},     //
            {mkwl(12288, MemoryLocation::CMX, MemoryLocation::CMX), 355 * c},    //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::CMX), 456 * c},    //
            {mkwl(24576, MemoryLocation::CMX, MemoryLocation::CMX), 557 * c},    //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::CMX), 709 * c},    //
            {mkwl(49152, MemoryLocation::CMX, MemoryLocation::CMX), 961 * c},    //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::CMX), 1214 * c},   //
            {mkwl(98304, MemoryLocation::CMX, MemoryLocation::CMX), 1720 * c},   //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::CMX), 2226 * c},  //
            {mkwl(196608, MemoryLocation::CMX, MemoryLocation::CMX), 3288 * c},  //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::CMX), 4299 * c},  //
            {mkwl(393216, MemoryLocation::CMX, MemoryLocation::CMX), 6373 * c},  //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::CMX), 8396 * c},  //

            ///
            // 40 CMX2DDR
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},          //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},          //
            {mkwl(3, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},          //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},          //
            {mkwl(6, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},          //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},          //
            {mkwl(12, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(24, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},         //
            {mkwl(48, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},         //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},         //
            {mkwl(96, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},         //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(192, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},        //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(384, MemoryLocation::CMX, MemoryLocation::DRAM), 152 * c},        //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(768, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},        //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM), 203 * c},       //
            {mkwl(1536, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(3072, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},       //
            {mkwl(6144, MemoryLocation::CMX, MemoryLocation::DRAM), 506 * c},       //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::DRAM), 355 * c},       //
            {mkwl(12288, MemoryLocation::CMX, MemoryLocation::DRAM), 506 * c},      //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::DRAM), 506 * c},      //
            {mkwl(24576, MemoryLocation::CMX, MemoryLocation::DRAM), 658 * c},      //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::DRAM), 810 * c},      //
            {mkwl(49152, MemoryLocation::CMX, MemoryLocation::DRAM), 1113 * c},     //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::DRAM), 1417 * c},     //
            {mkwl(98304, MemoryLocation::CMX, MemoryLocation::DRAM), 2023 * c},     //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::DRAM), 2681 * c},    //
            {mkwl(196608, MemoryLocation::CMX, MemoryLocation::DRAM), 3945 * c},    //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::DRAM), 5210 * c},    //
            {mkwl(393216, MemoryLocation::CMX, MemoryLocation::DRAM), 7688 * c},    //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::DRAM), 10166 * c},   //
            {mkwl(786432, MemoryLocation::CMX, MemoryLocation::DRAM), 15172 * c},   //
            {mkwl(1048576, MemoryLocation::CMX, MemoryLocation::DRAM), 20078 * c},  //
            ///

            // 40  DDR2CMX
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::CMX), 304 * c},          //
            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},          //
            {mkwl(3, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},          //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},          //
            {mkwl(6, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},          //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},          //
            {mkwl(12, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},         //
            {mkwl(24, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(48, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},         //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},         //
            {mkwl(96, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},         //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::CMX), 355 * c},        //
            {mkwl(192, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},        //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::CMX), 405 * c},        //
            {mkwl(384, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},        //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},        //
            {mkwl(768, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},        //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},       //
            {mkwl(1536, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},       //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX), 456 * c},       //
            {mkwl(3072, MemoryLocation::DRAM, MemoryLocation::CMX), 506 * c},       //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::CMX), 506 * c},       //
            {mkwl(6144, MemoryLocation::DRAM, MemoryLocation::CMX), 759 * c},       //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::CMX), 709 * c},       //
            {mkwl(12288, MemoryLocation::DRAM, MemoryLocation::CMX), 709 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::CMX), 860 * c},      //
            {mkwl(24576, MemoryLocation::DRAM, MemoryLocation::CMX), 1012 * c},     //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::CMX), 1063 * c},     //
            {mkwl(49152, MemoryLocation::DRAM, MemoryLocation::CMX), 1467 * c},     //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::CMX), 1771 * c},     //
            {mkwl(98304, MemoryLocation::DRAM, MemoryLocation::CMX), 2327 * c},     //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::CMX), 2782 * c},    //
            {mkwl(196608, MemoryLocation::DRAM, MemoryLocation::CMX), 3692 * c},    //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::CMX), 4805 * c},    //
            {mkwl(393216, MemoryLocation::DRAM, MemoryLocation::CMX), 6878 * c},    //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::CMX), 9255 * c},    //
            {mkwl(786432, MemoryLocation::DRAM, MemoryLocation::CMX), 13605 * c},   //
            {mkwl(1048576, MemoryLocation::DRAM, MemoryLocation::CMX), 18106 * c},  //
                                                                                    ///

            // 43  DDR2DDR
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},          //
            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},          //
            {mkwl(3, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},          //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},          //
            {mkwl(6, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},          //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},          //
            {mkwl(12, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(24, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},         //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},         //
            {mkwl(48, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},         //
            {mkwl(96, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},         //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::DRAM), 355 * c},        //
            {mkwl(192, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},        //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},        //
            {mkwl(384, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},        //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::DRAM), 405 * c},        //
            {mkwl(768, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},        //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},       //
            {mkwl(1536, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},       //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM), 456 * c},       //
            {mkwl(3072, MemoryLocation::DRAM, MemoryLocation::DRAM), 658 * c},       //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::DRAM), 506 * c},       //
            {mkwl(6144, MemoryLocation::DRAM, MemoryLocation::DRAM), 607 * c},       //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::DRAM), 658 * c},       //
            {mkwl(12288, MemoryLocation::DRAM, MemoryLocation::DRAM), 709 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::DRAM), 810 * c},      //
            {mkwl(24576, MemoryLocation::DRAM, MemoryLocation::DRAM), 961 * c},      //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::DRAM), 1164 * c},     //
            {mkwl(49152, MemoryLocation::DRAM, MemoryLocation::DRAM), 1518 * c},     //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::DRAM), 1973 * c},     //
            {mkwl(98304, MemoryLocation::DRAM, MemoryLocation::DRAM), 2782 * c},     //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::DRAM), 3541 * c},    //
            {mkwl(196608, MemoryLocation::DRAM, MemoryLocation::DRAM), 5412 * c},    //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::DRAM), 7384 * c},    //
            {mkwl(393216, MemoryLocation::DRAM, MemoryLocation::DRAM), 10671 * c},   //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::DRAM), 13908 * c},   //
            {mkwl(786432, MemoryLocation::DRAM, MemoryLocation::DRAM), 21089 * c},   //
            {mkwl(1048576, MemoryLocation::DRAM, MemoryLocation::DRAM), 28574 * c},  //
            {mkwl(1572864, MemoryLocation::DRAM, MemoryLocation::DRAM), 40914 * c},  //
            {mkwl(2097152, MemoryLocation::DRAM, MemoryLocation::DRAM), 57300 * c},  //
            {mkwl(3145728, MemoryLocation::DRAM, MemoryLocation::DRAM), 84356 * c},  //
                                                                                     ///

    };

    // DMACostModel<DMANNWorkload_NPU40> model_4_0{VPU_DMA_4_0_MODEL_PATH};
    // EXPECT_TRUE(model_4_0.nn_initialized());
    VPUCostModel old_model{"dummy"};

    constexpr int abs_err_negligible{200};  // cycles
    constexpr float rel_err_max{0.8f};      // rel to GT  HUGE for now!!!!!!!!!!!!!!!!!!

    constexpr bool force_fail{false};                       // forces one fail
    constexpr bool ignore_relative_errors_failures{false};  // no t allowing this failures

    auto runTest = [=, &old_model](const TestCase& tc) {
        const auto dmaNN_{tc.t_in};

        std::string info{"NA"};
        // const auto cycles_ = model_4_0.computeCycles(dmaNN_, info);
        const auto cycles_ = old_model.DMA(dmaNN_ /*, info*/);

        if (Cycles::isErrorCode(cycles_)) {
            // const auto = model_4_0.computeBandwidthMsg(dmaNN_);
            ASSERT_FALSE(Cycles::isErrorCode(cycles_))
                    << "Error info:" << info << "\n Error at inference not expected: " << Cycles::toErrorText(cycles_)
                    << " , \n " << dmaNN_;
        }

        CyclesInterfaceType exp{CyclesInterfaceType(ctoDPU * tc.t_exp)};

        const auto delta{std::abs((long long)cycles_ - (long long)exp)};
        const auto rel_err{(float)delta / (float)exp};
        const auto bytesAccesed{dmaNN_.input.size()};
        const auto txDir{DMAWorkloadTransformer::create_direction(dmaNN_.input_location, dmaNN_.output_location)};

        if (delta > abs_err_negligible) {
            if (dmaNN_.input.size() >= 512 /*bytes*/) {
                // look at relative error
                if (!ignore_relative_errors_failures) {
                    ASSERT_LE(rel_err, rel_err_max)
                            << "Bytes accessed:" << bytesAccesed << "\n"
                            << dmaNN_ << " Inferred time: " << cycles_ << ", GT " << exp << "\n";
                }
            } else if (bytesAccesed >= 5 /*bytes*/) {
                // delta should not be more than 150%
                ASSERT_LE(delta, (3 * exp) / 2) << "Bytes accessed:" << bytesAccesed << "\n"
                                                << dmaNN_ << " INferred time: " << cycles_ << ", GT " << exp << "\n";
            } else {  // small , very small
                // delta  , 8x
                ASSERT_LE(delta, (8 * exp)) << "Bytes accessed:" << bytesAccesed << "\n"
                                            << dmaNN_ << " INferred time: " << cycles_ << ", GT " << exp << "\n";
            }
        }

        std::cout << "\nBYTES:, " << bytesAccesed << " , " << (int)txDir << ":" << MemoryDirection_ToText.at((int)txDir)
                  << " ,gt:  " << exp << " , inferred: " << cycles_ << " , delta: " << delta
                  << " ,rel err:  " << rel_err << " ,  " << (rel_err > rel_err_max ? "     TOO BIG%!" : "");
    };

    EXPECT_TRUE(!force_fail);

    std::cout << "\nMax Relative error allowed: " << rel_err_max;
    std::cout << "\nBYTES,   , Direction    ,  GT  , inferred  , delta  ,  err \n";

    for (const auto& t : tc) {
        if constexpr (!PerformanceMode::forceLegacy_G4) {
            runTest(t);
        }
    }
}

// this test is for New/Updated theoretical model
TEST_F(TestDMA_TH_CostModelNPU4x, DMA_Theoretical_regresion_NPU40) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    VPUDevice device{VPUDevice::VPU_4_0};
    const int p = (50 * (int)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::VPU_4_0)) /
                  (int)GlobalHarwdwareCharacteristics::get_cmx_fclk(VPUDevice::VPU_4_0);
    EXPECT_EQ(87, p);

    const std::vector<TestCase> tc{
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 539 + p + 1, "1k DC"},     //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 567 + p + 2, "2k DC"},     //
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::CMX, device), 784 + p + 12, "10k DC"},  //

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 539 + p + 1, "1k CD"},     //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 567 + p + 2, "2k CD"},     //
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::DRAM, device), 784 + p + 12, "10k CD"},  //

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 85 + p, "1k CC"},     //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "2k CC"},    //
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::CMX, device), 330 + p, "10k CC"},  //

            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 539 + p, "1k DD"},    //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 567 + p, "2k DD"},    //
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 784 + p, "10k DD"},  //

            // compressed
            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 539 + p + 1, "1kto2k DC"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 567 + p, "2kto1k DC"},      //

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 567 + p, "1kto2k CD"},      //
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 567 + p + 2, "2kto1k CD"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "1kto2k CC"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 113 + p, "2kto1k CC"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 567 + p, "1kto2k  DD"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 567 + p, "2kto1k DD"},   //

            // permute
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, CMX, device), 539 + p + 1, "1k DCperm"},   //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, DRAM, device), 539 + p + 1, "1k CD perm"},  //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, CMX, device), 1849 + p, "1k CC perm"},      //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, DRAM, device), 539 + p, "1k DD perm"},     //

    };

    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        if constexpr (!PerformanceMode::forceLegacy_G4) {
            EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
        } else {
            EXPECT_GT(dma_cyc, 0) << tc.t_name << "\n" << tc.t_in;
        }
    };

    for (const auto& t : tc) {
        check(t);
    }
}

}  // namespace VPUNN_unit_tests