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

class TestDMANNCostModelVPU2x : public TestDMANNCostModel {
public:
protected:
    static constexpr int index_DMA_path_27{0};
};

class TestDMA_TH_CostModelVPU2x : public TestDMA_TH_CostModel {
public:
protected:
    DMANNWorkload_NPU27 wl_glob_27{
            VPUNN::VPUDevice::VPU_2_7,  // VPUDevice device;  ///< NPU device

            3,     // int num_planes;  ///< starts from 0. 1 plane = 0 as value?
            8192,  // int length;

            4096,  // int src_width;
            512,   // int dst_width;
            128,   // int src_stride;
            0,     // int dst_stride;
            128,   // int src_plane_stride;
            1024,  // int dst_plane_stride;

            MemoryDirection::DDR2DDR  // MemoryDirection transfer_direction;

            //
    };
};

TEST_F(TestDMANNCostModelVPU2x, LoadModels_BasicAssertions) {
    {  // 2_7
        const std::string model_path = the_NN_models.all_DMA_model_paths[index_DMA_path_27].first;
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x{model_path});
        DMACostModel<DMANNWorkload_NPU27> vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }
}

TEST_F(TestDMANNCostModelVPU2x, LoadModels_NN_Valid_Interval) {
    float down_exp = -0.1F;
    float up_exp = 1.1F;

    {  // empty models

        ASSERT_FALSE(model.nn_initialized());
        auto minmax = model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_7
        const std::string model_path = the_NN_models.all_DMA_model_paths[index_DMA_path_27].first;
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x{model_path});
        DMACostModel<DMANNWorkload_NPU27> vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }
}

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModelVPU2x, SmokeEmptyTestDMA) {
    DMANNWorkload_NPU27 wl = wl_glob_27;
    DMACostModel<DMANNWorkload_NPU27> emptyDMAModel;

    auto dma_cycles = emptyDMAModel.computeCycles(wl);

    // EXPECT_EQ(dma_cycles, V(Cycles::ERROR_INFERENCE_NOT_POSSIBLE)) << wl << Cycles::toErrorText(dma_cycles);
    EXPECT_NEAR(dma_cycles, 2820, 100);  // Fallback to Theoretical
}

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModelVPU2x, SmokeTestDMA_27) {
    const std::string model_path = VPU_DMA_2_7_MODEL_PATH;
    DMACostModel<DMANNWorkload_NPU27> dma_model(model_path);
    ASSERT_TRUE(dma_model.nn_initialized());

    {
        DMANNWorkload_NPU27 wl = wl_glob_27;
        auto dma_cycles = dma_model.computeCycles(wl);

        // value chosen by hand to detect problems without changing the DMANN
        EXPECT_EQ(dma_cycles, 3235 /*2946*/) << wl << Cycles::toErrorText(dma_cycles);  // gear4:3910
    }

    {
        DMANNWorkload_NPU27 wl{
                VPUNN::VPUDevice::VPU_2_7,  // VPUDevice device;  ///< NPU device
                0,                          // int num_planes;  ///< starts from 0. 1 plane = 0 as value?
                65535,                      // int length;

                65535,                    // int src_width;
                65535,                    // int dst_width;
                0,                        // int src_stride;
                0,                        // int dst_stride;
                0,                        // int src_plane_stride;
                0,                        // int dst_plane_stride;
                MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;
        };
        auto dma_cycles = dma_model.computeCycles(wl);
        // value chosen by hand to detect problems without changing the DMANN
        EXPECT_EQ(dma_cycles, 5574 /*@1300MHz*/) << wl << Cycles::toErrorText(dma_cycles);  // 5600 gear4
    }
}

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModelVPU2x, SmokeTestDMA_RAWNN) {
    DMANNWorkload_NPU27 wl = wl_glob_27;

    const std::string model_path = VPU_DMA_2_7_MODEL_PATH;
    DMACostModel<DMANNWorkload_NPU27> dma_model(model_path);

    ASSERT_TRUE(dma_model.nn_initialized());

   auto dma_raw = dma_model.computeBandwidthMsg(wl);

    // Expect equality.
    EXPECT_NEAR(std::get<0>(dma_raw), 0.42211675643920898F, 0.01) << wl;  // gear4:0.347663F
}

TEST_F(TestDMANNCostModelVPU2x, DISABLED_SweepDMATime_27) {
    const std::vector<int> dmaTxSize{1024,   2048,   4096,   8192,   12288,  16384,  20480,  24576,  28672,  32768,
                                     35854,  40960,  45956,  49152,  53248,  57344,  61440,  65536,  69632,  73728,
                                     77824,  81920,  86016,  90112,  94208,  98304,  102400, 106496, 110592, 131072,
                                     262144, 393216, 524288, 655360, 786432, 917504, 1048576};

    using DMAResult =
            std::tuple<int, CyclesInterfaceType, CyclesInterfaceType, CyclesInterfaceType, CyclesInterfaceType,
                       CyclesInterfaceType, CyclesInterfaceType, CyclesInterfaceType, CyclesInterfaceType>;

    DMACostModel<DMANNWorkload_NPU27> model_2_7{VPU_DMA_2_7_MODEL_PATH};
    EXPECT_TRUE(model_2_7.nn_initialized());

    const std::string modelFile{VPU_2_7_MODEL_PATH};
    VPUCostModel old_model{modelFile};

    auto measureDMA = [&model_2_7, &old_model](const int bytes) {
        const DMAWorkload dmaOld_DC{
                VPUDevice::VPU_2_7,                            // device
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // input dimensions WHCB
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // output dimensions
                MemoryLocation::DRAM,                          // src
                MemoryLocation::CMX,                           // dst
                1,                                             // owt
        };
        DMAWorkload dmaOld_CD{dmaOld_DC};
        dmaOld_CD.input_location = MemoryLocation::CMX;
        dmaOld_CD.output_location = MemoryLocation::DRAM;

        DMAWorkload dmaOld_CC{dmaOld_DC};
        dmaOld_CC.input_location = MemoryLocation::CMX;
        dmaOld_CC.output_location = MemoryLocation::CMX;

        DMAWorkload dmaOld_DD{dmaOld_DC};
        dmaOld_DD.input_location = MemoryLocation::DRAM;
        dmaOld_DD.output_location = MemoryLocation::DRAM;

        const DMANNWorkload_NPU27 dmaNN_DC = DMAWorkloadTransformer::create_workload(dmaOld_DC);
        const DMANNWorkload_NPU27 dmaNN_CD = DMAWorkloadTransformer::create_workload(dmaOld_CD);
        const DMANNWorkload_NPU27 dmaNN_CC = DMAWorkloadTransformer::create_workload(dmaOld_CC);
        const DMANNWorkload_NPU27 dmaNN_DD = DMAWorkloadTransformer::create_workload(dmaOld_DD);

        // std::cout << "\n--------------------------------------------------------------------------------------";
        // std::cout << dmaNN_DC << std::endl << dmaNN_CD << std::endl << dmaNN_CC << std::endl << dmaNN_DD;
        // std::cout << "\n--------------------------------------------------------------------------------------";

        auto cycles_NN_DC = model_2_7.computeCycles(dmaNN_DC);
        auto cycles_NN_CD = model_2_7.computeCycles(dmaNN_CD);
        auto cycles_NN_CC = model_2_7.computeCycles(dmaNN_CC);
        auto cycles_NN_DD = model_2_7.computeCycles(dmaNN_DD);

        auto cycles_DC = old_model.DMA(dmaOld_DC);
        auto cycles_CD = old_model.DMA(dmaOld_CD);
        auto cycles_CC = old_model.DMA(dmaOld_CC);
        auto cycles_DD = old_model.DMA(dmaOld_DD);

        DMAResult res{bytes,                                                   //
                      cycles_NN_DC, cycles_NN_CD, cycles_NN_CC, cycles_NN_DD,  //
                      cycles_DC,    cycles_CD,    cycles_CC,    cycles_DD};
        return res;
    };

    using DMAData = std::vector<DMAResult>;
    auto show = [](const DMAData& d, const std::string& s, const auto& div) {
        const int w{8};
        const int p{3};
        std::cout << "\n *** " << s;
        std::cout << "\n BYTES      NN: DRAM>CMX  CMX>DRAM  CMX>CMX  DRAM>DRAM     \tOld: DRAM>CMX  CMX>DRAM  CMX>CMX  "
                     "DRAM>DRAM";
        for (const auto& m : d) {
            std::cout << "\n "                                   //
                      << std::setprecision(p)                    //
                      << std::setw(w) << std::get<0>(m) << "\t"  //
                      << std::setw(w) << std::get<1>(m) / div << "\t" << std::setw(w) << std::get<2>(m) / div << "\t"
                      << std::setw(w) << std::get<3>(m) / div << "\t" << std::setw(w) << std::get<4>(m) / div  //
                      << "\t\t"                                                                                //
                      << std::setw(w) << std::get<5>(m) / div << "\t" << std::setw(w) << std::get<6>(m) / div << "\t"
                      << std::setw(w) << std::get<7>(m) / div << "\t" << std::setw(w) << std::get<8>(m) / div  //
                    ;
        }
        std::cout << "\n ------------------------------";
        /* coverity[end_of_path] */
    };
    EXPECT_TRUE(false);
    DMAData r;
    for (const auto& i : dmaTxSize) {
        const auto m{measureDMA(i)};
        r.push_back(m);
    }
    show(r, "Cycles at 1300MHz (NPU2.7) , DMANNWorkload with all data in only one plane, ALL Data on channels for old",
         1);

    show(r, "MIcroseconds , DMANNWorkload with all data in only one plane, ALL Data on channels for old", 1300.0f);
}

TEST_F(TestDMANNCostModelVPU2x, SweepGT_DMATime_27) {
    class TestCase {
    public:
        const DMANNWorkload_NPU27 t_in;  // wl
        CyclesInterfaceType t_exp;       // expected time (GT)
    };

    auto mkwl = [](const int bytes, MemoryLocation in_loc, MemoryLocation out_loc) {
        const DMAWorkload dmaOld_{
                VPUDevice::VPU_2_7,                            // device
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // input dimensions WHCB
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // output dimensions
                in_loc,                                        // src
                out_loc,                                       // dst
                1,                                             // owt
        };
        const DMANNWorkload_NPU27 dmaNN = DMAWorkloadTransformer::create_workload(dmaOld_);
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return dmaNN;
    };
    const float ctoDPU{1300.0f / 975.0f};
    const int c{1};

    const std::vector<TestCase> tc{
            //// expected times are in VPU clock
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::CMX), 532 * c},   //
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},    //
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::DRAM), 195 * c},   //
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::DRAM), 497 * c},  //

            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::CMX), 535 * c},          //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::CMX), 101 * c},           //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::DRAM), 195 * c},          //
            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::DRAM), 501 * c},         //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::CMX), 570 * c},          //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},           //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::DRAM), 195 * c},          //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::DRAM), 503 * c},         //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::CMX), 555 * c},          //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::CMX), 101 * c},           //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::DRAM), 194 * c},          //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::DRAM), 505 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::CMX), 542 * c},         //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},          //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::DRAM), 195 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::DRAM), 516 * c},        //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::CMX), 536 * c},         //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},          //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::DRAM), 195 * c},         //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::DRAM), 504 * c},        //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::CMX), 532 * c},         //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::CMX), 102 * c},          //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::DRAM), 197 * c},         //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::DRAM), 497 * c},        //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::CMX), 536 * c},        //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::CMX), 105 * c},         //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::DRAM), 199 * c},        //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::DRAM), 505 * c},       //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::CMX), 554 * c},        //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::CMX), 109 * c},         //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::DRAM), 204 * c},        //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::DRAM), 530 * c},       //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::CMX), 601 * c},        //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::CMX), 117 * c},         //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::DRAM), 221 * c},        //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::DRAM), 542 * c},       //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX), 622 * c},       //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX), 147 * c},        //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM), 257 * c},       //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM), 635 * c},      //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX), 668 * c},       //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX), 210 * c},        //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM), 283 * c},       //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM), 645 * c},      //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::CMX), 744 * c},       //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::CMX), 338 * c},        //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::DRAM), 347 * c},       //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::DRAM), 750 * c},      //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::CMX), 845 * c},       //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::CMX), 594 * c},        //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::DRAM), 476 * c},       //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::DRAM), 880 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::CMX), 1166 * c},     //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::CMX), 1108 * c},      //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::DRAM), 736 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::DRAM), 1133 * c},    //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::CMX), 1675 * c},     //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::CMX), 2136 * c},      //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::DRAM), 1392 * c},     //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::DRAM), 1668 * c},    //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::CMX), 2676 * c},     //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::CMX), 4192 * c},      //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::DRAM), 2279 * c},     //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::DRAM), 2881 * c},    //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::CMX), 4736 * c},    //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::CMX), 8304 * c},     //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::DRAM), 4335 * c},    //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::DRAM), 5251 * c},   //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::CMX), 8859 * c},    //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::CMX), 16528 * c},    //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::DRAM), 8447 * c},    //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::DRAM), 10176 * c},  //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::CMX), 18415 * c},   //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::CMX), 32976 * c},    //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::DRAM), 16670 * c},   //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::DRAM), 19625 * c},  //

    };

    DMACostModel<DMANNWorkload_NPU27> model_2_7{VPU_DMA_2_7_MODEL_PATH};
    EXPECT_TRUE(model_2_7.nn_initialized());

    // const std::string modelFile{VPU_2_7_MODEL_PATH};
    // VPUCostModel old_model{modelFile};

    const int abs_err_negligible{200};  // cycles
    const float rel_err_max{0.15f};     // rel to GT  15%

    auto runTest = [=, &model_2_7](const TestCase& tc) {
        const DMANNWorkload_NPU27 dmaNN_{tc.t_in};

        auto cycles_ = model_2_7.computeCycles(dmaNN_);
        CyclesInterfaceType exp{CyclesInterfaceType(ctoDPU * tc.t_exp)};

        const auto delta{std::abs((long long)cycles_ - (long long)exp)};
        const auto rel_err{(float)delta / (float)exp};

        if (delta > abs_err_negligible) {
            if (dmaNN_.getAccessedBytes() >= 512 /*bytes*/) {
                // look at relative error
                ASSERT_LE(rel_err, rel_err_max) << "Bytes accessed:" << dmaNN_.getAccessedBytes() << "\n"
                                                << dmaNN_ << " Inferred time: " << cycles_ << ", GT " << exp << "\n";
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

    // EXPECT_TRUE(false);

    std::cout << "\nMax Relative error allowed: " << rel_err_max;
    std::cout << "\nBYTES,   , Direction    ,  GT  , inferred  , delta  ,  err \n";

    for (const auto& t : tc) {
        runTest(t);
    }
}

// this test uses the GT from gear 4. The NN trained for gear 4 has to be used also at runtime
// enable when gear4 nn is available in repo
TEST_F(TestDMANNCostModelVPU2x, SweepGT_DMATime_27_GEAR4) {
    class TestCase {
    public:
        const DMANNWorkload_NPU27 t_in;  // wl
        CyclesInterfaceType t_exp;       // expected time (GT)
    };

    auto mkwl = [](const int bytes, MemoryLocation in_loc, MemoryLocation out_loc) {
        const DMAWorkload dmaOld_{
                VPUDevice::VPU_2_7,                            // device
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // input dimensions WHCB
                {VPUTensor(bytes, 1, 1, 1, DataType::UINT8)},  // output dimensions
                in_loc,                                        // src
                out_loc,                                       // dst
                1,                                             // owt
        };
        const DMANNWorkload_NPU27 dmaNN = DMAWorkloadTransformer::create_workload(dmaOld_);
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return dmaNN;
    };
    const float ctoDPU{1300.0f / 975.0f};
    const int c{1};

    const std::vector<TestCase> tc{
            //// expected times are in VPU clock
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::CMX), 806 * c},   //
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::CMX), 104 * c},    //
            {mkwl(1, MemoryLocation::CMX, MemoryLocation::DRAM), 252 * c},   //
            {mkwl(1, MemoryLocation::DRAM, MemoryLocation::DRAM), 700 * c},  //

            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::CMX), 813 * c},          //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::CMX), 106 * c},           //
            {mkwl(2, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},          //
            {mkwl(2, MemoryLocation::DRAM, MemoryLocation::DRAM), 735 * c},         //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::CMX), 757 * c},          //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::CMX), 105 * c},           //
            {mkwl(4, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},          //
            {mkwl(4, MemoryLocation::DRAM, MemoryLocation::DRAM), 718 * c},         //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::CMX), 791 * c},          //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::CMX), 104 * c},           //
            {mkwl(8, MemoryLocation::CMX, MemoryLocation::DRAM), 252 * c},          //
            {mkwl(8, MemoryLocation::DRAM, MemoryLocation::DRAM), 700 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::CMX), 786 * c},         //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::CMX), 104 * c},          //
            {mkwl(16, MemoryLocation::CMX, MemoryLocation::DRAM), 252 * c},         //
            {mkwl(16, MemoryLocation::DRAM, MemoryLocation::DRAM), 721 * c},        //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::CMX), 761 * c},         //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::CMX), 105 * c},          //
            {mkwl(32, MemoryLocation::CMX, MemoryLocation::DRAM), 253 * c},         //
            {mkwl(32, MemoryLocation::DRAM, MemoryLocation::DRAM), 709 * c},        //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::CMX), 763 * c},         //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::CMX), 106 * c},          //
            {mkwl(64, MemoryLocation::CMX, MemoryLocation::DRAM), 254 * c},         //
            {mkwl(64, MemoryLocation::DRAM, MemoryLocation::DRAM), 713 * c},        //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::CMX), 736 * c},        //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::CMX), 108 * c},         //
            {mkwl(128, MemoryLocation::CMX, MemoryLocation::DRAM), 259 * c},        //
            {mkwl(128, MemoryLocation::DRAM, MemoryLocation::DRAM), 699 * c},       //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::CMX), 806 * c},        //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::CMX), 111 * c},         //
            {mkwl(256, MemoryLocation::CMX, MemoryLocation::DRAM), 265 * c},        //
            {mkwl(256, MemoryLocation::DRAM, MemoryLocation::DRAM), 751 * c},       //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::CMX), 810 * c},        //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::CMX), 119 * c},         //
            {mkwl(512, MemoryLocation::CMX, MemoryLocation::DRAM), 287 * c},        //
            {mkwl(512, MemoryLocation::DRAM, MemoryLocation::DRAM), 795 * c},       //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX), 874 * c},       //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX), 150 * c},        //
            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM), 329 * c},       //
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM), 826 * c},      //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX), 879 * c},       //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX), 212 * c},        //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM), 352 * c},       //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM), 856 * c},      //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::CMX), 920 * c},       //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::CMX), 342 * c},        //
            {mkwl(4096, MemoryLocation::CMX, MemoryLocation::DRAM), 415 * c},       //
            {mkwl(4096, MemoryLocation::DRAM, MemoryLocation::DRAM), 991 * c},      //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::CMX), 1162 * c},      //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::CMX), 598 * c},        //
            {mkwl(8192, MemoryLocation::CMX, MemoryLocation::DRAM), 543 * c},       //
            {mkwl(8192, MemoryLocation::DRAM, MemoryLocation::DRAM), 1034 * c},     //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::CMX), 1328 * c},     //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::CMX), 1113 * c},      //
            {mkwl(16384, MemoryLocation::CMX, MemoryLocation::DRAM), 807 * c},      //
            {mkwl(16384, MemoryLocation::DRAM, MemoryLocation::DRAM), 1315 * c},    //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::CMX), 1836 * c},     //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::CMX), 2130 * c},      //
            {mkwl(32768, MemoryLocation::CMX, MemoryLocation::DRAM), 1368 * c},     //
            {mkwl(32768, MemoryLocation::DRAM, MemoryLocation::DRAM), 1876 * c},    //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::CMX), 2891 * c},     //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::CMX), 4192 * c},      //
            {mkwl(65536, MemoryLocation::CMX, MemoryLocation::DRAM), 2347 * c},     //
            {mkwl(65536, MemoryLocation::DRAM, MemoryLocation::DRAM), 3320 * c},    //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::CMX), 4955 * c},    //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::CMX), 8309 * c},     //
            {mkwl(131072, MemoryLocation::CMX, MemoryLocation::DRAM), 4406 * c},    //
            {mkwl(131072, MemoryLocation::DRAM, MemoryLocation::DRAM), 6058 * c},   //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::CMX), 9095 * c},    //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::CMX), 16612 * c},    //
            {mkwl(262144, MemoryLocation::CMX, MemoryLocation::DRAM), 8598 * c},    //
            {mkwl(262144, MemoryLocation::DRAM, MemoryLocation::DRAM), 11706 * c},  //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::CMX), 17267 * c},   //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::CMX), 33178 * c},    //
            {mkwl(524288, MemoryLocation::CMX, MemoryLocation::DRAM), 16940 * c},   //
            {mkwl(524288, MemoryLocation::DRAM, MemoryLocation::DRAM), 22671 * c},  //

    };
    // insert a new define!
    DMACostModel<DMANNWorkload_NPU27> model_2_7{VPU_DMA_2_7_G4_MODEL_PATH};  // use here the Gear4 slow version of DMANN
    EXPECT_TRUE(model_2_7.nn_initialized());

    // const std::string modelFile{VPU_2_7_MODEL_PATH};
    // VPUCostModel old_model{modelFile};

    const int abs_err_negligible{200};  // cycles
    const float rel_err_max{0.10f};     // rel to GT

    auto runTest = [=, &model_2_7](const TestCase& tc) {
        const DMANNWorkload_NPU27 dmaNN_{tc.t_in};

        auto cycles_ = model_2_7.computeCycles(dmaNN_);
        CyclesInterfaceType exp{CyclesInterfaceType(ctoDPU * tc.t_exp)};

        const auto delta{std::abs((long long)cycles_ - (long long)exp)};
        const auto rel_err{(float)delta / (float)exp};

        if (delta > abs_err_negligible) {
            if (tc.t_in.getAccessedBytes() >= 512) {
                ASSERT_LE(rel_err, rel_err_max) << tc.t_in << " INferred time: " << cycles_ << ", GT " << exp << "\n";
            } else {  // small
                ASSERT_LE(delta, (3 * exp) / 2) << tc.t_in << " INferred time: " << cycles_ << ", GT " << exp << "\n";
            }
        }

        std::cout << "\nBYTES:, " << dmaNN_.getAccessedBytes() << " , " << (int)dmaNN_.transfer_direction << ":"
                  << MemoryDirection_ToText.at((int)dmaNN_.transfer_direction) << " , " << exp << " , " << cycles_
                  << " , " << delta << " , " << rel_err << " ,  " << (rel_err > rel_err_max ? "     TOO BIG%!" : "");
    };

    // EXPECT_TRUE(false);

    std::cout << "\nMax Relative error allowed: " << rel_err_max;
    std::cout << "\nBYTES,   , Direction    ,  GT  , inferred  , delta  ,  err \n";

    for (const auto& t : tc) {
        runTest(t);
    }
}

TEST_F(TestDMA_TH_CostModelVPU2x, DMA_Theoretical_experiments) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    VPUDevice device{VPUDevice::VPU_2_7};

    auto wl_8kddr = mkwl(8192, MemoryLocation::DRAM, MemoryLocation::CMX, device);

    auto dma_8kddr = cm.DMA(wl_8kddr);

    // Expect equality.
    EXPECT_EQ(dma_8kddr, 1637) << wl_8kddr;  //
}

TEST_F(TestDMA_TH_CostModelVPU2x, DMA_Theoretical_regresion_NPU27) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    VPUDevice device{VPUDevice::VPU_2_7};

    const std::vector<TestCase> tc{
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 1292, "1k"},    //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 1341, "2k"},    //
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::CMX, device), 1724, "10k"},  //

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 1292, "1k"},    //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 1341, "2k"},    //
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::DRAM, device), 1724, "10k"},  //

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "1k"},    //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 192, "2k"},    //
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::CMX, device), 855, "10k"},  //

            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 1292, "1k"},    //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 1341, "2k"},    //
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 1724, "10k"},  //

            // compressed
            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 1292, "1kto2k"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 1341, "2kto1k"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 1341, "1kto2k"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 1292, "2kto1k"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "1kto2k"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "2kto1k"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 1341, "1kto2k"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 1341, "2kto1k"},  //

            // permute
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, CMX, device), 2608, "1k"},   //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, DRAM, device), 2608, "1k"},   //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, CMX, device), 1387, "1k"},    //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, DRAM, device), 1292, "1k"},  //

    };

    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
    };

    for (const auto& t : tc) {
        check(t);
    }
}

}  // namespace VPUNN_unit_tests