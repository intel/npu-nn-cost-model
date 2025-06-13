// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
// #include "vpu_cost_model.h"
#include "vpu_dma_cost_model.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/compatibility/types01.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu_cost_model.h"

#include <algorithm>
#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestDMANNCostModel : public ::testing::Test {
public:
protected:
    const DMANNWorkload_NPU27 wl_glob_27{
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
    //   VPUNN::DPUWorkload wl_glob_20;
    DMANNWorkload_NPU27 wl_glob_40M{wl_glob_27};

    DMACostModel<DMANNWorkload_NPU27> model{};
    // DMACostModel specialEmptyDMAModel;

    const DMANNWorkload_NPU40 wl_glob_40{
            VPUNN::VPUDevice::VPU_4_0,  // VPUDevice device;  ///< NPU device

            8192,  // int src_width;
            8192,  // int dst_width;

            0,  // int num_dim;
            {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
            Num_DMA_Engine::Num_Engine_1,
            MemoryDirection::CMX2CMX  // MemoryDirection transfer_direction;

            //
    };

    static constexpr int index_DMA_path_27{0};

    void SetUp() override {
        wl_glob_40M.device = VPUNN::VPUDevice::VPU_4_0;
    }

    auto read_a_file(const std::string filename) const {
        std::vector<char> buf(0);
        std::ifstream myFile;
        myFile.open(filename, std::ios::binary | std::ios::in);
        if (myFile.fail()) {
            // File does not exist code here
            return buf;
        }
        myFile.seekg(0, std::ios::end);
        const auto length = myFile.tellg();
        myFile.seekg(0, std::ios::beg);

        buf.resize(static_cast<size_t>(length));

        myFile.read(buf.data(), length);
        myFile.close();
        return buf;
    }

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    // struct DataOut {
    //     int errors_cnt{-1};
    //     double correlation{-10.0};
    // };

    VPUNN::CyclesInterfaceType delta_cycles(const VPUNN::CyclesInterfaceType& v1,
                                            const VPUNN::CyclesInterfaceType& v2) {
        return (v1 >= v2) ? (v1 - v2) : (v2 - v1);  // aways positive
    }

    /// @brief max allowable delta between 2 cycles , so that we consider them still equal
    ///
    /// @param v1 a value
    /// @param v2 another value
    /// @param tolerance_level how permissive to be in delta.
    /// @returns max value that can be between v1 and v2 so that they are practically equal.
    VPUNN::CyclesInterfaceType max_tolerance_cycles(const VPUNN::CyclesInterfaceType& v1,
                                                    const VPUNN::CyclesInterfaceType& v2,
                                                    const int tolerance_level = 1) {
        const VPUNN::CyclesInterfaceType v{std::max(v1, v2)};

        VPUNN::CyclesInterfaceType tolerance{1U};  // rounding errors

        if (tolerance_level <= 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 10U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 8U;
            } else if (v >= 100000U) {  // 100k
                tolerance = 5U;
            } else if (v >= 1000U) {
                tolerance = 2U;
            }

        } else if (tolerance_level > 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 20U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 10U;
            } else if (v >= 1000U) {
                tolerance = 2U;
            }
        }

        return tolerance;
    }

private:
};

TEST_F(TestDMANNCostModel, LoadModels_BasicAssertions) {
    {  // 2_7
        const std::string model_path = the_NN_models.all_DMA_model_paths[index_DMA_path_27].first;
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x{model_path});
        DMACostModel<DMANNWorkload_NPU27> vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }
}

TEST_F(TestDMANNCostModel, LoadModels_NN_Valid_Interval) {
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
TEST_F(TestDMANNCostModel, SmokeEmptyTestDMA) {
    DMANNWorkload_NPU27 wl = wl_glob_27;
    DMACostModel<DMANNWorkload_NPU27> emptyDMAModel;

    auto dma_cycles = emptyDMAModel.computeCycles(wl);

    // Expect equality.
    EXPECT_EQ(dma_cycles, V(Cycles::ERROR_INFERENCE_NOT_POSSIBLE)) << wl << Cycles::toErrorText(dma_cycles);
}

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModel, SmokeTestDMA_27) {
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

TEST_F(TestDMANNCostModel, SmokeTestDMA_40) {
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

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModel, SmokeTestDMA_RAWNN) {
    DMANNWorkload_NPU27 wl = wl_glob_27;

    const std::string model_path = VPU_DMA_2_7_MODEL_PATH;
    DMACostModel<DMANNWorkload_NPU27> dma_model(model_path);

    ASSERT_TRUE(dma_model.nn_initialized());

    auto dma_raw = dma_model.run_NN(wl);

    // Expect equality.
    EXPECT_NEAR(dma_raw, 0.42211675643920898F, 0.01) << wl;  // gear4:0.347663F
}

// Demonstrate some basic assertions.
TEST_F(TestDMANNCostModel, InitAspects) {
    {  // 27
        const std::string model_path = VPU_DMA_2_7_MODEL_PATH;
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(model_path));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model(model_path);
        EXPECT_TRUE(vpunn_model.nn_initialized());

        const auto file_content{read_a_file(model_path)};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), true));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf(file_content.data(), file_content.size(), true);
        EXPECT_TRUE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), false));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_TRUE(vpunn_model_buf_copy.nn_initialized());
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(model_path));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model(model_path);
        EXPECT_FALSE(vpunn_model.nn_initialized());

        const decltype(read_a_file("")) file_content{'M', 'u', 's', 't', 'h', 'a', 'v', 'e', ' ', '0', '1'};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), true));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf(file_content.data(), file_content.size(), true);
        EXPECT_FALSE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(DMACostModel<DMANNWorkload_NPU27> x(file_content.data(), file_content.size(), false));
        DMACostModel<DMANNWorkload_NPU27> vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_FALSE(vpunn_model_buf_copy.nn_initialized());

        auto cycles_27 = vpunn_model_buf.computeCycles(wl_glob_27);

        EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_27));

        EXPECT_EQ(cycles_27, V(Cycles::ERROR_INFERENCE_NOT_POSSIBLE))
                << wl_glob_27 << Cycles::toErrorText(cycles_27);  // theoretical values??
    }
}

TEST_F(TestDMANNCostModel, Mock_40_vs_VPU27_DPU) {
    {  // 27 and 40
        DMACostModel<DMANNWorkload_NPU27> model_2_7{VPU_DMA_2_7_MODEL_PATH};
        EXPECT_TRUE(model_2_7.nn_initialized());
        DMACostModel<DMANNWorkload_NPU27> model_4_0M{VPU_DMA_2_7_MODEL_PATH};
        EXPECT_TRUE(model_4_0M.nn_initialized());

        auto cycles_27 = model_2_7.computeCycles(wl_glob_27);
        auto cycles_40 = model_4_0M.computeCycles(wl_glob_40M);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_27));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_40));

        auto conv_cyc27_40 = (cycles_27 * get_dpu_fclk(wl_glob_40M.device) / get_dpu_fclk(wl_glob_27.device) /
                              2);  // 2 is the speed up factor, 64 instead of 32?
        auto delta = std::abs((int)conv_cyc27_40 - (int)cycles_40);

        EXPECT_LE(delta, 9) << wl_glob_27 << wl_glob_40M << "\n"
                            << cycles_27 << " -> " << cycles_40;  // 2 is rounding errors
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        DMACostModel<DMANNWorkload_NPU27> model_2_7{model_path};
        EXPECT_FALSE(model_2_7.nn_initialized());
        DMACostModel<DMANNWorkload_NPU27> model_4_0M{model_path};
        EXPECT_FALSE(model_4_0M.nn_initialized());

        auto cycles_27 = model_2_7.computeCycles(wl_glob_27);
        auto cycles_40 = model_4_0M.computeCycles(wl_glob_40M);

        EXPECT_EQ(cycles_27, V(Cycles::ERROR_INFERENCE_NOT_POSSIBLE) /*3445*/)
                << wl_glob_27 << Cycles::toErrorText(cycles_27);  // theoretical, but at 1300MHz
        EXPECT_EQ(cycles_40, V(Cycles::ERROR_INFERENCE_NOT_POSSIBLE) /*3445*/)
                << wl_glob_40M << Cycles::toErrorText(cycles_40);  // theoretical, but at 1700MHz
    }
}

TEST_F(TestDMANNCostModel, DISABLED_SweepDMATime_27) {
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

TEST_F(TestDMANNCostModel, SweepGT_DMATime_27) {
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
TEST_F(TestDMANNCostModel, SweepGT_DMATime_27_GEAR4) {
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

TEST_F(TestDMANNCostModel, SweepGT_DMATime_40) {
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
        const DMANNWorkload_NPU40 dmaNN = DMAWorkloadTransformer::create_NPU40_workload(dmaOld_);
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
TEST_F(TestDMANNCostModel, SweepGT_DMATime_40_Theoretical) {
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

class TestDMA_TH_CostModel : public ::testing::Test {
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
    //   VPUNN::DPUWorkload wl_glob_20;
    DMANNWorkload_NPU27 wl_glob_40M{wl_glob_27};

    DMACostModel<DMANNWorkload_NPU27> model{};
    // DMACostModel specialEmptyDMAModel;

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

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    class TestCase {
    public:
        DMAWorkload t_in;              // wl
        CyclesInterfaceType t_exp{0};  // expected time (GT)
        std::string t_name{""};        // name of the test
    };

    auto mkwl(const int bytes, MemoryLocation in_loc, MemoryLocation out_loc, VPUDevice device) const {
        return mkwl_(bytes, bytes, DataType::UINT8, DataType::UINT8, Layout::ZXY, Layout::ZXY, in_loc, out_loc, device);
    };

    DMAWorkload mkwl_compr(const int bytes_src, const int bytes_dst, MemoryLocation in_loc, MemoryLocation out_loc,
                           VPUDevice device) const {
        return mkwl_(bytes_src, bytes_dst, DataType::UINT8, DataType::UINT8, Layout::ZXY, Layout::ZXY, in_loc, out_loc,
                     device);
    };

    DMAWorkload mkwl_(const int elm_src, const int elm_dst,              // elm
                      const DataType src_type, const DataType dst_type,  // type
                      const Layout src_layout, const Layout dst_layout,  // layout
                      MemoryLocation in_loc, MemoryLocation out_loc, VPUDevice device) const {
        const DMAWorkload dmaOld_{
                device,                                               // device
                {VPUTensor(elm_src, 1, 1, 1, src_type, src_layout)},  // input dimensions WHCB
                {VPUTensor(elm_dst, 1, 1, 1, dst_type, dst_layout)},  // output dimensions
                in_loc,                                               // src
                out_loc,                                              // dst
                1,                                                    // owt
        };
        return dmaOld_;
    };

    const DataType dt{DataType::UINT8};
    const MemoryLocation DRAM{MemoryLocation::DRAM};
    const MemoryLocation CMX{MemoryLocation::CMX};

    float computeMicroseconds(CyclesInterfaceType dpuCycles, const int frequencyMHz) const {
        return dpuCycles * (1.0f / frequencyMHz);
    }
    float computeMicroseconds(CyclesInterfaceType dpuCycles, const VPUDevice device) const {
        return dpuCycles * (1.0f / get_dpu_fclk(device));
    }

private:
};

TEST_F(TestDMA_TH_CostModel, DMA_Theoretical_experiments) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    VPUDevice device{VPUDevice::VPU_2_7};

    auto wl_8kddr = mkwl(8192, MemoryLocation::DRAM, MemoryLocation::CMX, device);

    auto dma_8kddr = cm.DMA(wl_8kddr);

    // Expect equality.
    EXPECT_EQ(dma_8kddr, 1637) << wl_8kddr;  //
}

TEST_F(TestDMA_TH_CostModel, DMA_Theoretical_regresion_NPU27) {
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

    for (auto t : tc) {
        check(t);
    }
}
// this test is for New/Updated theoretical model
TEST_F(TestDMA_TH_CostModel, DMA_Theoretical_regresion_NPU40) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    VPUDevice device{VPUDevice::VPU_4_0};

    const std::vector<TestCase> tc{
            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 539, "1k DC"},    //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 567, "2k DC"},    //
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::CMX, device), 784, "10k DC"},  //

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 539, "1k CD"},    //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 567, "2k CD"},    //
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::DRAM, device), 784, "10k CD"},  //

            {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 85, "1k CC"},     //
            {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113, "2k CC"},    //
            {mkwl(10000, MemoryLocation::CMX, MemoryLocation::CMX, device), 330, "10k CC"},  //

            {mkwl(1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 539, "1k DD"},    //
            {mkwl(2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 567, "2k DD"},    //
            {mkwl(10000, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 784, "10k DD"},  //

            // compressed
            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::CMX, device), 539, "1kto2k DC"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::CMX, device), 567, "2kto1k DC"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::DRAM, device), 567, "1kto2k CD"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::DRAM, device), 567, "2kto1k CD"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113, "1kto2k CC"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 113, "2kto1k CC"},  //

            {mkwl_compr(1024, 2048, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 567, "1kto2k  DD"},  //
            {mkwl_compr(2048, 1024, MemoryLocation::DRAM, MemoryLocation::DRAM, device), 567, "2kto1k DD"},   //

            // permute
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, CMX, device), 539, "1k DCperm"},    //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, DRAM, device), 539, "1k CD perm"},   //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, CMX, CMX, device), 1849, "1k CC perm"},   //
            {mkwl_(1024, 1024, dt, dt, Layout::ZXY, Layout::XYZ, DRAM, DRAM, device), 539, "1k DD perm"},  //

    };

    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        if constexpr (!PerformanceMode::forceLegacy_G4) {
            EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
        } else {
            EXPECT_GT(dma_cyc, 0) << tc.t_name << "\n" << tc.t_in;
        }
    };

    for (auto t : tc) {
        check(t);
    }
}

TEST_F(TestDMA_TH_CostModel, DMA_Theoretical_Debug) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        const VPUDevice d{tc.t_in.device};
        const auto exp_dpu{tc.t_exp};
        const auto f_dpu{get_dpu_fclk(d)};
        const auto f_cmx{get_cmx_fclk(d)};
        const float exp_CMX{(float)exp_dpu * f_cmx / f_dpu};
        const float dma_CMX_cyc{(float)dma_cyc * f_cmx / f_dpu};

        std::cout << "\n"
                  << tc.t_name << ",\t"                                            //
                  << "DPU frq: " << f_dpu << " , CMX frq: " << f_cmx << " MHz.  "  //
                  << "*** Expecting: DPUCyc " << exp_dpu << ", microsec :" << computeMicroseconds(exp_dpu, d)
                  << " VPU cyc: " << exp_CMX
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(exp_CMX), f_cmx)
                  << " ----->  Obtained: DPUCyc : " << dma_cyc << ", microsec :" << computeMicroseconds(dma_cyc, d)
                  << " VPU cyc: " << dma_CMX_cyc
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(dma_CMX_cyc), f_cmx);
        ;
        EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
    };

    {
        std::cout << "\n\n NPU40 \n";
        VPUDevice device{VPUDevice::VPU_4_0};

        const std::vector<TestCase> tc_40{

                {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 85, "1k CC"},   //
                {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113, "2k CC"},  //

                // compressed
                {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 113, "1kto2k CC"},  //
                {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 113, "2kto1k CC"},  //

                // zero
                {mkwl(0, DRAM, CMX, device), 510, "\nzero DC"},  //
                {mkwl(0, DRAM, DRAM, device), 510, "zero DD"},
                {mkwl(0, CMX, CMX, device), 56, "zero CC"},
                {mkwl(0, CMX, DRAM, device), 510, "zero CD"},

                // examples
                {mkwl(8192, DRAM, CMX, device), 735, "\n8k DC"},     // 0.572 expected at 800MHz VPU
                {mkwl(8192 * 2, CMX, DRAM, device), 959, "16k CD"},  // 0.468  expected at 800MHz VPU

                {mkwl(3 * 512 * 172 * 2, DRAM, CMX, device), 14965, "\n500K theory DC"},  //
                {mkwl(3 * 512 * 256 * 2, DRAM, CMX, device), 22024,
                 "\n786K real DC"},  // 10.2, and 13.8 exp at 800MHz VPU

        };

        for (auto t : tc_40) {
            if constexpr (!PerformanceMode::forceLegacy_G4) {
                check(t);
            }
        }
    }

    {
        std::cout << "\n\n NPU27\n";
        VPUDevice device{VPUDevice::VPU_2_7};

        const std::vector<TestCase> tc_27{

                {mkwl(1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "1k CC"},  //
                {mkwl(2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 192, "2k CC"},  //

                // compressed
                {mkwl_compr(1024, 2048, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "1kto2k CC"},  //
                {mkwl_compr(2048, 1024, MemoryLocation::CMX, MemoryLocation::CMX, device), 107, "2kto1k CC"},  //

                // zero
                {mkwl(0, DRAM, CMX, device), 1242, "\nzero DC"},  //
                {mkwl(0, DRAM, DRAM, device), 1242, "zero DD"},
                {mkwl(0, CMX, CMX, device), 21, "zero CC"},
                {mkwl(0, CMX, DRAM, device), 1242, "zero CD"},

                // examples
                {mkwl(8192, DRAM, CMX, device), 1637, "\n8k DC"},     //
                {mkwl(8192 * 2, CMX, DRAM, device), 2031, "16k CD"},  //

                {mkwl(3 * 512 * 172 * 2, DRAM, CMX, device), 26683, "\n500K theory DC"},  //
                {mkwl(3 * 512 * 256 * 2, DRAM, CMX, device), 39108, "\n786K real DC"},    //

        };

        for (auto t : tc_27) {
            check(t);
        }
    }

    // EXPECT_TRUE(false);
}

TEST_F(TestDMA_TH_CostModel, DMA_Th_Smoke_E162767) {
    VPUCostModel cm("empty");
    ASSERT_TRUE(!cm.nn_initialized());
    auto check = [&](const TestCase& tc) {
        auto dma_cyc = cm.DMA(tc.t_in);
        const VPUDevice d{tc.t_in.device};
        const auto exp_dpu{tc.t_exp};
        const auto f_dpu{get_dpu_fclk(d)};
        const auto f_cmx{get_cmx_fclk(d)};
        const float exp_CMX{(float)exp_dpu * f_cmx / f_dpu};
        const float dma_CMX_cyc{(float)dma_cyc * f_cmx / f_dpu};

        std::cout << "\n"
                  << tc.t_name << ",\t"                                            //
                  << "DPU frq: " << f_dpu << " , CMX frq: " << f_cmx << " MHz.  "  //
                  << "*** Expecting: DPUCyc " << exp_dpu << ", microsec :" << computeMicroseconds(exp_dpu, d)
                  << " VPU cyc: " << exp_CMX
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(exp_CMX), f_cmx)
                  << " ----->  Obtained: DPUCyc : " << dma_cyc << ", microsec :" << computeMicroseconds(dma_cyc, d)
                  << " VPU cyc: " << dma_CMX_cyc
                  << ", microsec: " << computeMicroseconds((unsigned)std::ceil(dma_CMX_cyc), f_cmx);
        ;
        EXPECT_EQ(dma_cyc, tc.t_exp) << tc.t_name << "\n" << tc.t_in;
    };

    {
        std::cout << "\n\n NPU40 \n";
        const VPUDevice device{VPUDevice::VPU_4_0};

        TestCase case1{mkwl(4864, MemoryLocation::DRAM, MemoryLocation::CMX, device),
                       PerformanceMode::forceLegacy_G4 ? 1891 : 644,
                       "2k CC"};  // original on develop branch the cost is 1891. With vpucostmodel updated, the cost
                                  // changes to 644
        check(case1);

        const DMAWorkload wl{case1.t_in};
        auto dma_now = cm.DMA(wl);
        auto dma_n = cm.DMATheoreticalCycles_RESERVED_ON(wl);
        auto dma_o = cm.DMATheoreticalCyclesLegacyLNL(wl);

        EXPECT_EQ(dma_now, PerformanceMode::forceLegacy_G4 ? 1891 : 644);
        EXPECT_EQ(dma_n, 644);
        EXPECT_EQ(dma_o, 1891);
    }

    // EXPECT_TRUE(false);
}

}  // namespace VPUNN_unit_tests