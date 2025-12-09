// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common/common_helpers.h"

#include <optional>
#include <variant>

namespace VPUNN_unit_tests {
using namespace VPUNN;

/// compare CLustering versus SOH attributes and FUll Workloads vs halved workloads with SOH
class TestSplitMethodsComparisonsVPU2x : public ::testing::Test {
public:
protected:
    VPUNN::VPUCostModel model_2_7{VPU_2_7_MODEL_PATH};
    std::string info{};

    float strategy_scale = 1.0f;  // adjust SOK/SOH valued by this

    float tolerance_even = 0.2f;  // used for SOH
    // float tolerance_odd = 0.3f;
    float tolerance_SOK = 0.2f;

    void SetUp() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::activate2ndlog();
    }
    void TearDown() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::deactivate2ndlog();
    }

    void check_for_noError(CyclesInterfaceType cost_cyc, const DPUWorkload& wl, const std::string& t_header = "xxx") {
        EXPECT_FALSE(Cycles::isErrorCode(cost_cyc))
                << t_header << " > Unexpected ERROR code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                << "\n " << wl << Logger::get2ndlog();
        Logger::clear2ndlog();
    }

    void check_Cluster_vs_SOH(const DPUWorkload& wl_base, const std::string& t_header) {
        DPUWorkload wl{wl_base};
        std::cout << "\n****** TEST : " << t_header << "\n";
        wl.isi_strategy = ISIStrategy::CLUSTERING;
        auto cycles_clu = model_2_7.DPU(wl, info);
        check_for_noError(cycles_clu, wl, "CLUSTERING" + t_header + info);

        wl.isi_strategy = ISIStrategy::SPLIT_OVER_H;
        auto cycles_soh = model_2_7.DPU(wl, info);
        check_for_noError(cycles_soh, wl, "SOH" + t_header + info);

        cycles_soh = static_cast<decltype(cycles_soh)>(cycles_soh * strategy_scale);

        const std::int32_t delta = std::abs((std::int32_t)cycles_clu - (std::int32_t)cycles_soh);
        const std::int32_t maxDelta = (std::int32_t)(cycles_clu * tolerance_even);

        const auto deltapercent = ((float)delta / cycles_clu) * 100;

        // EXPECT_EQ(cycles_clu, cycles_soh) << t_header << " Cost not similar enough.\n"
        //                                   << "cost clustering: " << cycles_clu << "\n"
        //                                   << "cost soh       : " << cycles_soh << "\n"
        //                                   << wl_base;
        EXPECT_LE(delta, maxDelta) << t_header << " Cost not similar enough.\n"
                                   << "cost clustering: " << cycles_clu << "\n"
                                   << "cost soh       : " << cycles_soh << "\n"
                                   << "delta%         : " << deltapercent << "\n"
                                   << wl_base;
        std::cout << "\n--------- END TEST : " << t_header << " cost clustering: " << cycles_clu
                  << ", cost soh: " << cycles_soh << ", delta%: " << (int)(deltapercent) << " % \n";
    }

    void check_Cluster_vs_SOK(const DPUWorkload& wl_base, const std::string& t_header) {
        DPUWorkload wl{wl_base};
        std::cout << "\n****** TEST : " << t_header << "\n";
        wl.isi_strategy = ISIStrategy::CLUSTERING;
        auto cycles_clu = model_2_7.DPU(wl, info);
        check_for_noError(cycles_clu, wl, "CLUSTERING" + t_header + info);

        wl.isi_strategy = ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;
        auto cycles_soh = model_2_7.DPU(wl, info);
        check_for_noError(cycles_soh, wl, "SOK" + t_header + info);

        cycles_soh = static_cast<decltype(cycles_soh)>(cycles_soh * strategy_scale);

        const std::int32_t delta = std::abs((std::int32_t)cycles_clu - (std::int32_t)cycles_soh);
        const std::int32_t maxDelta = static_cast<std::int32_t>(cycles_clu * tolerance_SOK);

        const auto deltapercent = ((float)delta / cycles_clu) * 100;

        // EXPECT_EQ(cycles_clu, cycles_soh) << t_header << " Cost not similar enough.\n"
        //                                   << "cost clustering: " << cycles_clu << "\n"
        //                                   << "cost soh       : " << cycles_soh << "\n"
        //                                   << wl_base;
        EXPECT_LE(delta, maxDelta) << t_header << " Cost not similar enough.\n"
                                   << "cost clustering: " << cycles_clu << "\n"
                                   << "cost sok       : " << cycles_soh << "\n"
                                   << "delta%         : " << deltapercent << "\n"
                                   << wl_base;
        std::cout << "\n--------- END TEST : " << t_header << " cost clustering: " << cycles_clu
                  << ", cost sok: " << cycles_soh << ", delta%: " << (int)(deltapercent) << " % \n";
    }
};
TEST_F(TestSplitMethodsComparisonsVPU2x, Convolution_3x3) {
    const DPUWorkload tst_refH16{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH20{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH38{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    {
        std::string tst = " 3x3H16 ";
        DPUWorkload wl{std::move(tst_refH16)};
        check_Cluster_vs_SOH(wl, tst);
    }

    {
        std::string tst = " 3x3H19 ";
        DPUWorkload wl{std::move(tst_refH19)};
        check_Cluster_vs_SOH(wl, tst);
    }
    check_Cluster_vs_SOH(tst_refH38, " 3x3H38 ");
    check_Cluster_vs_SOH(tst_refH20, " 3x3H20 ");
}
TEST_F(TestSplitMethodsComparisonsVPU2x, Convolution_5x5) {
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 5},                                               // kernels
            {1, 1},                                               // strides
            {2, 2, 2, 2},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH20{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 5},                                               // kernels
            {1, 1},                                               // strides
            {2, 2, 2, 2},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH40{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 5},                                               // kernels
            {1, 1},                                               // strides
            {2, 2, 2, 2},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    {
        std::string tst = " 5x5H19 ";
        check_Cluster_vs_SOH(tst_refH19, tst);
    }

    {
        std::string tst = " 5x5H20 ";
        check_Cluster_vs_SOH(tst_refH20, tst);
    }

    {
        std::string tst = " 5x5H40 ";
        check_Cluster_vs_SOH(tst_refH40, tst);
    }
}
TEST_F(TestSplitMethodsComparisonsVPU2x, Convolution_1x1) {
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                               // kernels
            {1, 1},                                               // strides
            {0, 0, 0, 0},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH20{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                               // kernels
            {1, 1},                                               // strides
            {0, 0, 0, 0},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH40{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                               // kernels
            {1, 1},                                               // strides
            {0, 0, 0, 0},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    // EXPECT_TRUE(false);

    {
        std::string tst = " 1x1H19 ";
        check_Cluster_vs_SOH(tst_refH19, tst);
    }

    {
        std::string tst = " 1x1H20 ";
        check_Cluster_vs_SOH(tst_refH20, tst);
    }

    {
        std::string tst = " 1x1H40 ";
        check_Cluster_vs_SOH(tst_refH40, tst);
    }
}
TEST_F(TestSplitMethodsComparisonsVPU2x, Convolution_3x3_SOK) {
    const DPUWorkload tst_refH16{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH38{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH38C64{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {1, 1, 1, 1},                                        // padding
            ExecutionMode::CUBOID_16x16,                         // execution mode
    };

    const DPUWorkload tst_refH38C256{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    check_Cluster_vs_SOK(tst_refH16, " 3x3H16 ");
    check_Cluster_vs_SOK(tst_refH19, " 3x3H19 ");
    check_Cluster_vs_SOK(tst_refH38, " 3x3H38C128 ");

    check_Cluster_vs_SOK(tst_refH38C64, " 3x3H38C64 ");
    check_Cluster_vs_SOK(tst_refH38C256, " 3x3H38C256 ");
}
TEST_F(TestSplitMethodsComparisonsVPU2x, Convolution_11x11) {
    const DPUWorkload tst_refH49{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 49, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 49, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {11, 11},                                            // kernels
            {1, 1},                                              // strides
            {5, 5, 5, 5},                                        // padding
            ExecutionMode::CUBOID_16x16,                         // execution mode
    };
    const DPUWorkload tst_refH50{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 50, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 50, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {11, 11},                                            // kernels
            {1, 1},                                              // strides
            {5, 5, 5, 5},                                        // padding
            ExecutionMode::CUBOID_16x16,                         // execution mode
    };
    const DPUWorkload tst_refH100{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 100, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 100, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {11, 11},                                             // kernels
            {1, 1},                                               // strides
            {5, 5, 5, 5},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    check_Cluster_vs_SOH(tst_refH49, " 11x11 H 49");
    check_Cluster_vs_SOH(tst_refH50, " 11x11 H 50");
    check_Cluster_vs_SOH(tst_refH100, " 11x11 H 100");
}
}  // namespace VPUNN_unit_tests