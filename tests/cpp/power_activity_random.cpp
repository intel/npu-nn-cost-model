// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the ?Software Package?)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the ?third-party-programs.txt? or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "core/logger.h"
#include "vpu/power.h"
#include "vpu/types.h"

#include "common_helpers.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class ActivityFactor : public testing::Test {
public:
protected:
    void SetUp() override {
    }

    VPUNN::VPUCostModel model{VPU_2_7_MODEL_PATH};
    const IEnergy& my_energy = model.getEnergyInterface();
    const HWPerformanceModel& performance{model.getPerformanceModel()};
    const DPUTheoreticalCostProvider dpu_theoretical{performance};
    // VPU_2_7_MODEL_PATH
    // VPUNNModelsFiles::getModels().fast_model_paths[1].first

    const float scale{1.0f};  ///< this can be used to reduce the tolerance interval towards zero(not all tests) DEBUG

    // absolute values
    const float norm_tol{0.05f};     // normal tolerance 5%
    const float weak_tol{0.10f};     // reduced/weak tolerance 10%
    const float strict_tol{0.005f};  // stricter tolerance 0.5%

    VPUNN::DPUWorkload generate_helper_layer(const unsigned int dim, const unsigned int ic, const unsigned int oc,
                                             VPUNN::Operation operation, VPUNN::VPUDevice device,
                                             VPUNN::DataType dtype) {
        VPUNN::DPUWorkload wl;
        wl.device = device;
        wl.op = operation;
        wl.execution_order = VPUNN::ExecutionMode::CUBOID_8x16;   // execution mode
        wl.inputs = {VPUNN::VPUTensor(dim, dim, ic, 1, dtype)};   // input dimensions
        wl.outputs = {VPUNN::VPUTensor(dim, dim, oc, 1, dtype)};  // output dimensions
        wl.kernels = {1, 1};                                      // kernels
        wl.strides = {1, 1};                                      // strides
        wl.padding = {0, 0, 0, 0};                                // padding
        wl.input_swizzling[0] = swz_def;
        wl.input_swizzling[1] = swz_def;
        wl.output_swizzling[0] = swz_def;
        return wl;
    }

    static float convertToFP16_powerVirus_27(float c) {
        const float FPOverInt_power_ratio_27{1.3f};
        return c / FPOverInt_power_ratio_27;
    }
    const float convToReferenceVirus{
            1.3f /*VPUPowerFactorLUT().getFP_overI8_maxPower_ratio(VPUDevice::VPU_2_7)*/};  // put by hand

    // the code below might be part of a base class (helper) (refactoring)
    const bool dictON{false};  ///< if true will print the Dict representation of WL
    std::string toDict(const DPUWorkload& wl) const {
        return (dictON ? WLHelp::toDictString(wl) : "\n");
    }
};

TEST_F(ActivityFactor, TestPowerActivityFactorU8Conv) {
    // Workloads with HxW 16x16 were profiled for a small sweep of input channels
    {
        const std::vector<unsigned int> ic = {16, 32, 64, 128, 256, 512, 1024, 2048};

        // Expected power activity factors based on implementation of LUT against profiled workloads.
        // The expect values are check with relaxed tolerance for input channels (HxW 16x16 again) were profiled but are
        // not explicit entries on the LUT . This is a weak test for the LUT entry interpolation functionality
        std::vector<float> expected{0.283f, 0.552f, 0.787f, 0.88f, 0.93f, 0.96f, 0.98f, 0.99f};  // GT based

        // std::transform(expected.cbegin(), expected.cend(),
        //                expected.begin(),  // write to the same location
        //                convertToFP16_powerVirus_27);

        const std::vector<float> tolerance{norm_tol,        norm_tol,        weak_tol, weak_tol, weak_tol,
                                           weak_tol * 1.2f, weak_tol * 1.1f, norm_tol};  // tolerances, @ todo restrict
        const std::vector<unsigned int> exp_theorCyc{512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        const std::vector<unsigned int> exp_idealCyc{512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        const std::vector<unsigned int> exp_GrndTCyc{1807, 1854, 2601, 4644, 8747, 16935, 33319, 66093};  // v16

        ASSERT_TRUE(ic.size() == expected.size());
        ASSERT_TRUE(ic.size() == tolerance.size());
        ASSERT_TRUE(ic.size() == exp_theorCyc.size());
        ASSERT_TRUE(ic.size() == exp_idealCyc.size());

        auto j = 0;
        for (auto i = ic.begin(); i != ic.end(); ++i, ++j) {
            auto wl = generate_helper_layer(16, *i, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                            VPUNN::DataType::UINT8);
            EXPECT_NEAR(my_energy.DPUActivityFactor(wl), expected[j] /** convToReferenceVirus*/, tolerance[j] * scale)
                    << "ic=" << *i << " NN cyc:" << model.DPU(wl)
                    << " , ThCyc: " << dpu_theoretical.DPUTheoreticalCycles(wl)
                    << " , GTcyc:" << exp_GrndTCyc[j] << " , Power Ideal Cyc: " << performance.DPU_Power_IdealCycles(wl)
                    << " , Efficiency Ideal Cyc: " << performance.DPU_Efficency_IdealCycles(wl) << " , ExpAF:" << expected[j]
                    << toDict(wl) << wl;

            EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), exp_theorCyc[j]);
            EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), exp_idealCyc[j]);
        }
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorFPConv) {
    // Workloads with HxW 16x16 were profiled for a tiny sweep of input channels
    const std::vector<unsigned int> ic = {64, 256};

    // Expected power activity factors based on implementation of LUT against profiled workloads
    const std::vector<float> expected{0.649f * convToReferenceVirus,
                                      0.916f * convToReferenceVirus};  // LIMITATION? no, NN values are larger
    const std::vector<float> tolerance{norm_tol, norm_tol};            // tolerances
    const std::vector<unsigned int> exp_theorCyc{4096, 16384};
    const std::vector<unsigned int> exp_idealCyc{4096, 16384};
    const std::vector<unsigned int> exp_GrndTCyc{
            6306, 17884};  // v17{6455 (0.63),17758 (0.92)} v16:{6681 (0.61), 18682 (0.87)}

    ASSERT_TRUE(ic.size() == expected.size());
    ASSERT_TRUE(ic.size() == tolerance.size());
    ASSERT_TRUE(ic.size() == exp_theorCyc.size());
    ASSERT_TRUE(ic.size() == exp_idealCyc.size());

    auto j = 0;
    for (auto i = ic.begin(); i != ic.end(); ++i, ++j) {
        auto wl = generate_helper_layer(16, *i, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                        VPUNN::DataType::FLOAT16);
        EXPECT_NEAR(my_energy.DPUActivityFactor(wl), expected[j], tolerance[j] * scale)
                << "ic=" << *i << " NN cyc:" << model.DPU(wl) << ", ThCyc: " << dpu_theoretical.DPUTheoreticalCycles(wl)
                << " , GTcyc:" << exp_GrndTCyc[j] << ", Power Ideal Cyc: " << performance.DPU_Power_IdealCycles(wl)
                << ", Efficiency Ideal Cyc: " << performance.DPU_Efficency_IdealCycles(wl) << " , ExpAF:" << expected[j]
                << toDict(wl) << wl;

        EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), exp_theorCyc[j]);
        EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), exp_idealCyc[j]);
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorConv576) {
    // IC = 576 not in LUT, results interpolated, expected values derived empirically
    const std::vector<VPUNN::DataType> dtypes = {VPUNN::DataType::FLOAT16, VPUNN::DataType::UINT8};
    const std::vector<float> expected = {0.9884f * convToReferenceVirus, 0.9884f};  //{0.98f, 0.75f}; // GT based
    const std::vector<unsigned int> exp_theorCyc{82944, 41472};
    const std::vector<unsigned int> exp_idealCyc{82944, 41472};
    const std::vector<unsigned int> exp_GrndTCyc{83917, 41955};  // v16

    ASSERT_TRUE(dtypes.size() == expected.size());
    ASSERT_TRUE(dtypes.size() == exp_theorCyc.size());
    ASSERT_TRUE(dtypes.size() == exp_idealCyc.size());

    auto j{0};
    for (auto i = dtypes.begin(); i != dtypes.end(); ++i, ++j) {
        auto wl = generate_helper_layer(16, 576, 576, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7, *i);
        EXPECT_NEAR(my_energy.DPUActivityFactor(wl), expected[j], weak_tol * scale)  //@todo make not so weak
                << "dtype=" << (int)*i << " NN cyc:" << model.DPU(wl)
                << " , ThCyc: " << dpu_theoretical.DPUTheoreticalCycles(wl)
                << " , GTcyc:" << exp_GrndTCyc[j] << " , Power Ideal Cyc: " << performance.DPU_Power_IdealCycles(wl)
                << " , Efficiency Ideal Cyc: " << performance.DPU_Efficency_IdealCycles(wl) << " , ExpAF:" << expected[j]
                << toDict(wl) << wl;

        EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), exp_theorCyc[j]);
        EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), exp_idealCyc[j]);
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorConvMisc) {
    // Sweep over some tensor dimensions which aren't in the profiled set
    std::list<unsigned int> dims = {4, 8, 16};
    std::list<unsigned int> ic = {112, 288, 400};
    std::list<VPUNN::DataType> dtype = {VPUNN::DataType::UINT8, VPUNN::DataType::FLOAT16};

    auto i = dims.begin();
    auto j = ic.begin();
    auto k = dtype.begin();

    for (; i != dims.end(); ++i) {
        for (; j != ic.end(); ++j) {
            for (; k != dtype.end(); ++k) {
                auto wl = generate_helper_layer(*i, *j, 256, VPUNN::Operation::CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                                *k);
                float af = my_energy.DPUActivityFactor(wl);
                // Just range check
                EXPECT_GT(af, 0.2f);
                EXPECT_LT(af, 0.9f);
            }
        }
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorDwConv) {
    // Only one DW workload profiled 64x28x28x64 3x3
    VPUNN::DPUWorkload wl;

    {
        wl = generate_helper_layer(28, 64, 64, VPUNN::Operation::DW_CONVOLUTION, VPUNN::VPUDevice::VPU_2_7,
                                   VPUNN::DataType::UINT8);
        wl.kernels = {3, 3};
        wl.padding = {1, 1, 1, 1};
        EXPECT_NEAR(my_energy.DPUActivityFactor(wl), convertToFP16_powerVirus_27(0.451f) * convToReferenceVirus,
                    weak_tol * scale)
                << " NN cyc:" << model.DPU(wl) << " ThCyc: " << dpu_theoretical.DPUTheoreticalCycles(wl)
                << " Power Ideal Cyc: " << performance.DPU_Power_IdealCycles(wl) << toDict(wl) << wl;
        EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), 4374);
        EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), 221);
    }
    {  // Only one AVEPOOL (which is implemented as DW) workload profiled 2048x7x7x2048 7x7
        wl = generate_helper_layer(7, 64, 64,  // Max workload size is 64
                                   VPUNN::Operation::AVEPOOL, VPUNN::VPUDevice::VPU_2_7, VPUNN::DataType::UINT8);
        // Override specific fields to correctly configure AVEPOOL workload
        wl.kernels = {7, 7};
        wl.outputs = {VPUNN::VPUTensor(1, 1, 64, 1, VPUNN::DataType::UINT8)};
        EXPECT_NEAR(my_energy.DPUActivityFactor(wl), convertToFP16_powerVirus_27(0.025f) * convToReferenceVirus,
                    strict_tol * scale)
                << " NN cyc:" << model.DPU(wl) << " ThCyc: " << dpu_theoretical.DPUTheoreticalCycles(wl)
                << " Power Cyc: " << performance.DPU_Power_IdealCycles(wl) << toDict(wl) << wl;
        EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), 1195);
        EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), 2);
    }
}

TEST_F(ActivityFactor, TestPowerActivityFactorEltWise) {
    // Only one ELTWISE workload profiled 256x56x56x256
    VPUNN::DPUWorkload wl;

    wl = generate_helper_layer(56, 256, 256, VPUNN::Operation::ELTWISE, VPUNN::VPUDevice::VPU_2_7,
                               VPUNN::DataType::UINT8);

    EXPECT_NEAR(my_energy.DPUActivityFactor(wl), convertToFP16_powerVirus_27(0.084f) * convToReferenceVirus,
                strict_tol * scale)
            << " NN cyc:" << model.DPU(wl) << " ThCyc: " << dpu_theoretical.DPUTheoreticalCycles(wl)
            << " Power Ideal Cyc: " << performance.DPU_Power_IdealCycles(wl) << toDict(wl);
    EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), 20160);
    EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), 392);
}

TEST_F(ActivityFactor, TestPowerActivityFactorMaxPool) {
    // Only one MAXPOOL workload profiled 64x28x28x64 3x3
    VPUNN::DPUWorkload wl;

    wl = generate_helper_layer(28, 64, 64, VPUNN::Operation::MAXPOOL, VPUNN::VPUDevice::VPU_2_7,
                               VPUNN::DataType::UINT8);

    // Override specific fields to correctly configure AVEPOOL workload
    wl.kernels = {3, 3};
    wl.padding = {1, 1, 1, 1};
    wl.strides = {2, 2};
    wl.outputs = {VPUNN::VPUTensor(14, 14, 64, 1, VPUNN::DataType::UINT8)};
    EXPECT_NEAR(my_energy.DPUActivityFactor(wl), 0.20f, norm_tol * scale)
            << " NN cyc:" << model.DPU(wl) << " , ThCyc: "
            << dpu_theoretical.DPUTheoreticalCycles(wl)
            /* << " , GTcyc:" << exp_GrndTCyc[j]*/
            << " , Power Ideal Cyc: " << performance.DPU_Power_IdealCycles(wl)
            << " , Efficiency Ideal Cyc: " << performance.DPU_Efficency_IdealCycles(wl) << toDict(wl) << wl;

    EXPECT_EQ(dpu_theoretical.DPUTheoreticalCycles(wl), 1094);
    EXPECT_EQ(performance.DPU_Power_IdealCycles(wl), 56);
}
}  // namespace VPUNN_unit_tests
