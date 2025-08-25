// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/power.h"
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/cycles_interface_types.h"

#include <algorithm>
#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestEnergyandPF_CostModel : public ::testing::Test /* TestCostModel*/ {
public:
protected:
    DPUWorkload wl_conv{VPUDevice::VPU_2_7,
                        Operation::CONVOLUTION,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM{VPUDevice::VPU_2_7,
                          Operation::CM_CONVOLUTION,
                          {VPUTensor(56, 56, 15, 1, DataType::UINT8)},  // input dimensions
                          {VPUTensor(56, 56, 32, 1, DataType::UINT8)},  // output dimensions
                          {3, 3},                                       // kernels
                          {1, 1},                                       // strides
                          {1, 1, 1, 1},                                 // padding
                          VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP{VPUDevice::VPU_2_7,
                        Operation::MAXPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP{VPUDevice::VPU_2_7,
                        Operation::AVEPOOL,
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                        {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                        {3, 3},                                       // kernels
                        {1, 1},                                       // strides
                        {1, 1, 1, 1},                                 // padding
                        VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv{VPUDevice::VPU_2_7,
                           Operation::DW_CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                           {3, 3},                                       // kernels
                           {1, 1},                                       // strides
                           {1, 1, 1, 1},                                 // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT{VPUDevice::VPU_2_7,
                       Operation::ELTWISE,
                       {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
                       {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
                       {1, 1},                                       // kernels
                       {1, 1},                                       // strides
                       {0, 0, 0, 0},                                 // padding
                       VPUNN::ExecutionMode::CUBOID_16x16};
    const std::vector<DPUWorkload> wl_list{wl_conv, wl_convCM, wl_MAXP, wl_AVGP, wl_DW_conv, wl_ELT};

    DPUWorkload wl_conv_FP{VPUDevice::VPU_2_7,
                           Operation::CONVOLUTION,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 32, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_convCM_FP{VPUDevice::VPU_2_7,
                             Operation::CM_CONVOLUTION,
                             {VPUTensor(56, 56, 15, 1, DataType::FLOAT16)},  // input dimensions
                             {VPUTensor(56, 56, 32, 1, DataType::FLOAT16)},  // output dimensions
                             {3, 3},                                         // kernels
                             {1, 1},                                         // strides
                             {1, 1, 1, 1},                                   // padding
                             VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_MAXP_FP{VPUDevice::VPU_2_7,
                           Operation::MAXPOOL,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_AVGP_FP{VPUDevice::VPU_2_7,
                           Operation::AVEPOOL,
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                           {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                           {3, 3},                                         // kernels
                           {1, 1},                                         // strides
                           {1, 1, 1, 1},                                   // padding
                           VPUNN::ExecutionMode::CUBOID_16x16};
    DPUWorkload wl_DW_conv_FP{VPUDevice::VPU_2_7,
                              Operation::DW_CONVOLUTION,
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                              {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                              {3, 3},                                         // kernels
                              {1, 1},                                         // strides
                              {1, 1, 1, 1},                                   // padding
                              VPUNN::ExecutionMode::CUBOID_16x16};

    DPUWorkload wl_ELT_FP{VPUDevice::VPU_2_7,
                          Operation::ELTWISE,
                          {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // input dimensions
                          {VPUTensor(56, 56, 16, 1, DataType::FLOAT16)},  // output dimensions
                          {1, 1},                                         // kernels
                          {1, 1},                                         // strides
                          {0, 0, 0, 0},                                   // padding
                          VPUNN::ExecutionMode::CUBOID_16x16};
    const std::vector<DPUWorkload> wl_list_FP{wl_conv_FP, wl_convCM_FP,  wl_MAXP_FP,
                                              wl_AVGP_FP, wl_DW_conv_FP, wl_ELT_FP};

    const float w_sparsity_level{0.69f};                                //< to be used for lists
    std::vector<DPUWorkload> wl_list_sparse{wl_conv, wl_ELT};           // only supported
    std::vector<DPUWorkload> wl_list_FP_sparse{wl_conv_FP, wl_ELT_FP};  // only supported

    void SetUp() override {
        // TestCostModel::SetUp();
    }
    TestEnergyandPF_CostModel() {
        auto transformer = [this](DPUWorkload& c)  // modify in-place
        {
            c.weight_sparsity = w_sparsity_level;
            c.weight_sparsity_enabled = true;
        };

        std::for_each(wl_list_sparse.begin(), wl_list_sparse.end(), transformer);
        std::for_each(wl_list_FP_sparse.begin(), wl_list_FP_sparse.end(), transformer);
    }

    void basicTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        const VPUPowerFactorLUT power_factor_lut;
        const float operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl);
        const float exceedMax = power_factor_lut.get_PowerVirus_exceed_factor(wl.device);

        const auto energy = crt_model.DPUEnergy(wl);
        const auto af = crt_model.DPUActivityFactor(wl);
        const auto util = crt_model.hw_utilization(wl);
        const auto util_idealCyc = crt_model.DPU_Power_IdealCycles(wl);
        const auto efficiency_idealCyc = crt_model.DPU_Efficency_IdealCycles(wl);
        const auto theorCyc = crt_model.DPUTheoreticalCycles(wl);
        std::string errInfo;
        const auto nnCyc = crt_model.DPU(wl, errInfo);

        std::stringstream buffer;
        buffer << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op)) << "\n  NN cyc:" << nnCyc
               << ", ThCyc: " << theorCyc << ", Power Ideal Cyc: " << util_idealCyc
               << ", Efficiency Ideal Cyc: " << efficiency_idealCyc << "\n Utilization(ideal/NNcyc): " << util
               << " Energy: " << energy << " powerAF: " << af << " pf oper correlation :" << operation_pf << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(nnCyc)) << info << wl << errInfo << details;

        EXPECT_GT(energy, 0) << details;

        EXPECT_GT(af, 0) << details;
        EXPECT_LE(af, 1.0f * exceedMax) << details;
        EXPECT_GT(util, 0) << details;
        EXPECT_LE(util, 1.0f) << details;

        EXPECT_NEAR(energy, (float)util_idealCyc * operation_pf, 1) << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

    void basicDPUPackEquivalenceTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};
        const VPUPowerFactorLUT power_factor_lut;
        const float operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl);

        const auto energy = crt_model.DPUEnergy(wl);
        const auto af = crt_model.DPUActivityFactor(wl);
        const auto util = crt_model.hw_utilization(wl);
        const auto util_idealCyc = crt_model.DPU_Power_IdealCycles(wl);
        const auto efficiency_idealCyc = crt_model.DPU_Efficency_IdealCycles(wl);
        const auto theorCyc = crt_model.DPUTheoreticalCycles(wl);
        std::string errInfo;
        const auto nnCyc = crt_model.DPU(wl, errInfo);

        std::stringstream buffer;
        buffer << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op)) << "\n  NN cyc:" << nnCyc
               << ", ThCyc: " << theorCyc << ",Power Ideal Cyc: " << util_idealCyc
               << ", Efficiency Ideal Cyc: " << efficiency_idealCyc << "\n Utilization(ideal/NNcyc): " << util
               << " Energy: " << energy << " powerAF: " << af << " pf oper correlation :" << operation_pf << "\n ";
        std::string details = buffer.str();

        DPUWorkload wl_pack{workload};
        const DPUInfoPack allinfo = crt_model.DPUInfo(wl_pack);

        EXPECT_TRUE(!Cycles::isErrorCode(nnCyc)) << info << wl << errInfo << details;

        EXPECT_EQ(allinfo.DPUCycles, nnCyc) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.errInfo, errInfo) << info << wl << errInfo << details << allinfo;

        EXPECT_FLOAT_EQ(allinfo.energy, energy) << info << wl << errInfo << details << allinfo;
        EXPECT_FLOAT_EQ(allinfo.power_activity_factor, af) << info << wl << errInfo << details << allinfo;
        EXPECT_FLOAT_EQ(allinfo.power_mac_utilization, util) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.power_ideal_cycles, util_idealCyc) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.efficiency_ideal_cycles, efficiency_idealCyc) << info << wl << errInfo << details << allinfo;
        EXPECT_EQ(allinfo.hw_theoretical_cycles, theorCyc) << info << wl << errInfo << details << allinfo;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

    void basicSparseTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        const VPUPowerFactorLUT power_factor_lut;
        const float operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl);

        // const auto energy = crt_model.DPUEnergy(wl);
        // const auto af = crt_model.DPUActivityFactor(wl);
        // const auto util = crt_model.hw_utilization(wl);
        // const auto idealCyc = crt_model.DPUIdealCycles(wl);
        // const auto theorCyc = crt_model.DPUTheoreticalCycles(wl);
        // std::string errInfo;
        // const auto nnCyc = crt_model.DPU(wl, errInfo);

        DPUWorkload wl_pack{workload};
        const DPUInfoPack dpu = crt_model.DPUInfo(wl_pack);

        std::stringstream buffer;
        buffer << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op)) << "\n"
               << "W sparsity: " << wl.weight_sparsity << ", Act sparsity: " << wl.act_sparsity << "\t" << dpu
               << " pf oper correlation :" << operation_pf << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(dpu.DPUCycles)) << info << wl << dpu.errInfo << details;

        EXPECT_GT(dpu.energy, 0) << details;

        EXPECT_GT(dpu.power_activity_factor, 0) << details;
        EXPECT_LE(dpu.power_activity_factor, 1.0f) << details;
        EXPECT_GT(dpu.power_mac_utilization, 0) << details;
        EXPECT_LE(dpu.power_mac_utilization, 1.0f) << details;

        EXPECT_NEAR(dpu.energy, (float)dpu.power_ideal_cycles * operation_pf, 1) << details;
        //-----------------
        DPUWorkload wl_pack_denseW{workload};
        wl_pack_denseW.weight_sparsity_enabled = false;
        wl_pack_denseW.weight_sparsity = 0.0f;
        const DPUInfoPack dpuD_W = crt_model.DPUInfo(wl_pack_denseW);
        {
            EXPECT_TRUE(!Cycles::isErrorCode(dpuD_W.DPUCycles)) << info << dpuD_W << dpuD_W.errInfo << details;

            EXPECT_GT(dpuD_W.energy, 0) << details;

            EXPECT_GT(dpuD_W.energy, dpu.energy);  // dense should be higher  (if not much sparse HW overhead)

            // this raises also the problem : is power virus to be done with sparse enabled but no sparsity at all?,
            // or we allow for a potentially >1 Activity factor

            EXPECT_GT(dpuD_W.power_ideal_cycles, dpu.power_ideal_cycles);  // dense should be higher
            EXPECT_EQ(dpuD_W.efficiency_ideal_cycles,
                      dpu.efficiency_ideal_cycles);  // dense should be equal with sparse for efficiency
            EXPECT_EQ(dpuD_W.hw_theoretical_cycles, dpu.hw_theoretical_cycles);  // unimplemented part
        }

        std::cout << details << "\t" << dpuD_W
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

private:
};

TEST_F(TestEnergyandPF_CostModel, BasicEnergy_INT8) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};
    {
        DPUWorkload wl{wl_conv};
        basicTest(wl, crt_model);
    }

    for (const auto& wl : wl_list) {
        basicTest(wl, crt_model, "All int8:");
    }
}
TEST_F(TestEnergyandPF_CostModel, BasicEnergy_FP16) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_FP) {
        basicTest(wl, crt_model, "All FP16:");
    }
}

TEST_F(TestEnergyandPF_CostModel, DPUInfoBasics) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All int8:");
    }
    for (const auto& wl : wl_list_FP) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All FP16:");
    }
}
TEST_F(TestEnergyandPF_CostModel, DPUInfoBasicsSparse) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_sparse) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All int8 sparse:");
    }
    for (const auto& wl : wl_list_FP_sparse) {
        basicDPUPackEquivalenceTest(wl, crt_model, "All FP16 sparse:");
    }
}

TEST_F(TestEnergyandPF_CostModel, SparseEnergy_INT8) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_sparse) {
        basicSparseTest(wl, crt_model, "All int8 sparse:");
    }
}
TEST_F(TestEnergyandPF_CostModel, SparseEnergy_FP16) {
    VPUCostModel crt_model{VPU_2_7_MODEL_PATH};

    for (const auto& wl : wl_list_FP_sparse) {
        basicSparseTest(wl, crt_model, "All FP16 sparse:");
    }
}

class TestVPUPowerFactorLUT : public ::testing::Test {
public:
protected:
    const DataType defaultTensorType{DataType::FLOAT16};
    const VPUDevice defaultDevice{VPUDevice::VPU_2_0};
    const float refPowerVirusFactor{VPUPowerFactorLUT().getFP_overI8_maxPower_ratio(VPUDevice::VPU_2_0) /* 0.87f*/};

    const std::array<VPUTensor, 1> outputs{VPUTensor(56, 56, 32, 1, defaultTensorType)};
    const std::array<unsigned int, 2> kernels{3, 3};                   ///< kernel sizes WH
    const std::array<unsigned int, 2> strides{1, 1};                   ///< kernel strides WH
    const std::array<unsigned int, 4> padding{1, 1, 1, 1};             ///< kernel padding  Top, Bottom, Left,  Right
    const ExecutionMode execution_order{ExecutionMode::CUBOID_16x16};  ///< execution mod

    // vpu_2_0_values{{Operation::CONVOLUTION,
    //                 {
    //                         {4, 0.87f},
    //                         {5, 0.92f},
    //                         {6, 1.0f},
    //                         {7, 0.95f},
    //                         {8, 0.86f},
    //                         {9, 0.87f},
    //                 }},

    void SetUp() override {
        //        TestCostModel::SetUp();
    }
    TestVPUPowerFactorLUT() {
    }

private:
};

TEST_F(TestVPUPowerFactorLUT, InsideMatchSamples) {
    const VPUPowerFactorLUT power_factor_lut;

    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 5), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.92F * refPowerVirusFactor, 0.005) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 7), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.95F * refPowerVirusFactor, 0.005) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 9), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.005) << wl;
    }

    // inside intemediary
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 7.5), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, ((0.95F + 0.86F) / 2) * refPowerVirusFactor, 0.001) << wl;
    }

    // inside intemediary
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 6.333), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, (1.0F + 0.333F * (0.95F - 1.0F)) * refPowerVirusFactor, 0.001) << wl;
    }
}
TEST_F(TestVPUPowerFactorLUT, BeforeFirstSample) {
    const VPUPowerFactorLUT power_factor_lut;

    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 0), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 1), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    // just before it
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, 15 /*(unsigned int)std::pow(2, 3.8F)*/, 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    // exactly at first
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 4), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
}
TEST_F(TestVPUPowerFactorLUT, AfterLastSample) {
    const VPUPowerFactorLUT power_factor_lut;

    {  // last one
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 9), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 9.9), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
    {
        DPUWorkload wl{defaultDevice,
                       Operation::CONVOLUTION,
                       {VPUTensor(56, 56, (unsigned int)std::pow(2, 11), 1, defaultTensorType)},
                       outputs,
                       kernels,
                       strides,
                       padding,
                       execution_order};

        float operation_pf{0.0f};
        ASSERT_NO_THROW(operation_pf = power_factor_lut.getOperationAndPowerVirusAdjustementFactor(wl)) << wl;
        EXPECT_NEAR(operation_pf, 0.87F * refPowerVirusFactor, 0.001) << wl;
    }
}
}  // namespace VPUNN_unit_tests