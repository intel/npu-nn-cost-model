// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "costmodel/cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

//@todo Add all LNL profiling results for JIra tickets to a fixture of UNit tests for LNL
//@todo Enhance DW-conv conversion factors to include the new ones from the LNL profiling results

/// Test batch values for devices
TEST_F(TestCostModel, BatchValues_DPU_Test) {
    auto mkWl = [](VPUDevice dev, unsigned int b, ExecutionMode exec = ExecutionMode::CUBOID_16x16) -> DPUWorkload {
        VPUNN::DPUWorkload wl{
                dev,
                Operation::CONVOLUTION,
                {VPUNN::VPUTensor(15, 50, 64, b, VPUNN::DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(15, 50, 64, b, VPUNN::DataType::UINT8)},  // output dimensions
                {1, 1},                                                     // kernels
                {1, 1},                                                     // strides
                {0, 0, 0, 0},                                               // padding
                exec                                                        // execution mode
        };

        return wl;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        std::string model_path;
    };

    struct TestExpectation {
        CyclesInterfaceType cyc_err;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string text = "";
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_cyc = [](const TestsVector& tests) {
        int i = 1;  // test case index
        for (const auto& t : tests) {
            std::cout << "Test case: " << i << "\n";
            VPUNN::VPUCostModel model{t.t_in.model_path};
            auto cyc = model.DPU(t.t_in.wl);

            if (t.t_exp.cyc_err == Cycles::NO_ERROR) {
                EXPECT_TRUE(!Cycles::isErrorCode(cyc));
            } else {
                EXPECT_EQ(cyc, t.t_exp.cyc_err) << "Actual: " << Cycles::toErrorText(cyc)
                                                << " Expected: " << Cycles::toErrorText(t.t_exp.cyc_err);
            }

            i++;
        }
    };

    /// for VPUDevice::VPU2_0, VPUDevice::VPU2_7, VPUDevice::VPU4_0 we accept any value for batch (ex: 1, 2, 3, ...)
    /// but for VPUDevice::NPU5_0 batch is restricted to 1
    const TestsVector tests = {{{mkWl(VPUDevice::VPU_2_0, 0U, ExecutionMode::VECTOR), VPU_2_0_MODEL_PATH},
                                {Cycles::ERROR_INVALID_INPUT_CONFIGURATION},
                                "VPU2_0, B=0"},
                               {{mkWl(VPUDevice::VPU_2_0, 1U, ExecutionMode::VECTOR), VPU_2_0_MODEL_PATH},
                                {Cycles::NO_ERROR},
                                "VPU2_0, B=1"},
                               {{mkWl(VPUDevice::VPU_2_0, 2U, ExecutionMode::VECTOR), VPU_2_0_MODEL_PATH},
                                {Cycles::NO_ERROR},
                                "VPU2_0, B=2"},

                               {{mkWl(VPUDevice::VPU_2_7, 0U), VPU_2_7_MODEL_PATH},
                                {Cycles::ERROR_INVALID_INPUT_CONFIGURATION},
                                "VPU2_7, B=0"},
                               {{mkWl(VPUDevice::VPU_2_7, 1U), VPU_2_7_MODEL_PATH}, {Cycles::NO_ERROR}, "VPU2_7, B=1"},
                               {{mkWl(VPUDevice::VPU_2_7, 2U), VPU_2_7_MODEL_PATH}, {Cycles::NO_ERROR}, "VPU2_7, B=2"},

                               {{mkWl(VPUDevice::VPU_4_0, 0U), VPU_4_0_MODEL_PATH},
                                {Cycles::ERROR_INVALID_INPUT_CONFIGURATION},
                                "VPU4_0, B=0"},
                               {{mkWl(VPUDevice::VPU_4_0, 1U), VPU_4_0_MODEL_PATH}, {Cycles::NO_ERROR}, "VPU4_0, B=1"},
                               {{mkWl(VPUDevice::VPU_4_0, 2U), VPU_4_0_MODEL_PATH}, {Cycles::NO_ERROR}, "VPU4_0, B=2"},
                               {{mkWl(VPUDevice::NPU_5_0, 0U), NPU_5_0_MODEL_PATH},
                                {Cycles::ERROR_INVALID_INPUT_CONFIGURATION},
                                "NPU5_0, B=0"},
                               {{mkWl(VPUDevice::NPU_5_0, 1U), NPU_5_0_MODEL_PATH}, {Cycles::NO_ERROR}, "NPU5_0, B=1"},
                               {{mkWl(VPUDevice::NPU_5_0, 2U), NPU_5_0_MODEL_PATH},
                                {Cycles::ERROR_INVALID_INPUT_CONFIGURATION},
                                "NPU5_0, B=2"},

    };

    verify_cyc(tests);
}

TEST_F(TestCostModel, Mock_40_vs_VPU27_DPU) {
    {  // 27 and 40
        VPUNN::VPUCostModel model_2_7{VPU_2_7_MODEL_PATH};
        EXPECT_TRUE(model_2_7.nn_initialized());
        VPUNN::VPUCostModel model_4_0{VPU_2_7_MODEL_PATH};  // no post processing mock
        EXPECT_TRUE(model_4_0.nn_initialized());

        auto cycles_27 = model_2_7.DPU(wl_glob_27);
        auto cycles_40 = model_4_0.DPU(wl_glob_40);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_27));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_40));

        // weights have different alignment, so we can't expect the same
        EXPECT_LE(cycles_27, cycles_40) << wl_glob_27 << wl_glob_40;
        // 9583 vs 9603
        EXPECT_GE(cycles_27 + 100, cycles_40) << wl_glob_27 << wl_glob_40;
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        VPUNN::VPUCostModel model_2_7{model_path};
        EXPECT_FALSE(model_2_7.nn_initialized());
        VPUNN::VPUCostModel model_4_0{model_path};
        EXPECT_FALSE(model_4_0.nn_initialized());

        auto cycles_27 = model_2_7.DPU(wl_glob_27);
        auto cycles_40 = model_4_0.DPU(wl_glob_40);

        EXPECT_EQ(cycles_27, 3445);  // theoretical, but at 1300MHz
        EXPECT_EQ(cycles_40, 3445);  // theoretical, but at 1700MHz
    }
}

TEST_F(TestCostModel, Mock_40_vs_VPU27_DMA) {
    const auto ratio_27per40 =
            (float)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::VPU_2_7) /
            (float)GlobalHarwdwareCharacteristics::get_dpu_fclk(VPUDevice::VPU_4_0);  //[cycles27/cycles40] is the unit
    {                                                                                 // DMA
        const std::string model_path = "NoFileHere.vpunn";
        VPUNN::VPUCostModel model_2_7{model_path};
        VPUNN::VPUCostModel model_4_0{model_path};

        auto cycles_27_DtoC = model_2_7.DMA(wl_glob_27.device, wl_glob_27.inputs[0], wl_glob_27.outputs[0],
                                            MemoryLocation::DRAM, MemoryLocation::CMX);
        auto cycles_40_DtoC = model_4_0.DMA(wl_glob_40.device, wl_glob_40.inputs[0], wl_glob_40.outputs[0],
                                            MemoryLocation::DRAM, MemoryLocation::CMX);

        EXPECT_GT(cycles_27_DtoC, (cycles_40_DtoC * ratio_27per40));                         // 2.7 is slower
        EXPECT_EQ(cycles_27_DtoC, 3658);                                                     // theoretical DMA
        EXPECT_EQ(cycles_40_DtoC, PerformanceMode::forceLegacy_G4 ? 4359 : 1883 + 87 + 59);  // theoretical DMA

        auto cycles_27_CtoD = model_2_7.DMA(wl_glob_27.device, wl_glob_27.inputs[0], wl_glob_27.outputs[0],
                                            MemoryLocation::CMX, MemoryLocation::DRAM);
        auto cycles_40_CtoD = model_4_0.DMA(wl_glob_40.device, wl_glob_40.inputs[0], wl_glob_40.outputs[0],
                                            MemoryLocation::CMX, MemoryLocation::DRAM);

        EXPECT_GT(cycles_27_CtoD, (cycles_40_CtoD * ratio_27per40));  // 2.7 is slower

        EXPECT_EQ(cycles_27_CtoD, 3658);                                                     // theoretical DMA
        EXPECT_EQ(cycles_40_CtoD, PerformanceMode::forceLegacy_G4 ? 4359 : 1883 + 87 + 59);  // theoretical DMA
    }
}

TEST_F(TestCostModel, Establish_unique_swizz_Test) {
    struct TestInput {
        Swizzling in0;
        Swizzling in1;
        Swizzling out0;
        Operation op;
    };

    struct TestExpectations {
        Swizzling in0;
        Swizzling in1;
        Swizzling out0;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectations t_exp;
    };
    using TestsVector = std::vector<TestCase>;

    auto lambda = [](TestsVector& tests) {
        int i = 1;
        for (auto& t : tests) {
            // std::cout << "\nTestCase: "<<i <<" Op:" << Operation_ToText.at(static_cast<int>(t.t_in.op));
            std::tuple<Swizzling, Swizzling, Swizzling> result_swizz =
                    NN40InputAdapter::establishUniqueSwizzling(t.t_in.in0, t.t_in.in1, t.t_in.out0, t.t_in.op);

            EXPECT_EQ(std::get<0>(result_swizz), t.t_exp.in0)
                    << "TestCase: " << i << " Op:" << Operation_ToText.at(static_cast<int>(t.t_in.op));
            EXPECT_EQ(std::get<1>(result_swizz), t.t_exp.in1)
                    << "TestCase: " << i << " Op:" << Operation_ToText.at(static_cast<int>(t.t_in.op));
            EXPECT_EQ(std::get<2>(result_swizz), t.t_exp.out0)
                    << "TestCase: " << i << " Op:" << Operation_ToText.at(static_cast<int>(t.t_in.op));

            i++;
        }
    };

    TestsVector tests = {
            // clang-format off
            {{Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0, Operation::ELTWISE}, {Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0}},
            {{Swizzling::KEY_1, Swizzling::KEY_0, Swizzling::KEY_5, Operation::ELTWISE}, {Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5}},
            {{Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_2, Operation::ELTWISE}, {Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_5}},// input!=output
            {{Swizzling::KEY_3, Swizzling::KEY_4, Swizzling::KEY_0, Operation::ELTWISE}, {Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_0}},// input!=output
            {{Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5, Operation::ELTWISE}, {Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5}},
            {{Swizzling::KEY_0, Swizzling::KEY_5, Swizzling::KEY_3, Operation::ELTWISE}, {Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5}},

            {{Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5, Operation::CONVOLUTION}, {Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5}},
            {{Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0, Operation::CONVOLUTION}, {Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5}},

            {{Swizzling::KEY_5, Swizzling::KEY_5, Swizzling::KEY_5, Operation::MAXPOOL}, {Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5}},
            {{Swizzling::KEY_0, Swizzling::KEY_0, Swizzling::KEY_0, Operation::MAXPOOL}, {Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5}},
            {{Swizzling::KEY_2, Swizzling::KEY_1, Swizzling::KEY_3, Operation::MAXPOOL}, {Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5}},
            {{Swizzling::KEY_5, Swizzling::KEY_4, Swizzling::KEY_0, Operation::MAXPOOL}, {Swizzling::KEY_5, Swizzling::KEY_0, Swizzling::KEY_5}},

            // clang-format on
    };

    lambda(tests);
}

TEST_F(TestCostModel, SmokeTests_DPUInfo) {
    {  // 20
        const DPUWorkload wl{wl_glob_20};
        const std::string modelFile{VPU_2_0_MODEL_PATH};

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());
        auto cycles_dpu = test_model.DPU(wl);
        auto cycles_Pack = test_model.DPUInfo(wl);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_dpu));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Pack.DPUCycles));
        EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
    }  // namespace VPUNN_unit_tests
    {  // 27
        const DPUWorkload wl{wl_glob_27};
        const std::string modelFile{VPU_2_7_MODEL_PATH};

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());
        auto cycles_dpu = test_model.DPU(wl);
        auto cycles_Pack = test_model.DPUInfo(wl);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_dpu));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Pack.DPUCycles));
        EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
    }
    {  // 40
        const DPUWorkload wl{wl_glob_40};
        const std::string modelFile{VPU_4_0_MODEL_PATH};

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());
        auto cycles_dpu = test_model.DPU(wl);
        auto cycles_Pack = test_model.DPUInfo(wl);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_dpu));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Pack.DPUCycles));
        EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
    }

    {  // 50
        const DPUWorkload wl{wl_glob_50};
        const std::string modelFile{NPU_5_0_MODEL_PATH};

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());
        auto cycles_dpu = test_model.DPU(wl);
        auto cycles_Pack = test_model.DPUInfo(wl);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_dpu));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Pack.DPUCycles));
        EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
    }

}
TEST_F(TestCostModel, SmokeTests_DPUInfo_stochastic) {
    {  // 20
        const DPUWorkload wl_device{wl_glob_20};
        const std::string modelFile{VPU_2_0_MODEL_PATH};
        constexpr unsigned int n_workloads = 100;

        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(wl_device.device));

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());

        for (const auto& wl : workloads) {
            auto cycles_dpu = test_model.DPU(wl);
            DPUInfoPack cycles_Pack;
            ASSERT_NO_THROW(cycles_Pack = test_model.DPUInfo(wl)) << wl;
            EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
        }
    }
    {  // 27
        const DPUWorkload wl_device{wl_glob_27};
        const std::string modelFile{VPU_2_7_MODEL_PATH};
        constexpr unsigned int n_workloads = 100;

        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(wl_device.device));

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());

        for (const auto& wl : workloads) {
            auto cycles_dpu = test_model.DPU(wl);
            DPUInfoPack cycles_Pack;
            ASSERT_NO_THROW(cycles_Pack = test_model.DPUInfo(wl)) << wl;
            EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
        }
    }
    {  // 40
        const DPUWorkload wl_device{wl_glob_40};
        const std::string modelFile{VPU_4_0_MODEL_PATH};
        constexpr unsigned int n_workloads = 100;

        auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
        std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(wl_device.device));

        VPUNN::VPUCostModel test_model{modelFile};
        EXPECT_TRUE(test_model.nn_initialized());

        for (const auto& wl : workloads) {
            auto cycles_dpu = test_model.DPU(wl);
            DPUInfoPack cycles_Pack;
            ASSERT_NO_THROW(cycles_Pack = test_model.DPUInfo(wl)) << wl;
            EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
        }
    }
}

/// test for verifying the influence of sparsities on the values of ideal cycles calculated based on MAC operations
TEST_F(TestCostModel, Ideal_cycles_based_on_MAC_operations_Test) {
    VPUNN::DPUWorkload wl_ref_2_0 = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(10, 10, 100, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(8, 8, 50, 1, VPUNN::DataType::UINT8)},     // output dimensions
            {3, 3},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    VPUNN::DPUWorkload wl_ref_2_7 = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(5, 5, 100, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(3, 3, 50, 1, VPUNN::DataType::UINT8)},   // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {0, 0, 0, 0},                                              // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
            VPUNN::ActivationFunction::NONE,                           // activation
            0.0F,                                                      // act_sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled
    };

    VPUNN::DPUWorkload wl_ref_4_0{
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {3, 3},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false                                                       // weight_sparsity_enabled
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
    };

    struct TestExpectation {
        unsigned long int power_ideal_cycles;       // DPU power ideal cycles
        unsigned long int efficiency_ideal_cycles;  // DPU efficiency ideal cycles
    };

    struct TestCase {
        TestInput t_in;
        std::string info = "";
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    /// @brief lambda function: set sparsities for an wl, they are given as parameters
    ///
    /// @param wl: The DPU Workload, we set its sparsities
    /// @param input_sparsity_enable: activate/deactivate input sparsity, it's value could be true (that means input
    /// sparsity is activate) or false (that means input sparsity is deactivate)
    /// @param act_sparsity: value for input sparsity; should be [0, 1]
    /// @param weight_sparsity_enable: activate/deactivate weight sparsity , it's value could be true (that means weight
    /// sparsity is activate) or false (that means weight sparsity is deactivate)
    /// @param weight_sparsity: value for weight sparsity; should be [0, 1]
    ///
    /// @return the wl with new sparsities
    auto wl_sparsity_initialization = [](const DPUWorkload& wl, bool input_sparsity_enable, float act_sparsity,
                                         bool weight_sparsity_enable, float weight_sparsity) -> DPUWorkload {
        DPUWorkload wl_ref{wl};
        // input sparsity
        wl_ref.inputs[0].set_sparsity(input_sparsity_enable);
        wl_ref.act_sparsity = act_sparsity;

        // weight sparsity
        wl_ref.weight_sparsity_enabled = weight_sparsity_enable;
        wl_ref.weight_sparsity = weight_sparsity;

        return wl_ref;
    };

    // this lambda function should verify if ideal cycles are computed correctly when sparsity influences the value
    // (power ideal cycles) and when not (efficiency ideal cycles)
    auto verify_ideal_cyc = [](const TestCase& t, VPUCostModel& test_model, const HWPerformanceModel& performance) {
        DPUInfoPack sparse_mac_op_info;

        ASSERT_NO_THROW(sparse_mac_op_info = test_model.DPUInfo(t.t_in.wl)) << t.t_in.wl;

        // direct method (calling functions) --> cycles
        EXPECT_EQ(performance.DPU_Power_IdealCycles(t.t_in.wl), t.t_exp.power_ideal_cycles);
        EXPECT_EQ(performance.DPU_Efficency_IdealCycles(t.t_in.wl), t.t_exp.efficiency_ideal_cycles);

        // DPU Info Pack -->cycles
        EXPECT_EQ(sparse_mac_op_info.power_ideal_cycles, t.t_exp.power_ideal_cycles);
        EXPECT_EQ(sparse_mac_op_info.efficiency_ideal_cycles, t.t_exp.efficiency_ideal_cycles);
    };

    /// this lambda function verify that power ideal cyc is smaller than efficiency ideal cyc when input or/and weight
    /// sparsity is/are active when both sparsity are inactive then power ideal cyc should be equal with efficiency
    /// ideal cyc
    auto verify_sparsity_influence = [](const TestCase& t, const VPUCostModel& /* test_model*/,
                                        const HWPerformanceModel& performance) {
        DPUInfoPack sparse_mac_op_info;

        unsigned long int power_cyc = performance.DPU_Power_IdealCycles(t.t_in.wl);
        unsigned long int efficiency_cyc = performance.DPU_Efficency_IdealCycles(t.t_in.wl);
        if (t.t_in.wl.inputs[0].get_sparsity() || t.t_in.wl.weight_sparsity_enabled)
            EXPECT_LT(power_cyc, efficiency_cyc);
        else
            EXPECT_EQ(power_cyc, efficiency_cyc);
    };

    ///@brief this lambda function executes a given lambda function as a parameter on each test case in a test vector
    ///@param tests a test vector
    ///@param testChecker is a lambda function
    auto run_Tests = [](const TestsVector& tests, VPUCostModel& test_model,
                        const HWPerformanceModel& performance, auto testCheck) {
        for (const auto& t : tests) {
            testCheck(t, test_model, performance);
        }
    };

    // 2_0
    {
        const TestsVector tests2_0 = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
        /*  ||                              workload                           ||         test info         || spars MAC op || cycles ||  */
            
            {{wl_sparsity_initialization(wl_ref_2_0,  false, 0.0F, false, 0.0F) }, "Device 2_0: No sparsity active, power cyc should be equal with efficiency cyc",  {11250, 11250}},
            {{wl_sparsity_initialization(wl_ref_2_0,  true, 0.7F, false, 0.0F)}, "Device 2_0: Input sparsity active, power cyc < efficiency cyc", {3376,  11250}},
            {{wl_sparsity_initialization(wl_ref_2_0,  false, 0.0F, true, 0.6F)}, "Device 2_0: Weight sparsity active, power cyc < efficiency cyc", {4500,  11250}},
            {{wl_sparsity_initialization(wl_ref_2_0, true, 0.2F, true, 0.4F)}, "Device 2_0: Input + Weight sparsity active, power cyc < efficiency cyc", {6751,  11250}},
            {{wl_sparsity_initialization(wl_ref_2_0,  true, 0.8F, true, 0.9F)}, "Device 2_0: Input + Weight sparsity active, power cyc < efficiency cyc", {1126,  11250}},

                // clang-format on
        };

        VPUCostModel test_model{VPU_2_0_MODEL_PATH};
        const HWPerformanceModel& performance{test_model.getPerformanceModel()};
        EXPECT_TRUE(test_model.nn_initialized());
        run_Tests(tests2_0, test_model, performance, verify_ideal_cyc);
        run_Tests(tests2_0, test_model, performance, verify_sparsity_influence);
    }

    // 2_7
    {
        const TestsVector tests2_7 = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
        /*  ||                              workload                           ||         test info         || spars MAC op || cycles ||   */

            {{wl_sparsity_initialization(wl_ref_2_7,  false, 0.0F, false, 0.0F) }, "Device 2_7: No sparsity active, power cyc should be equal with efficiency cyc", {198, 198}},
            {{wl_sparsity_initialization(wl_ref_2_7,  true, 0.7F, false, 0.0F)}, "Device 2_7: Input sparsity active, power cyc < efficiency cyc", {60, 198}},
            {{wl_sparsity_initialization(wl_ref_2_7,  false, 0.0F, true, 0.6F)}, "Device 2_7: Weight sparsity active, power cyc < efficiency cyc", {80, 198}},
            {{wl_sparsity_initialization(wl_ref_2_7, true, 0.2F, true, 0.4F)}, "Device 2_7: Input + Weight sparsity, power cyc < efficiency cyc", {119, 198}},
            {{wl_sparsity_initialization(wl_ref_2_7,  true, 0.8F, true, 0.9F)}, "Device 2_7: Input + Weight sparsity active, power cyc < efficiency cyc", {20, 198}},

                // clang-format on
        };

        VPUCostModel test_model{VPU_2_7_MODEL_PATH};
        const HWPerformanceModel& performance{test_model.getPerformanceModel()};
        EXPECT_TRUE(test_model.nn_initialized());
        run_Tests(tests2_7, test_model, performance, verify_ideal_cyc);
        run_Tests(tests2_7, test_model, performance, verify_sparsity_influence);
    }

    // 4_0
    {
        const TestsVector tests4_0 = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
        /*  ||                              workload                           ||         test info         || power cyc || efficiency cyc ||  */

            {{wl_sparsity_initialization(wl_ref_4_0,  false, 0.0F, false, 0.0F) }, "Device 4_0: No sparsity active, power cyc should be equal with efficiency cyc", {497, 497}},
            {{wl_sparsity_initialization(wl_ref_4_0,  true, 0.7F, false, 0.0F)}, "Device 4_0: Input sparsity active, power cyc < efficiency cyc", {149, 497}},
            {{wl_sparsity_initialization(wl_ref_4_0,  false, 0.0F, true, 0.6F)}, "Device 4_0: Weight sparsity active, power cyc < efficiency cyc", {199, 497}},
            {{wl_sparsity_initialization(wl_ref_4_0, true, 0.2F, true, 0.4F)}, "Device 4_0: Input + Weight sparsity active, power cyc < efficiency cyc", {298, 497}},
            {{wl_sparsity_initialization(wl_ref_4_0,  true, 0.8F, true, 0.9F)}, "Device 4_0: Input + Weight sparsity active, power cyc < efficiency cyc", {50, 497}},

                // clang-format on
        };
        VPUCostModel test_model{VPU_4_0_MODEL_PATH};
        const HWPerformanceModel& performance{test_model.getPerformanceModel()};
        EXPECT_TRUE(test_model.nn_initialized());
        run_Tests(tests4_0, test_model, performance, verify_ideal_cyc);
        run_Tests(tests4_0, test_model, performance, verify_sparsity_influence);
    }
}

TEST_F(TestCostModel, SEP_smoke_test) {
    Logger::activate2ndlog();
    {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        const DPUWorkload wl_ref_cnv = {
                VPUDevice::VPU_2_7,
                Operation::CONVOLUTION,
                {VPUTensor(7, 7, 512, 1, DataType::INT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(7, 7, 128, 1, DataType::INT8, Layout::ZXY)},  // output dimensions
                {3, 3},                                                  // kernels
                {1, 1},                                                  // strides
                {1, 1, 1, 1},                                            // padding
                ExecutionMode::CUBOID_4x16,                              // execution mode
                ActivationFunction::NONE,                                // activation
                0.0F,                                                    // act_sparsity
                0.0F,                                                    // weight_sparsity
                {swz_def, swz_def},                                      // input_swizzling
                {swz_def},                                               // output_swizzling
                1,                                                       // output_write_tiles
                {0, 0, 0, 0},                                            // offsets
                ISIStrategy::CLUSTERING,                                 // isi_strategy
                true,                                                    // weight_sparsity_enabled
                zeroHalo,                                                // halo
                sepInfo,                                                 // SEP
        };

        {
            VPUCostModel model_2_7{VPU_2_7_MODEL_PATH};
            // sparsity (input) is allowed only for CONV and ELEMENTwise, prohibited for rest
            // output sparsity should be enabled for all// not influencing  inferred runtime

            {  // conv sparse output
                DPUWorkload wl{std::move(wl_ref_cnv)};
                std::string info;
                auto cycles = model_2_7.DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
            }
        }
    }

    {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo107262{
                true,              // sep on
                {2050, 22, 1, 1},  // sep table,  4 bytes per element
                {512, 5, 64, 1},   // actual activator input,  same datatype as compute tensor input
        };
        const DPUWorkload wl_107262 = {
                VPUDevice::VPU_4_0,
                Operation::CONVOLUTION,
                {VPUTensor(2050, 22, 64, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
                {VPUTensor(1024, 10, 64, 1, DataType::FLOAT16, Layout::ZXY)},  // output dimensions
                {4, 4},                                                        // kernels
                {2, 2},                                                        // strides
                {0, 0, 0, 0},                                                  // padding
                ExecutionMode::CUBOID_16x16,                                   // execution mode
                ActivationFunction::NONE,                                      // activation
                0.0F,                                                          // act_sparsity
                0.984375F,                                                     // weight_sparsity
                {swz_def, swz_def},                                            // input_swizzling
                {swz_def},                                                     // output_swizzling
                1,                                                             // output_write_tiles
                {0, 0, 0, 0},                                                  // offsets
                ISIStrategy::CLUSTERING,                                       // isi_strategy
                true,                                                          // weight_sparsity_enabled
                zeroHalo,                                                      // halo
                sepInfo107262,                                                 // SEP
        };
        {
            VPUCostModel my_model{VPU_4_0_MODEL_PATH};
            // sparsity (input) is allowed only for CONV and ELEMENTwise, prohibited for rest
            // output sparsity should be enabled for all// not influencing  inferred runtime

            {  // conv sparse output
                DPUWorkload wl{std::move(wl_107262)};
                std::string info;
                auto cycles = my_model.DPU(wl, info);  // will change

                EXPECT_TRUE(Cycles::isErrorCode(cycles))  // memo is too big (with SEP) larger than 2MB vs 1.5MB limit
                                                          // see also: DPU_OperationValidator_Test.SEPMemorySize_Test
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << "\n info:" << info << std::endl
                        << "LOG:" << Logger::get2ndlog();

                EXPECT_TRUE((cycles != 0));

                EXPECT_EQ(cycles, V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG))
                        << "\n error is : " << VPUNN::Cycles::toErrorText(cycles) << "\n INFO: " << wl
                        << "\n info:" << info << std::endl
                        << "LOG:" << Logger::get2ndlog();

                // WL CMX MemorySize (aligned):
                //  total: 	2211840 ;
                //  input_0: 	884736 ;
                //  input_1: 	16384 ;
                //  output_0: 	1310720 ;
                //  inplace_output: 	false ;
                //  cmx overhead: 	98304 ;
                //  ignore_overhead: 	true ;
            }
        }
    }
}

}  // namespace VPUNN_unit_tests