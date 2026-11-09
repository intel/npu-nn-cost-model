// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "dpu_validator.h"
#include "logger/logger.h"
#include "vpu/cycles_interface_types.h"
#include "vpu_cost_model.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPU_OperationValidator_TestNPU6x : public DPU_OperationValidator_Test {};

TEST_F(DPU_OperationValidator_TestNPU6x, NPU60_presence_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::NPU_6_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                      // kernels
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

    // elementwise has in-place output, so no output contribution to total cmx size

    {  // no ISI strategy
        auto wl{std::move(wl_ref)};
        wl.device = VPUDevice::NPU_6_0;
        EXPECT_TRUE(dut.is_supported(wl.device)) << " device: \t" << (int)wl.device << " : "
                                                 << VPUDevice_ToText.at(static_cast<int>(wl.device)) << " ;\n";
    }
}

TEST_F(DPU_OperationValidator_TestNPU6x, WL_does_not_fit_in_CMX_InvestigationTest) {
    auto mk_wl = [](const unsigned int w_in, const unsigned int h_in, const unsigned int c_in, const unsigned int w_out,
                    const unsigned int h_out, const unsigned int c_out) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                Operation::CONVOLUTION,
                {VPUNN::VPUTensor(w_in, h_in, c_in, 1, DataType::UINT16)},      // input dimensions
                {VPUNN::VPUTensor(w_out, h_out, c_out, 1, DataType::FLOAT16)},  // output dimensions
                {3, 3},                                                         // kernels
                {1, 1},                                                         // strides
                {1, 0, 1, 1},                                                   // padding
                VPUNN::ExecutionMode::dCIM_32x128,                              // execution mode
                VPUNN::ActivationFunction::NONE,                                // activation
                0.0F,                                                           // act_sparsity
                0.0F,                                                           // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                           // input_swizzling
                {Swizzling::KEY_0},                                             // output_swizzling
                1,                                                              // output_write_tiles
                {0, 0, 0, 0},                                                   // offsets
                VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
                false,                                                          // weight_sparsity_enabled
                zeroHalo,                                                       // halo
                sepInfo,                                                        // sep
                DataType::UINT8                                                 // datatype for weights

        };
        wl.mpe_engine = MPEEngine::DCIM;

        return wl;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
    };

    struct TestCase {
        TestInput t_in;
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_wl = [](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            utest_log::debug() << "Test case "
                              << " " << i << "\n";
            VPUNN::VPUCostModel test_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

            unsigned int cycles = test_model.DPU(wl_ref);

            // we expect a valid cost
            EXPECT_FALSE(Cycles::isErrorCode(cycles))
                    << "Test case " << i << " produced error code " << cycles << " for wl: \n"
                    << wl_ref << "\n";
            EXPECT_GT(cycles, static_cast<CyclesInterfaceType>(1000)) << wl_ref;  // cost should not be negative or zero

            i++;
        }
    };

    {
        const TestsVector tests = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || cyc err expected ||    */
            {{mk_wl(160, 55, 48, 160, 54, 48)}},
            {{mk_wl(128, 3, 512, 128, 2, 256)}},
            {{mk_wl(256, 5, 512, 256, 4, 48)} },

                // clang-format on
        };

        verify_wl(tests);
    }
}

TEST_F(DPU_OperationValidator_TestNPU6x, Check_Memory_size_int32Bit_output_NPU60) {
    const VPUDevice device_req{VPUDevice::NPU_6_0};
    const DPUWorkload wl_{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 512, 1, DataType::UINT8)},                    // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::INT32)},                    // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };

    MemorySize memory;
    {
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_)) << wl_ << std::endl;
        // EXPECT_EQ(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(2097152, device_req)); /* =64*64*512*1*1(last 1 is the number of bytes for dtype */
        EXPECT_EQ(mem.input_1, align(2367488, device_req)); /* input1 shape: 1x1x4608x512 */
        EXPECT_EQ(mem.output_0, align(21 * 21 * 512 * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    {
        DPUWorkload wl_x{std::move(wl_)};
        wl_x.outputs[0].change_datatype_superficial(DataType::FLOAT32);
        EXPECT_EQ(wl_x.outputs[0].get_dtype(), DataType::FLOAT32);

        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_x)) << wl_x << std::endl;
        // EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(2097152, device_req)); /* =64*64*512*1*1(last 1 is the number of bytes for dtype */
        EXPECT_EQ(mem.input_1, align(2367488, device_req)); /* input1 shape: 1x1x4608x512 */
        EXPECT_EQ(mem.output_0, align(21 * 21 * 512 * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    const DPUWorkload wl_2{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 - 16, 1, DataType::INT32)},               // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };

    {
        DPUWorkload wl_x{std::move(wl_2)};
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_x)) << wl_x << std::endl;
        // EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(262144, device_req));        /* */
        EXPECT_EQ(mem.input_1, align(293632, device_req)) << mem; /* 1*1*(512-16)*576+((512-16)*16)(<-weight_table) */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512 - 16) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    const DPUWorkload wl_3{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::INT32)},                    // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT8,                                                 // input1 data type
    };

    {
        DPUWorkload wl_x{std::move(wl_3)};
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_x)) << wl_x << std::endl;
        // EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(262144, device_req));                   /* */
        EXPECT_EQ(mem.input_1, align(303104, device_req)) << mem;            /* 1*1*512*576+(512*16)(<-weight_table) */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    const DPUWorkload wl_4{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::INT32)},                    // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            ExecutionMode::CUBOID_16x16,                                     // execution mode
            ActivationFunction::NONE,                                        // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            DataType::UINT2,                                                 // input1 data type
    };

    {
        DPUWorkload wl_x{std::move(wl_4)};
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_x)) << wl_x << std::endl;
        // EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(262144, device_req));       /* */
        EXPECT_EQ(mem.input_1, align(81920, device_req)) << mem; /* (1*1*512*576) / 4 + (512*16)(<-weight_table) */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }
}

TEST_F(DPU_OperationValidator_TestNPU6x, UINT32_DataType_valid_for_NPU6_Output) {
    const DPUWorkload wl{
            VPUDevice::NPU_6_0,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},   // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::UINT32)}, // output dimensions
            {3, 3},                                         // kernels
            {3, 3},                                         // strides
            {0, 0, 0, 0},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                            // act_sparsity
            0.0F,                                            // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::CLUSTERING,                        // isi_strategy
            false,                                          // weight_sparsity_enabled
    };

    VPUCostModel model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};
    auto wl_copy{wl};
    const auto cycles = model.DPU(wl_copy);
    EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << "\n" << wl;
}

TEST_F(DPU_OperationValidator_TestNPU6x, Check_Memory_size_HALO_KH_output_NPU60) {
    const VPUDevice device_req{VPUDevice::NPU_6_0};
    const HaloWorkload halo_zero{
            {0, 0, 0, 0},  // input_0_halo
            {0, 0, 0, 0},  // output_0_halo
            {0, 0, 0, 0},  // output_0_halo_broadcast_cnt
            {0, 0, 0, 0}   // output_0_inbound_halo
    };

    const DPUWorkload wl_ref{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 256, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(64, 64, 128, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                        // kernels
            {1, 1},                                        // strides
            {1, 1, 1, 1},                                  // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::SPLIT_OVER_K,                     // isi_strategy
            false,                                         // weight_sparsity_enabled
            {
                    {0, 0, 0, 0, 0, 0},  // input_0_halo
                    {0, 0, 0, 0, 0, 0},  // output_0_halo
                    {0, 0, 0, 0, 0, 0},  // output_0_halo_broadcast_cnt
                    {0, 0, 0, 0, 0, 0}   // output_0_inbound_halo
            },                           // halo aspects

            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},  // SEP
            DataType::UINT8,                             // input1 data type
    };

    auto set_halo_wl = [](const DPUWorkload& wl, const HaloWorkload halo) {
        DPUWorkload r{wl};
        r.halo = halo;
        return r;
    };

    {
        MemorySize memoryToCompute;
        DPUWorkload wl_{set_halo_wl(wl_ref, halo_zero)};
        MemorySize& mem{memoryToCompute};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_)) << wl_ << std::endl;
        EXPECT_EQ(mem.input_0, align(1048576, device_req)) << mem;
        EXPECT_EQ(mem.output_0, align(524288, device_req)) << mem;
        EXPECT_EQ(mem.input_1, align(3 * 3 * 256 * 128 + 128 * 16, device_req)) << mem; /*  */
    }

    const HaloWorkload halo_forget_BTMhalf{
            // top half remains locally
            // TBLRFB
            {0, 0, 0, 0, 0, 0},   // input_0_halo TBLRFB
            {0, 32, 0, 0, 0, 0},  // output_0_halo  // bottom broadcast
            {0, 1, 0, 0, 0, 0},   // output_0_halo_broadcast_cnt
            {0, -32, 0, 0, 0, 0}  // output_0_inbound_halo // bottom forget
    };
    const HaloWorkload halo_forget_BTMhalf_addAllchannels{
            // top half remains locally
            // TBLRFB
            {0, 0, 0, 0, 0, 0},     // input_0_halo TBLRFB
            {0, 32, 0, 0, 0, 0},    // output_0_halo  // bottom broadcast
            {0, 1, 0, 0, 0, 0},     // output_0_halo_broadcast_cnt
            {0, -32, 0, 0, 0, 128}  // output_0_inbound_halo // bottom forget, broadcast in 128 ch
    };

    {
        MemorySize memoryToCompute;
        DPUWorkload wl_{set_halo_wl(wl_ref, halo_forget_BTMhalf)};
        MemorySize& mem{memoryToCompute};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_)) << wl_ << std::endl;
        // EXPECT_EQ(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(1048576, device_req)) << mem;
        EXPECT_EQ(mem.input_1, align(3 * 3 * 256 * 128 + 128 * 16, device_req)) << mem; /*  */

        EXPECT_EQ(mem.output_0, align(524288 / 2, device_req)) << mem;
    }
    {
        MemorySize memoryToCompute;
        DPUWorkload wl_{set_halo_wl(wl_ref, halo_forget_BTMhalf_addAllchannels)};
        MemorySize& mem{memoryToCompute};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_)) << wl_ << std::endl;
        // EXPECT_EQ(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(1048576, device_req)) << mem;
        EXPECT_EQ(mem.input_1, align(3 * 3 * 256 * 128 + 128 * 16, device_req)) << mem; /*  */

        EXPECT_EQ(mem.output_0, align(524288, device_req)) << mem;
    }
}

TEST_F(DPU_OperationValidator_TestNPU6x, Memory_Test_for_Reduce_Operations) {
    auto mk_wl = [](const Operation op, const DataType dtypeIn, const DataType dtypeOut) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                op,
                {VPUNN::VPUTensor(56, 56, 256, 1, dtypeIn)},  // input dimensions
                {VPUNN::VPUTensor(
                        56, 56, 1, 1,
                        dtypeOut)},  // output dimensions  -> reduce_* operations should have only 1 channel at output
                {1, 1},              // kernels
                {1, 1},              // strides
                {0, 0, 0, 0},        // padding
                VPUNN::ExecutionMode::CUBOID_16x16,    // execution mode
                VPUNN::ActivationFunction::NONE,       // activation
                0.0F,                                  // act_sparsity
                0.0F,                                  // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},  // input_swizzling
                {Swizzling::KEY_0},                    // output_swizzling
                1,                                     // output_write_tiles
                {0, 0, 0, 0},                          // offsets
                VPUNN::ISIStrategy::CLUSTERING,        // isi_strategy
                false,                                 // weight_sparsity_enabled
                zeroHalo,                              // halo
                sepInfo,                               // sep

        };

        return wl;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
    };

    struct TestExpectation {
        const long long input0_mem;
        const long long input1_mem;
        const long long output0_mem;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    const VPUDevice device_req{VPUDevice::NPU_6_0};
    using TestsVector = std::vector<TestCase>;

    auto verify_wl = [&](const TestsVector& tests) {
        for (const auto& t : tests) {
            MemorySize memoryToCompute;
            MemorySize& mem{memoryToCompute};

            EXPECT_NO_THROW(mem = dut.compute_wl_memory(t.t_in.wl)) << t.t_in.wl << std::endl;

            EXPECT_EQ(mem.input_0, align(t.t_exp.input0_mem, device_req)) << mem << std::endl << t.t_in.wl;
            EXPECT_EQ(mem.input_1, align(t.t_exp.input1_mem, device_req)) << mem << std::endl << t.t_in.wl;
            EXPECT_EQ(mem.output_0, align(t.t_exp.output0_mem, device_req)) << mem << std::endl << t.t_in.wl;
        }
    };

    {
        const TestsVector tests = {
                // clang-format off
            // !!! when dtype >=8 bits , we align innermost dimension to 16 elements when we compute out0 mem
            /************************************************ TABLE HEADER ********************************************************/
            /*  ||  workload operation   | wl in0/1 dtype |  wl out0 dtype       ||    in0 mem   | in1 mem  | out0 mem  ||    */
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT1, DataType::UINT1)},      {(56*56*256/8),  0,      (56*56*16) }},
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT2, DataType::UINT2)},      {(56*56*256/4),  0,      (56*56*16) }},
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT4, DataType::UINT4)},      {(56*56*256/2),  0,      (56*56*16) }},
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT8, DataType::UINT8)},      {(56*56*256),    0,      (56*56*16)  }},
            {{mk_wl(Operation::REDUCE_MS, DataType::FLOAT16, DataType::FLOAT16)},  {(56*56*256*2),  0,      (56*56*8*2) }},

            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT1, DataType::UINT1)},     {(56*56*256/8),  0,      (56*56*16) }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT2, DataType::UINT2)},     {(56*56*256/4),  0,      (56*56*16) }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT4, DataType::UINT4)},     {(56*56*256/2),  0,      (56*56*16) }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT8, DataType::UINT8)},     {(56*56*256),    0,      (56*56*16),  }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::FLOAT16, DataType::FLOAT16)}, {(56*56*256*2),  0,      (56*56*8*2) }},

                // clang-format on
        };

        verify_wl(tests);
    }
}

TEST_F(DPU_OperationValidator_TestNPU6x, DataTypes_Test_for_Reduce_Operations) {
    auto mk_wl = [](const Operation op, const DataType dtypeIn, const DataType dtypeOut,
                    const DataType dtypeWts) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                op,
                {VPUNN::VPUTensor(56, 56, 64, 1, dtypeIn)},  // input dimensions
                {VPUNN::VPUTensor(56, 56, 1, 1, dtypeOut)},  // output dimensions
                {1, 1},                                      // kernels
                {1, 1},                                      // strides
                {0, 0, 0, 0},                                // padding
                VPUNN::ExecutionMode::CUBOID_16x16,          // execution mode
                VPUNN::ActivationFunction::NONE,             // activation
                0.0F,                                        // act_sparsity
                0.0F,                                        // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},        // input_swizzling
                {Swizzling::KEY_0},                          // output_swizzling
                1,                                           // output_write_tiles
                {0, 0, 0, 0},                                // offsets
                VPUNN::ISIStrategy::CLUSTERING,              // isi_strategy
                false,                                       // weight_sparsity_enabled
                zeroHalo,                                    // halo
                sepInfo,                                     // sep
                dtypeWts                                     // weights data type

        };

        return wl;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
    };

    struct TestExpectation {
        const unsigned int error_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_wl = [](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            utest_log::debug() << "Test case "
                              << " " << i << "\n";
            VPUNN::VPUCostModel test_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

            unsigned int cycles = test_model.DPU(wl_ref);
            if (t.t_exp.error_expected == Cycles::NO_ERROR) {
                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "Test case " << i << " produced error code " << Cycles::toErrorText(cycles) << " for wl: \n"
                        << wl_ref << "\n";

            } else {
                EXPECT_EQ(Cycles::toCycleInterfaceType(cycles), t.t_exp.error_expected)
                        << "Test case " << i << " produced error code " << Cycles::toErrorText(cycles) << " for wl: \n"
                        << wl_ref << "\n";
            }

            i++;
        }
    };

    {
        const TestsVector tests = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  ||  workload operation   | wl in0/1 dtype |  wl out0 dtype                 ||                      cycles error                 ||    */
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT1, DataType::UINT1, DataType::UINT1)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION  }}, // unsupported dtype for input0/output0
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT2, DataType::UINT2, DataType::UINT2)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION  }}, // unsupported dtype for input0/output0
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT4, DataType::UINT4, DataType::UINT4)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION  }}, // unsupported dtype for input0/output0
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT8, DataType::UINT8, DataType::UINT8)},       {Cycles::NO_ERROR}},
            {{mk_wl(Operation::REDUCE_MS, DataType::FLOAT16, DataType::FLOAT16, DataType::FLOAT16)}, {Cycles::NO_ERROR}},

            {{mk_wl(Operation::REDUCE_MS, DataType::UINT8, DataType::UINT8, DataType::UINT1)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }}, // weights do not support INT1 as dtype
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT8, DataType::UINT8, DataType::UINT2)},       {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_MS, DataType::UINT8, DataType::UINT8, DataType::UINT4)},       {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_MS, DataType::FLOAT16, DataType::FLOAT16, DataType::UINT1)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }}, // weights do not support INT1 as dtype
            {{mk_wl(Operation::REDUCE_MS, DataType::FLOAT16, DataType::FLOAT16, DataType::UINT2)},       {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_MS, DataType::FLOAT16, DataType::FLOAT16, DataType::UINT4)},       {Cycles::NO_ERROR }},


            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT1, DataType::UINT1, DataType::UINT1)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION  }}, // unsupported dtype for input0/output0
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT2, DataType::UINT2, DataType::UINT2)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION  }}, // unsupported dtype for input0/output0
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT4, DataType::UINT4, DataType::UINT4)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION  }}, // unsupported dtype for input0/output0
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT8, DataType::UINT8, DataType::UINT8)},       {Cycles::NO_ERROR}},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::FLOAT16, DataType::FLOAT16, DataType::FLOAT16)}, {Cycles::NO_ERROR}},

             {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT8, DataType::UINT8, DataType::UINT1)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }}, // weights do not support INT1 as dtype
             {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT8, DataType::UINT8, DataType::UINT2)},       {Cycles::NO_ERROR }},
             {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::UINT8, DataType::UINT8, DataType::UINT4)},       {Cycles::NO_ERROR }},
             {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::FLOAT16, DataType::FLOAT16, DataType::UINT1)},       {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }}, // weights do not support INT1 as dtype
             {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::FLOAT16, DataType::FLOAT16, DataType::UINT2)},       {Cycles::NO_ERROR }},
             {{mk_wl(Operation::REDUCE_SUMSQUARES, DataType::FLOAT16, DataType::FLOAT16, DataType::UINT4)},       {Cycles::NO_ERROR }},

                // clang-format on
        };

        verify_wl(tests);
    }
}
TEST_F(DPU_OperationValidator_TestNPU6x, ExecModes_Test_for_Reduce_Operations) {
    auto mk_wl = [](const Operation op, const ExecutionMode execMode) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                op,
                {VPUNN::VPUTensor(56, 56, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(56, 56, 1, 1, DataType::UINT8)},   // output dimensions
                {1, 1},                                              // kernels
                {1, 1},                                              // strides
                {0, 0, 0, 0},                                        // padding
                execMode,                                            // execution mode
                VPUNN::ActivationFunction::NONE,                     // activation
                0.0F,                                                // act_sparsity
                0.0F,                                                // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                // input_swizzling
                {Swizzling::KEY_0},                                  // output_swizzling
                1,                                                   // output_write_tiles
                {0, 0, 0, 0},                                        // offsets
                VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
                false,                                               // weight_sparsity_enabled
                zeroHalo,                                            // halo
                sepInfo,                                             // sep

        };

        return wl;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
    };

    struct TestExpectation {
        const unsigned int error_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_wl = [](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            utest_log::debug() << "Test case "
                              << " " << i << "\n";
            VPUNN::VPUCostModel test_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

            unsigned int cycles = test_model.DPU(wl_ref);
            if (t.t_exp.error_expected == Cycles::NO_ERROR) {
                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "Test case " << i << " produced error code " << Cycles::toErrorText(cycles) << " for wl: \n"
                        << wl_ref << "\n";

            } else {
                EXPECT_EQ(Cycles::toCycleInterfaceType(cycles), t.t_exp.error_expected)
                        << "Test case " << i << " produced error code " << Cycles::toErrorText(cycles) << " for wl: \n"
                        << wl_ref << "\n";
            }

            i++;
        }
    };

    {
        const TestsVector tests = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  ||  workload operation   |        EXECUTION MODE         ||              cycles error           ||    */
            {{mk_wl(Operation::REDUCE_MS, ExecutionMode::CUBOID_16x16 )}, {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_MS, ExecutionMode::CUBOID_8x16 )},  {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_MS, ExecutionMode::CUBOID_4x16)},   {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_MS, ExecutionMode::MATRIX)},        {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_MS, ExecutionMode::VECTOR)},        {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},

            {{mk_wl(Operation::REDUCE_SUMSQUARES, ExecutionMode::CUBOID_16x16 )}, {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, ExecutionMode::CUBOID_8x16 )},  {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, ExecutionMode::CUBOID_4x16)},   {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, ExecutionMode::MATRIX)},        {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, ExecutionMode::VECTOR)},        {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},

                // clang-format on
        };

        verify_wl(tests);
    }
}

TEST_F(DPU_OperationValidator_TestNPU6x, OutChannels_Test_for_Reduce_Operations) {
    auto mk_wl = [](const Operation op, const unsigned int ch) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        DPUWorkload wl{
                VPUDevice::NPU_6_0,
                op,
                {VPUNN::VPUTensor(56, 56, 64, 1, DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(56, 56, ch, 1, DataType::UINT8)},  // output dimensions
                {1, 1},                                              // kernels
                {1, 1},                                              // strides
                {0, 0, 0, 0},                                        // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                  // execution mode
                VPUNN::ActivationFunction::NONE,                     // activation
                0.0F,                                                // act_sparsity
                0.0F,                                                // weight_sparsity
                {Swizzling::KEY_0, Swizzling::KEY_0},                // input_swizzling
                {Swizzling::KEY_0},                                  // output_swizzling
                1,                                                   // output_write_tiles
                {0, 0, 0, 0},                                        // offsets
                VPUNN::ISIStrategy::CLUSTERING,                      // isi_strategy
                false,                                               // weight_sparsity_enabled
                zeroHalo,                                            // halo
                sepInfo,                                             // sep

        };

        return wl;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
    };

    struct TestExpectation {
        const unsigned int error_expected;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_wl = [](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            utest_log::debug() << "Test case "
                              << " " << i << "\n";
            VPUNN::VPUCostModel test_model{VPUDevice::NPU_6_0, NPU_6_0_MODEL_PATH};

            unsigned int cycles = test_model.DPU(wl_ref);
            if (t.t_exp.error_expected == Cycles::NO_ERROR) {
                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "Test case " << i << " produced error code " << Cycles::toErrorText(cycles) << " for wl: \n"
                        << wl_ref << "\n";

            } else {
                EXPECT_EQ(Cycles::toCycleInterfaceType(cycles), t.t_exp.error_expected)
                        << "Test case " << i << " produced error code " << Cycles::toErrorText(cycles) << " for wl: \n"
                        << wl_ref << "\n";
            }

            i++;
        }
    };

    {
        const TestsVector tests = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload operation |   ch    ||              cycles error           ||    */
            // for reduce operations, output channels must be 1, this rule cand be finded in device_valid_valuesVPU6.h in function get_output_channels_restriction()
            {{mk_wl(Operation::REDUCE_MS, 1 )}, {Cycles::NO_ERROR }}, 
            {{mk_wl(Operation::REDUCE_MS, 2 )}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_MS, 5 )}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_MS, 11)}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_MS, 64)}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},

            {{mk_wl(Operation::REDUCE_SUMSQUARES, 1 )}, {Cycles::NO_ERROR }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, 2 )}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, 5 )}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, 11)}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},
            {{mk_wl(Operation::REDUCE_SUMSQUARES, 64)}, {Cycles::ERROR_INVALID_INPUT_CONFIGURATION }},

                // clang-format on
        };

        verify_wl(tests);
    }
}

}  // namespace VPUNN_unit_tests