// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "dpu_validator.h"
#include "vpu_cost_model.h"
namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPU_OperationValidator_TestNPU4x : public DPU_OperationValidator_Test {};

TEST_F(DPU_OperationValidator_TestNPU4x, SEPMemorySize_Test) {
    // const auto max_cmx{dut.get_config(VPUDevice::VPU_4_0).get_cmx_size(VPUDevice::VPU_4_0)};
    const HaloWorkload zeroHalo;
    const SEPModeInfo sepInfo{};

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
            sepInfo,                                                       // SEP
    };

    // elemntwise has in-place output, so no output contribution to total cmx size

    {  // no ISI strategy
        auto wl{wl_107262};
        EXPECT_TRUE(dut.is_supported(wl.device));
        const int in_compute_tensor{2050 * 22 * 64 * 2};
        const int al_in_ct{(int)align(in_compute_tensor, wl.device)};

        EXPECT_TRUE(isAligned(al_in_ct, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, al_in_ct) << mem << std::endl;
    }
    {  // no ISI strategy
        auto wl{std::move(wl_107262)};
        const SEPModeInfo sepInfo107262{
                true,              // sep on
                {2050, 22, 1, 1},  // sep table,  4 bytes per element
                {512, 5, 64, 1},   // actual activator input,  same datatype as compute tensor input
        };
        wl.sep_activators = sepInfo107262;

        EXPECT_TRUE(dut.is_supported(wl.device));

        const int actual_in{512 * 5 * 64 * 2};
        const int AST{(2050 * 22 * 64) / 8};  // to be aligned at 16?, already aligned 2050*11* 2*8
        const int SEPsize{2050 * 22 * 4};

        const int sep_input{(int)align(actual_in + AST + SEPsize, wl.device)};  // aligned to 16K

        const int wt{(int)std::ceil((4 * 4 * 64 * 64 * 2) * (1.0F - 0.984375F))};
        const int wt_aligned{(int)align(wt, wl.device)};

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, sep_input) << mem << wl << std::endl;
        EXPECT_EQ(mem.output_0, 1024 * 10 * 64 * 2) << mem << std::endl;
        EXPECT_GE(mem.input_1, wt) << mem << std::endl;
        EXPECT_EQ(mem.input_1, wt_aligned) << mem << std::endl;
        // EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1) << mem << std::endl;
        // EXPECT_LE(mem.cmx, max_cmx) << mem << std::endl;

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

TEST_F(DPU_OperationValidator_TestNPU4x, Output_tensor_memory_computation_test_for_different_innermost_dim_NPU4) {
    auto mk_wl = [](const unsigned int w, const unsigned int h, const unsigned int c, Layout layout) -> DPUWorkload {
        return VPUNN::DPUWorkload{
                VPUNN::VPUDevice::VPU_4_0,
                VPUNN::Operation::CONVOLUTION,
                {VPUNN::VPUTensor(32, 18, 64, 1, VPUNN::DataType::UINT8)},       // input dimensions
                {VPUNN::VPUTensor(w, h, c, 1, VPUNN::DataType::UINT8, layout)},  // output dimensions
                {3, 3},                                                          // kernels
                {1, 1},                                                          // strides
                {0, 0, 0, 0},                                                    // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
                VPUNN::ActivationFunction::NONE,                                 // activation
                0.0F,                                                            // act_sparsity
                0.0F,                                                            // weight_sparsity
                {swz_def, swz_def},                                              // input_swizzling
                {swz_def},                                                       // output_swizzling
                1,                                                               // output_write_tiles
                {0, 0, 0, 0},                                                    // offsets
                VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
                false                                                            // weight_sparsity_enabled

        };
        ;
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
    };

    struct TestExpectation {
        long long mem_size_exp;  // memory expected; it depends on test input
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto test_info = [](const DPUWorkload& wl) -> std::string {
        std::string info = "output tensor layout: " + Layout_ToText.at(static_cast<int>(wl.outputs[0].get_layout())) +
                           ", output channels: " + std::to_string(wl.outputs[0].z());
        return info;
    };

    // this lambda function verify if the output memory is computed correctly for different workloads
    auto verify_output0_memory = [this, &test_info](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            std::cout << "Test case "
                      << " " << i << ": " << test_info(wl_ref) << "\n";

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << "Test : " << i << " ," << wl << std::endl;
                EXPECT_EQ(mem.output_0, align(t.t_exp.mem_size_exp, wl.device))
                        << "Test : " << i << " ," << mem << std::endl
                        << t.t_exp.mem_size_exp << std::endl;
            }
            i++;
        }
    };

    /**************************************** HOW output0_memory is computed
    **********************************************
    !!! unaligned memory:
    First we align by 16 the innermost dimension of output tensor (W or H or C), then we compute memory based on these
    formulas:
    sparsity_map_bytes = (W*H*C) / 8 and aligned to 16
      if data type is FLOAT16 or BFLOAT16
           unaligned_output0_memory = W*H*C * 2 + sparsity_map_bytes
      else: unaligned_output0_memory = W*H*C + sparsity_map_bytes

      W, H, C are workload's dimensions

    */
    const TestsVector tests = {
            // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || output0 memory expected ||    */
            
            //innermost dim is C and is divisible by 16 (NEW:DEACTIVATED, align to 1)
            {{mk_wl(32, 18, 64, Layout::ZXY)},{36864}}, //32*18*64
            {{mk_wl(32, 18, 32, Layout::ZYX)},{18432}}, //32*18*32

            //innermost dim is C and is NOT divisible by 16
            {{mk_wl(32, 18, 63, Layout::ZXY)},{36288/*36864*/}}, //32*18*64 -> 63 aligned to 64
            {{mk_wl(32, 18, 67, Layout::ZYX)},{38592/*46080*/}}, //32*18*80 -> 67 aligned to 80

            //innermost dim is W and is divisible by 16
            {{mk_wl(32, 18, 65, Layout::XYZ)}, {37440 /*37440*/}}, //32*18*65
            {{mk_wl(128, 18, 65, Layout::XZY)},{149760/*149760*/}}, //128*18*65

            //innermost dim is W and is NOT divisible by 16
            {{mk_wl(33, 18, 65, Layout::XZY)}, {38610 /*56160*/}}, //48**18*65 -> 33 aligned to 48
            {{mk_wl(113, 18, 65, Layout::XYZ)},{132210/*149760*/}}, //128*18*65 -> 113 aligned to 128

            //innermost dim is H and is divisible by 16
            {{mk_wl(33, 32, 64, Layout::YXZ)},{67584/*67584*/}}, //33*32*64
            {{mk_wl(32, 64, 65, Layout::YZX)},{133120/*133120*/}}, //32*64*65

            //innermost dim is H and is NOT divisible by 16
            {{mk_wl(32, 18, 64, Layout::YXZ)},{36864/*65536*/}}, //32*32*64 -> 18 aligned to 32
            {{mk_wl(32, 37, 64, Layout::YZX)},{75776/*98304*/}}, //32*48*64 -> 37 aligned to 48

            // clang-format on
    };

    verify_output0_memory(tests);
}

TEST_F(DPU_OperationValidator_TestNPU4x, Input1_memory_computation_and_cycles_investigation_test_DW_CONV_old_NN) {
    VPUNN::DPUWorkload wl{
            VPUDevice::VPU_4_0,
            Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(4, 4, 1536, 1, DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(4, 4, 64, 1, DataType::FLOAT16)},    // output dimensions
            {1, 1},                                                // kernels
            {1, 1},                                                // strides
            {0, 0, 0, 0},                                          // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                    // execution mode
            VPUNN::ActivationFunction::NONE,                       // activation
            0.0F,                                                  // act_sparsity
            0.0F,                                                  // weight_sparsity
            {swz_def, swz_def},                                    // input_swizzling
            {swz_def},                                             // output_swizzling
            4,                                                     // output_write_tiles
            {0, 0, 0, 0},                                          // offsets
            VPUNN::ISIStrategy::SPLIT_OVER_K,                      // isi_strategy
            false,                                                 // weight_sparsity_enabled

    };

    auto test_info = [](const DPUWorkload& wl) -> std::string {
        std::string info =
                " device: \t" + VPUDevice_ToText.at(static_cast<int>(wl.device)) + " Operation: \t" +
                Operation_ToText.at(static_cast<int>(wl.op)) + " input1 dtype: \t" +
                (wl.weight_type.has_value() ? DataType_ToText.at(static_cast<int>(wl.weight_type.value())) : "Same") +
                " channels: \t" + std::to_string(wl.inputs[0].z()) + " weight_sparsity_enabled: \t" +
                (wl.weight_sparsity_enabled ? "true" : "false") + "\n";
        return info;
    };

    VPUNN::DPUWorkload wl_ref{wl};
    std::cout << "Test case " << test_info(wl_ref) << "\n";

    long long mem = 0;
    const auto& config = dut.get_config(wl.device);
    const DPUOperation op(wl, config);
    const IOperationDynamicConstraints& operation_behaviour{config.get_specific_behaviour(op.operation)};

    EXPECT_NO_THROW(mem = operation_behaviour.input_1_contiguous_size_bytes(config, op)) << wl << std::endl;
    EXPECT_EQ(mem, 3072) << test_info(wl_ref) << "\n";

    VPUCostModel model40_strict{VPU_4_0_159_STRICT_MODEL_PATH};
    auto cycles = model40_strict.DPU(std::move(wl));
    EXPECT_EQ(cycles, 326) << cycles;
}

TEST_F(DPU_OperationValidator_TestNPU4x, Input1_memory_computation_test) {
    auto mk_wl = [](const Operation op, const DataType dtype, const unsigned int c,
                    bool weights_sparsity = false) -> DPUWorkload {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        return VPUNN::DPUWorkload{
                VPUDevice::VPU_4_0,
                op,
                {VPUNN::VPUTensor(27, 18, c, 1, DataType::UINT8)},  // input dimensions
                {VPUNN::VPUTensor(27, 18, c, 1, DataType::UINT8)},  // output dimensions
                {3, 3},                                             // kernels
                {1, 1},                                             // strides
                {0, 0, 0, 0},                                       // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                 // execution mode
                VPUNN::ActivationFunction::NONE,                    // activation
                0.0F,                                               // act_sparsity
                0.0F,                                               // weight_sparsity
                {swz_def, swz_def},                                 // input_swizzling
                {swz_def},                                          // output_swizzling
                1,                                                  // output_write_tiles
                {0, 0, 0, 0},                                       // offsets
                VPUNN::ISIStrategy::CLUSTERING,                     // isi_strategy
                weights_sparsity,                                   // weight_sparsity_enabled
                zeroHalo,                                           // halo
                sepInfo,                                            // sep
                dtype                                               // datatype for weights

        };
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
    };

    struct TestExpectation {
        long long in1_mem_size_exp;  // memory expected; it depends on test input
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto test_info = [](const DPUWorkload& wl) -> std::string {
        std::string info =
                " device: \t" + VPUDevice_ToText.at(static_cast<int>(wl.device)) + " Operation: \t" +
                Operation_ToText.at(static_cast<int>(wl.op)) + " input1 dtype: \t" +
                (wl.weight_type.has_value() ? DataType_ToText.at(static_cast<int>(wl.weight_type.value())) : "Same") +
                " channels: \t" + std::to_string(wl.inputs[0].z()) + " weight_sparsity_enabled: \t" +
                (wl.weight_sparsity_enabled ? "true" : "false") + "\n";
        return info;
    };

    // this lambda function verify if the input1 memory is computed correctly for different workloads
    auto verify_input1_memory = [this, &test_info](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            VPUNN::DPUWorkload wl_ref{t.t_in.wl};
            std::cout << "Test case "
                      << " " << i << ": " << test_info(wl_ref) << "\n";

            {
                long long mem = 0;
                auto wl{wl_ref};
                const auto& config = dut.get_config(wl.device);
                const DPUOperation op(wl, config);
                const IOperationDynamicConstraints& operation_behaviour{config.get_specific_behaviour(op.operation)};

                EXPECT_NO_THROW(mem = operation_behaviour.input_1_contiguous_size_bytes(config, op))
                        << "Test : " << i << " ," << wl << std::endl;
                EXPECT_EQ(mem, t.t_exp.in1_mem_size_exp) << "Test case "
                                                         << " " << i << ": " << test_info(std::move(wl_ref)) << "\n";
            }
            i++;
        }
    };

    {
        const TestsVector tests_old_mechanism_active = {
                // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || input1 memory expected ||    */
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  64)},{5632}},  //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT1  => 72B for channels   */  + 64*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  64)},{10240}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT2  => 144B for channels  */  + 64*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  64)},{19456}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT4  => 288B for channels  */  + 64*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  64)},{37888}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT8  => 576B for channels  */  + 64*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 64)},{74752}}, //1*1*(64*3*3)*64*2 /* 576ch aligned at 16 samples -> 576ch and dtype UINT16 => 1152B for channels */  + 64*16 /*weights table*/

            //WEIGHT SPARSITY = TRUE
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  64, true)},{10752}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT1  => 72B for channels   */  + 64*16 /*weights table*/ + 80*64 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  64, true)},{15360}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT2  => 144B for channels  */  + 64*16 /*weights table*/ + 80*64 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  64, true)},{24576}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT4  => 288B for channels  */  + 64*16 /*weights table*/ + 80*64 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  64, true)},{43008}}, //1*1*(64*3*3)*64   /* 576ch aligned at 32 samples -> 576ch and dtype UINT8  => 576B for channels  */  + 64*16 /*weights table*/ + 80*64 /*sparsity map*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 64, true)},{79872}}, //1*1*(64*3*3)*64*2 /* 576ch aligned at 16 samples -> 576ch and dtype UINT16 => 1152B for channels */  + 64*16 /*weights table*/ + 80*64 /*sparsity map*/

            {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  21)},{840}},   //1*1*(21*3*3)*21   /* 189ch aligned at 32 samples -> 192ch and dtype UINT1  => 24B for channels  */  + 21*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  21)},{1344}},  //1*1*(21*3*3)*21   /* 189ch aligned at 32 samples -> 192ch and dtype UINT2  => 48B for channels  */  + 21*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  21)},{2352}},  //1*1*(21*3*3)*21   /* 189ch aligned at 32 samples -> 192ch and dtype UINT4  => 96B for channels  */  + 21*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  21)},{4368}},  //1*1*(21*3*3)*21   /* 189ch aligned at 32 samples -> 192ch and dtype UINT8  => 192B for channels */  + 21*16 /*weights table*/
            {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 21)},{8400}},  //1*1*(21*3*3)*21*2 /* 189ch aligned at 16 samples -> 192ch and dtype UINT16 => 384B for channels */  + 21*16 /*weights table*/

            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT1,  5)},{110}}, //1*1*(5*3*3)*5   /* 45ch aligned at 16 samples -> 48ch and dtype UINT1  => 6B  for channels */  + 16*5 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT2,  5)},{140}}, //1*1*(5*3*3)*5   /* 45ch aligned at 16 samples -> 48ch and dtype UINT2  => 12B for channels */  + 16*5 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT4,  5)},{200}}, //1*1*(5*3*3)*5   /* 45ch aligned at 16 samples -> 48ch and dtype UINT4  => 24B for channels */  + 16*5 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT8,  5)},{320}}, //1*1*(5*3*3)*5   /* 45ch aligned at 16 samples -> 48ch and dtype UINT8  => 48B for channels */  + 16*5 /*weights table*/
            {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT16, 5)},{560}}, //1*1*(5*3*3)*5*2 /* 45ch aligned at 8  samples -> 48ch and dtype UINT16 => 96B for channels */  + 16*5 /*weights table*/

            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  64)},{1280}}, //1*1*(3*3)*64   /* 9ch aligned at 32 samples -> 32ch and dtype UINT1  =>  4B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT2,  64)},{1536}}, //1*1*(3*3)*64   /* 9ch aligned at 32 samples -> 32ch and dtype UINT2  =>  8B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  64)},{2048}}, //1*1*(3*3)*64   /* 9ch aligned at 32 samples -> 32ch and dtype UINT4  => 16B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT8,  64)},{3072}}, //1*1*(3*3)*64   /* 9ch aligned at 32 samples -> 32ch and dtype UINT8  => 32B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
            {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT16, 64)},{3072}}, //1*1*(3*3)*64*2 /* 9ch aligned at 16 samples -> 16ch and dtype UINT16 => 32B for channels  should be aligned to 16B => 32B */  + 64*16 /*weights table*/

            {{mk_wl(Operation::ELTWISE, DataType::UINT1,  64)},{4608}}, //(4B)*18*64     -> same dim as input0
            {{mk_wl(Operation::ELTWISE, DataType::UINT2,  64)},{8064}}, //(7B)*18*64     -> same dim as input0
            {{mk_wl(Operation::ELTWISE, DataType::UINT4,  64)},{16128}}, //(14B)*18*64    -> same dim as input0
            {{mk_wl(Operation::ELTWISE, DataType::UINT8,  64)},{31104}}, //(27B)*18*64    -> same dim as input0   
            {{mk_wl(Operation::ELTWISE, DataType::UINT16, 64)},{62208}}, //(27*2B)*18*64) -> same dim as input0

            {{mk_wl(Operation::MAXPOOL, DataType::UINT1,  64)},{0}}, // maxpool operation doesn't have weights
            {{mk_wl(Operation::MAXPOOL, DataType::UINT2,  64)},{0}}, // maxpool operation doesn't have weights
            {{mk_wl(Operation::MAXPOOL, DataType::UINT4,  64)},{0}}, // maxpool operation doesn't have weights
            {{mk_wl(Operation::MAXPOOL, DataType::UINT8,  64)},{0}}, // maxpool operation doesn't have weights
            {{mk_wl(Operation::MAXPOOL, DataType::UINT16, 64)},{0}}, // maxpool operation doesn't have weights

                // clang-format on
        };

        verify_input1_memory(tests_old_mechanism_active);
    }

    //{
    //    const TestsVector tests_new_mechanism_active = {
    //            // clang-format off
    //        /************************************************ TABLE HEADER
    //        ********************************************************/
    //        /*  || workload || input1 memory expected ||    */
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  64)},{6144}},  //1*1*(64*3*3)*64   /* 576ch and dtype
    //        UINT1  => 72B for channels   should be aligned to 16B => 80B   */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  64)},{10240}}, //1*1*(64*3*3)*64   /* 576ch and dtype
    //        UINT2  => 144B for channels  should be aligned to 16B => 144B  */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  64)},{19456}}, //1*1*(64*3*3)*64   /* 576ch and dtype
    //        UINT4  => 288B for channels  should be aligned to 16B => 288B  */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  64)},{37888}}, //1*1*(64*3*3)*64   /* 576ch and dtype
    //        UINT8  => 576B for channels  should be aligned to 16B => 576B  */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 64)},{74752}}, //1*1*(64*3*3)*64*2 /* 576ch and dtype
    //        UINT16 => 1152B for channels should be aligned to 16B => 1152B */  + 64*16 /*weights table*/

    //        //WEIGHT SPARSITY = TRUE
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  64, true)},{11264}}, //1*1*(64*3*3)*64   /* 576ch and
    //        dtype UINT1  => 72B for channels   should be aligned to 16B => 80B   */  + 64*16 /*weights table*/ + 80*64
    //        /*sparsity map*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  64, true)},{15360}}, //1*1*(64*3*3)*64   /* 576ch and
    //        dtype UINT2  => 144B for channels  should be aligned to 16B => 144B  */  + 64*16 /*weights table*/ + 80*64
    //        /*sparsity map*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  64, true)},{24576}}, //1*1*(64*3*3)*64   /* 576ch and
    //        dtype UINT4  => 288B for channels  should be aligned to 16B => 288B  */  + 64*16 /*weights table*/ + 80*64
    //        /*sparsity map*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  64, true)},{43008}}, //1*1*(64*3*3)*64   /* 576ch and
    //        dtype UINT8  => 576B for channels  should be aligned to 16B => 576B  */  + 64*16 /*weights table*/ + 80*64
    //        /*sparsity map*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 64, true)},{79872}}, //1*1*(64*3*3)*64*2 /* 576ch and
    //        dtype UINT16 => 1152B for channels should be aligned to 16B => 1152B */  + 64*16 /*weights table*/ + 80*64
    //        /*sparsity map*/

    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT1,  21)},{1008}},  //1*1*(21*3*3)*21   /* 189ch and dtype
    //        UINT1  => 24B for channels  should be aligned to 16B => 32B  */  + 21*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT2,  21)},{1344}},  //1*1*(21*3*3)*21   /* 189ch and dtype
    //        UINT2  => 48B for channels  should be aligned to 16B => 48B  */  + 21*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT4,  21)},{2352}},  //1*1*(21*3*3)*21   /* 189ch and dtype
    //        UINT4  => 95B for channels  should be aligned to 16B => 96B  */  + 21*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT8,  21)},{4368}},  //1*1*(21*3*3)*21   /* 189ch and dtype
    //        UINT8  => 189B for channels should be aligned to 16B => 193B */  + 21*16 /*weights table*/
    //        {{mk_wl(Operation::CONVOLUTION, DataType::UINT16, 21)},{8400}},  //1*1*(21*3*3)*21*2 /* 189ch and dtype
    //        UINT16 => 378B for channels should be aligned to 16B => 384B */  + 21*16 /*weights table*/

    //        {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT1,  16)},{768}},  //1*1*(16*3*3)*16   /* 144ch and dtype
    //        UINT1  => 18B for channels  should be aligned to 16B => 32B  */  + 16*16 /*weights table*/
    //        {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT2,  16)},{1024}}, //1*1*(16*3*3)*16   /* 144ch and dtype
    //        UINT2  => 36B for channels  should be aligned to 16B => 48B  */  + 16*16 /*weights table*/
    //        {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT4,  16)},{1536}}, //1*1*(16*3*3)*16   /* 144ch and dtype
    //        UINT4  => 72B for channels  should be aligned to 16B => 80B  */  + 16*16 /*weights table*/
    //        {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT8,  16)},{2560}}, //1*1*(16*3*3)*16   /* 144ch and dtype
    //        UINT8  => 144B for channels should be aligned to 16B => 144B */  + 16*16 /*weights table*/
    //        {{mk_wl(Operation::CM_CONVOLUTION, DataType::UINT16, 16)},{4864}}, //1*1*(16*3*3)*16*2 /* 144ch and dtype
    //        UINT16 => 288B for channels should be aligned to 16B => 288B */  + 16*16 /*weights table*/

    //        {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT1,  64)},{2048}}, //1*1*(3*3)*64   /* 9ch and dtype UINT1
    //        => 2B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT2,  64)},{2048}}, //1*1*(3*3)*64   /* 9ch and dtype UINT2
    //        => 3B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT4,  64)},{2048}}, //1*1*(3*3)*64   /* 9ch and dtype UINT4
    //        => 5B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT8,  64)},{2048}}, //1*1*(3*3)*64   /* 9ch and dtype UINT8
    //        => 9B for channels  should be aligned to 16B => 16B */  + 64*16 /*weights table*/
    //        {{mk_wl(Operation::DW_CONVOLUTION, DataType::UINT16, 64)},{3072}}, //1*1*(3*3)*64*2 /* 9ch and dtype
    //        UINT16 => 18B for channels should be aligned to 16B => 32B */  + 64*16 /*weights table*/

    //        {{mk_wl(Operation::ELTWISE, DataType::UINT1,  64)},{4608}}, //(4B)*18*64     -> same dim as input0
    //        {{mk_wl(Operation::ELTWISE, DataType::UINT2,  64)},{8064}}, //(7B)*18*64     -> same dim as input0
    //        {{mk_wl(Operation::ELTWISE, DataType::UINT4,  64)},{16128}}, //(14B)*18*64    -> same dim as input0
    //        {{mk_wl(Operation::ELTWISE, DataType::UINT8,  64)},{31104}}, //(27B)*18*64    -> same dim as input0
    //        {{mk_wl(Operation::ELTWISE, DataType::UINT16, 64)},{62208}}, //(27*2B)*18*64) -> same dim as input0

    //        {{mk_wl(Operation::MAXPOOL, DataType::UINT1,  64)},{0}}, // maxpool operation doesn't have weights
    //        {{mk_wl(Operation::MAXPOOL, DataType::UINT2,  64)},{0}}, // maxpool operation doesn't have weights
    //        {{mk_wl(Operation::MAXPOOL, DataType::UINT4,  64)},{0}}, // maxpool operation doesn't have weights
    //        {{mk_wl(Operation::MAXPOOL, DataType::UINT8,  64)},{0}}, // maxpool operation doesn't have weights
    //        {{mk_wl(Operation::MAXPOOL, DataType::UINT16, 64)},{0}}, // maxpool operation doesn't have weights

    //            // clang-format on
    //    };

    //    verify_input1_memory(tests_new_mechanism_active);
    //}
}

TEST_F(DPU_OperationValidator_TestNPU4x, Elementwise_weightless_inplace_MemorySize_Test) {
    const DPUWorkload wl_{
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(180, 4, 640, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(180, 4, 640, 1, DataType::UINT8)},    // output dimensions
            {1, 1},                                          // kernels
            {1, 1},                                          // strides
            {0, 0, 0, 0},                                    // padding
            ExecutionMode::CUBOID_16x16,                     // execution mode
            ActivationFunction::NONE,                        // activation
            0.0F,                                            // act_sparsity
            0.0F,                                            // weight_sparsity
            {swz_def, swz_def},                              // input_swizzling
            {swz_def},                                       // output_swizzling
            1,                                               // output_write_tiles
            {0, 0, 0, 0},                                    // offsets
            ISIStrategy::CLUSTERING,                         // isi_strategy
            false,                                           // weight_sparsity_enabled
    };

    {  // weightless operation
        auto wl{wl_};
        wl.weightless_operation = true;
        wl.in_place_output_memory = false;

        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 2};   // float 16
        long long out_mem_bytes{180 * 4 * 640 * 1};  // int8

        EXPECT_FALSE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_FALSE(isAligned(out_mem_bytes, wl.device));
        auto alignedOutMemory{align(out_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedOutMemory, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.output_0, alignedOutMemory) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 0) << mem << std::endl;
        EXPECT_EQ(mem.inplace_output, false) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + mem.output_0) << mem << std::endl;
    }

    {  // inplace out
        auto wl{wl_};
        wl.weightless_operation = false;
        wl.in_place_output_memory = true;
        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 2};   // float 16
        long long out_mem_bytes{180 * 4 * 640 * 1};  // int8

        EXPECT_FALSE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_FALSE(isAligned(out_mem_bytes, wl.device));
        auto alignedOutMemory{align(out_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedOutMemory, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.output_0, alignedOutMemory) << mem << std::endl;
        EXPECT_EQ(mem.input_1, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.inplace_output, true) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + 0)
                << mem << std::endl;  // +0 because wl.in_place_output_memory = true;
    }

    {  // inplace out mem and weightless operation
        auto wl{std::move(wl_)};
        wl.weightless_operation = true;
        wl.in_place_output_memory = true;
        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 2};   // float 16
        long long out_mem_bytes{180 * 4 * 640 * 1};  // int8

        EXPECT_FALSE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_FALSE(isAligned(out_mem_bytes, wl.device));
        auto alignedOutMemory{align(out_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedOutMemory, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.output_0, alignedOutMemory) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 0) << mem << std::endl;
        EXPECT_EQ(mem.inplace_output, true) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + 0)
                << mem << std::endl;  // +0 because wl.in_place_output_memory = true;
    }
}

// test situations :
// - no in place memory due to layout OR data change
// - no weights input (detected by: layout + float to int change)
// test changed for out innermost dim alignment to 16 deactivation!
TEST_F(DPU_OperationValidator_TestNPU4x, elementwiseMemorySizeNoInout1ANdNoInplace_Test_NPU4) {
    const DPUWorkload wl_ref_full{
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(180, 4, 640, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(180, 4, 640, 1, DataType::UINT8, Layout::YZX)},    // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            ExecutionMode::CUBOID_16x16,                                  // execution mode
            ActivationFunction::NONE,                                     // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {swz_def, swz_def},                                           // input_swizzling
            {swz_def},                                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            ISIStrategy::CLUSTERING,                                      // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };
    const DPUWorkload wl_no_in_place_layout{
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(180, 4, 640, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(180, 4, 640, 1, DataType::UINT8, Layout::YZX)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            ExecutionMode::CUBOID_16x16,                                // execution mode
            ActivationFunction::NONE,                                   // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            ISIStrategy::CLUSTERING,                                    // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };
    const DPUWorkload wl_no_in_place_datasize{
            VPUDevice::VPU_4_0,
            Operation::ELTWISE,
            {VPUTensor(180, 4, 640, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(180, 4, 640, 1, DataType::UINT8, Layout::ZXY)},    // output dimensions
            {1, 1},                                                       // kernels
            {1, 1},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            ExecutionMode::CUBOID_16x16,                                  // execution mode
            ActivationFunction::NONE,                                     // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {swz_def, swz_def},                                           // input_swizzling
            {swz_def},                                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            ISIStrategy::CLUSTERING,                                      // isi_strategy
            false,                                                        // weight_sparsity_enabled
    };

    {  // no in place + no weights
        auto wl{std::move(wl_ref_full)};
        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 2};                                              // float 16
        long long out_mem_bytes{180 * 4 /*16 */ /*innermost dim is aligned to 16*/ * 640 * 1};  // int8

        EXPECT_FALSE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_FALSE(isAligned(out_mem_bytes, wl.device));
        auto alignedOutMemory{align(out_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedOutMemory, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.output_0, alignedOutMemory) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 0) << mem << std::endl;
        EXPECT_EQ(mem.inplace_output, false) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + mem.output_0) << mem << std::endl;
    }
    {  // no in place(due to layout) + no weights (due to layout change)
        auto wl{std::move(wl_no_in_place_layout)};
        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 1};                                             // int8
        long long out_mem_bytes{180 * 4 /*16*/ /*innermost dim is aligned to 16*/ * 640 * 1};  // int8

        EXPECT_FALSE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_FALSE(isAligned(out_mem_bytes, wl.device));
        auto alignedOutMemory{align(out_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedOutMemory, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.output_0, alignedOutMemory) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 0 /*alignedInMemory*/) << mem << std::endl;
        EXPECT_EQ(mem.inplace_output, false) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + mem.output_0) << mem << std::endl;
    }
    {  // no in place (due to datasize change) + no weights (layout does not change, but datasize changes)
        auto wl{std::move(wl_no_in_place_datasize)};
        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 2};   // FP16
        long long out_mem_bytes{180 * 4 * 640 * 1};  // int8

        EXPECT_FALSE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_FALSE(isAligned(out_mem_bytes, wl.device));
        auto alignedOutMemory{align(out_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedOutMemory, wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, alignedInMemory) << mem << std::endl;
        EXPECT_EQ(mem.output_0, alignedOutMemory) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 0 /*alignedInMemory*/) << mem << std::endl;
        EXPECT_EQ(mem.inplace_output, false) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + mem.output_0) << mem << std::endl;
    }
}

TEST_F(DPU_OperationValidator_TestNPU4x, Check_Memory_size_32Bit_output_NPU40) {
    const VPUDevice device_req{VPUDevice::VPU_4_0};
    const DPUWorkload wl_{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 512, 1, DataType::UINT8)},                    // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::FLOAT32)},                  // output dimensions
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
            {VPUTensor(21, 21, 512 - 16, 1, DataType::FLOAT32)},             // output dimensions
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
        EXPECT_EQ(mem.input_0, align(262144, device_req));                        /* */
        EXPECT_EQ(mem.input_1, align(294912, device_req)) << mem;                 /* aligned to 32 channels */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512 - 16) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    const DPUWorkload wl_3{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512, 1, DataType::FLOAT32)},                  // output dimensions
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
        EXPECT_EQ(mem.input_1, align(311296, device_req)) << mem;            /* aligned to 32 channels */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }
}

TEST_F(DPU_OperationValidator_TestNPU4x, Maxpool_HALOTest2) {
    const VPUDevice device_req{VPUDevice::VPU_4_0};
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 112, 64, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},       // output dimensions
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {1, 0, 1, 0},                                                    // padding TBLR
            VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload top_wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 28, 64, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(56, 14, 64, 1, VPUNN::DataType::UINT8)},       // output dimensions
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {1, 0, 1, 0},                                                    // padding TBLR
            VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload middle1_wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 31, 64, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(56, 15, 64, 1, VPUNN::DataType::UINT8)},       // output dimensions
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {0, 0, 1, 0},                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload middle2_wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 24, 64, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(56, 12, 64, 1, VPUNN::DataType::UINT8)},       // output dimensions
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {0, 0, 1, 0},                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload btm_wl_ref{
            device_req,  // VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(112, 29, 64, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(56, 15, 64, 1, VPUNN::DataType::UINT8)},       // output dimensions
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {0, 0, 1, 0},                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    // initial wl
    const auto input_0_raw{112 * 112 * 64};
    const auto output_0_raw{56 * 56 * 64};

    const auto input_0{align(input_0_raw, device_req)};
    const auto output_0{align(output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(input_0, device_req));
    EXPECT_TRUE(isAligned(output_0, device_req));

    // top wl
    const auto top_input_0_raw{112 * 28 * 64};
    const auto top_output_0_raw{56 * 14 * 64};

    const auto top_input_0{align(top_input_0_raw, device_req)};
    const auto top_output_0{align(top_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(top_input_0, device_req));
    EXPECT_TRUE(isAligned(top_output_0, device_req));

    // middle1 wl
    const auto mid1_input_0_raw{112 * 31 * 64};
    const auto mid1_output_0_raw{56 * 15 * 64};

    const auto mid1_input_0{align(mid1_input_0_raw, device_req)};
    const auto mid1_output_0{align(mid1_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(mid1_input_0, device_req));
    EXPECT_TRUE(isAligned(mid1_output_0, device_req));

    // middle2 wl
    const auto mid2_input_0_raw{112 * 24 * 64};
    const auto mid2_output_0_raw{56 * 12 * 64};

    const auto mid2_input_0{align(mid2_input_0_raw, device_req)};
    const auto mid2_output_0{align(mid2_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(mid2_input_0, device_req));
    EXPECT_TRUE(isAligned(mid2_output_0, device_req));

    // bottom wl
    const auto btm_input_0_raw{112 * 29 * 64};
    const auto btm_output_0_raw{56 * 15 * 64};

    const auto btm_input_0{align(btm_input_0_raw, device_req)};
    const auto btm_output_0{align(btm_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(btm_input_0, device_req));
    EXPECT_TRUE(isAligned(btm_output_0, device_req));

    ASSERT_TRUE(dut.is_supported(wl_ref.device));
    ASSERT_TRUE(dut.is_supported(top_wl_ref.device));
    ASSERT_TRUE(dut.is_supported(middle1_wl_ref.device));
    ASSERT_TRUE(dut.is_supported(middle2_wl_ref.device));
    ASSERT_TRUE(dut.is_supported(btm_wl_ref.device));

    {
        MemorySize mem_clean;
        {
            MemorySize& mem{mem_clean};
            auto wl{std::move(wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
            EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        }

        MemorySize top_mem_halo;
        {
            MemorySize& mem{top_mem_halo};
            auto wl{std::move(top_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, top_input_0) << mem << std::endl;
            EXPECT_EQ(mem.output_0, top_output_0) << mem << std::endl;
        }

        EXPECT_GT(mem_clean.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_GT(mem_clean.output_0, top_mem_halo.output_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << std::endl;

        MemorySize mid1_mem_halo;
        {
            MemorySize& mem{mid1_mem_halo};
            auto wl{std::move(middle1_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_GT(mem.input_0, mid1_input_0 - (112 * 64)) << mem << std::endl;
            EXPECT_EQ(mem.output_0, mid1_output_0) << mem << std::endl;
        }

        EXPECT_GT(mid1_mem_halo.input_0, top_mem_halo.input_0) << mid1_mem_halo << top_mem_halo << std::endl;
        EXPECT_EQ(mid1_mem_halo.output_0, top_mem_halo.output_0) << mid1_mem_halo << top_mem_halo << std::endl;
        EXPECT_EQ(mid1_mem_halo.input_1, top_mem_halo.input_1) << mid1_mem_halo << top_mem_halo << std::endl;

        MemorySize mid2_mem_halo;
        {
            MemorySize& mem{mid2_mem_halo};
            auto wl{std::move(middle2_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_LT(mem.input_0, mid2_input_0 - (2 * 112 * 64)) << mem << std::endl;
            EXPECT_EQ(mem.output_0, mid2_output_0) << mem << std::endl;
        }

        EXPECT_LT(mid2_mem_halo.input_0, mid1_mem_halo.input_0) << mid2_mem_halo << mid1_mem_halo << std::endl;
        EXPECT_LT(mid2_mem_halo.output_0, mid1_mem_halo.output_0) << mid2_mem_halo << mid1_mem_halo << std::endl;
        EXPECT_EQ(mid2_mem_halo.input_1, mid1_mem_halo.input_1) << mid2_mem_halo << mid1_mem_halo << std::endl;

        MemorySize btm_mem_halo;
        {
            MemorySize& mem{btm_mem_halo};
            auto wl{std::move(btm_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_LT(mem.input_0, btm_input_0 - (2 * 112 * 64)) << mem << std::endl;
            EXPECT_EQ(mem.output_0, btm_output_0) << mem << std::endl;
        }

        EXPECT_GT(btm_mem_halo.input_0, mid2_mem_halo.input_0)
                << mem_clean << top_mem_halo << mid2_mem_halo << btm_mem_halo << std::endl;
        EXPECT_GT(btm_mem_halo.output_0, mid2_mem_halo.output_0)
                << mem_clean << top_mem_halo << mid2_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.input_1, mid2_mem_halo.input_1)
                << mem_clean << top_mem_halo << mid2_mem_halo << btm_mem_halo << std::endl;
    }
}

TEST_F(DPU_OperationValidator_TestNPU4x, VPU40_presence_Test) {
    const VPUDevice device_req{VPUDevice::VPU_4_0};
    const VPUNN::DPUWorkload wl_ref{
            device_req,
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

    // elemntwise has in-place output, so no output contribution to total cmx size

    {  // no ISI strategy
        auto wl{std::move(wl_ref)};
        EXPECT_TRUE(dut.is_supported(wl.device));
    }
}

}  // namespace VPUNN_unit_tests