// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/dpu_operations_validator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPU_OperationValidator_Test : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    VPUNN::DPU_OperationValidator dut;  // no overhead by default

    const int cmx_overhead{0 /*80 * 1024 + 16 * 1024*/};  // cmx_memory_aligned_overhead
    // const int alignment{16384};                           // alignement_size_bytes

    const std::map<VPUDevice, int> alignement_data{
            {VPUDevice::VPU_2_0, 16 * 1024},    //
            {VPUDevice::VPU_2_1, 16 * 1024},    //
            {VPUDevice::VPU_2_7, 16 * 1024},    //
            {VPUDevice::VPU_4_0, 16 * 1024},    //
    };

    int get_alignment(const VPUDevice device) const {
        auto it = alignement_data.find(device);
        if (it != alignement_data.end()) {
            return it->second;
        }
        return 0;  // no alignment
    }

    bool isAligned(long long mem_size, int alignment) const {
        return ((mem_size % alignment) != 0) ? false : true;
    }
    bool isAligned(long long mem_size, const VPUDevice device) const {
        return ((mem_size % get_alignment(device)) != 0) ? false : true;
    }

    long long int align(long long mem_size, int alignment) const {
        const auto rem = mem_size % alignment;
        return (rem == 0) ? mem_size : mem_size + (alignment - rem);
    }
    long long int align(long long mem_size, const VPUDevice device) const {
        const auto rem = mem_size % get_alignment(device);
        return (rem == 0) ? mem_size : mem_size + (get_alignment(device) - rem);
    }

    DPU_OperationValidator_Test() {
    }

private:
};

TEST_F(DPU_OperationValidator_Test, elementwiseMemorySize_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
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
        auto wl{wl_ref};
        EXPECT_TRUE(dut.is_supported(wl.device));

        EXPECT_TRUE(isAligned(56 * 56 * 256, VPUDevice::VPU_2_7));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.output_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1) << mem << std::endl;
    }

    {  // SOH-> activators are halved in contribution (NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.output_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 / 1 + mem.input_1) << mem << std::endl;
    }
    {  // SOK-> weights are  is halved in contribution(NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.output_0, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.input_1, 56 * 56 * 256) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 / 1) << mem << std::endl;
    }
}

TEST_F(DPU_OperationValidator_Test, SEPMemorySize_Test) {
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

TEST_F(DPU_OperationValidator_Test, InputSparsity_and_SEP_memory_test) {
    const HaloWorkload zeroHalo;
    const SEPModeInfo sepInfo{};
    const VPUNN::DPUWorkload wl_ref_18x18x64{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(18, 18, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 18, 48, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                         // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled
            zeroHalo,                                                   // halo aspects
            sepInfo                                                     // SEP
    };

    const VPUNN::DPUWorkload wl_ref_64x64x512{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {3, 3},                                                      // strides
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
            zeroHalo,                                                    // halo aspects
            sepInfo                                                      // SEP
    };

    const VPUNN::DPUWorkload wl_ref_64x128x512{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::UINT8)},   // output dimensions
            {3, 3},                                                       // kernels
            {3, 3},                                                       // strides
            {0, 0, 0, 0},                                                 // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
            VPUNN::ActivationFunction::NONE,                              // activation
            0.0F,                                                         // act_sparsity
            0.0F,                                                         // weight_sparsity
            {swz_def, swz_def},                                           // input_swizzling
            {swz_def},                                                    // output_swizzling
            1,                                                            // output_write_tiles
            {0, 0, 0, 0},                                                 // offsets
            VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
            false,                                                        // weight_sparsity_enabled
            zeroHalo,                                                     // halo aspects
            sepInfo                                                       // SEP
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;  // the wl for which we compute memory
        // these fields below modify the wl
        SEPModeInfo sepInfo;  // SEP: sep_activators, storeage_elements_pointers, actual_activators_input, no_sparse_map
        bool input_sparsity_enable;  // activate/deactivate input sparsity
        float input_sparsity;        // value for input sparsity
    };

    struct TestExpectation {
        long long mem_size_exp;  // memory expected; it depends on test input
    };

    struct TestCase {
        TestInput t_in;
        std::string info = "";
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_input0_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << "Test case:"
                      << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref{t.t_in.wl};

            // SEP
            wl_ref.sep_activators = t.t_in.sepInfo;

            // input sparsity
            wl_ref.inputs[0].set_sparsity(t.t_in.input_sparsity_enable);
            wl_ref.act_sparsity = t.t_in.input_sparsity;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.input_0, t.t_exp.mem_size_exp) << mem << std::endl << t.t_exp.mem_size_exp << std::endl;
            }
            i++;
        }
    };

    /*
    a test case in TestsVector tests is comprised of Test input and Test expectation

    **************************************************************************************************************
     1.Test input consist of a wl, SEP and input sparsity
      - SEP:
               - sep_activators --> activators using Storage elements table with pointers, it's value could be true or
               false

               - storage_elements_pointers --> SEP pointer table, 32 bits pointers assumed, it's dimensions should be
               {W, H, 1, 1} W and H are input tensors's width and height dimensions

               - actual_activators_input --> actual tensor shape for activators, it's dimensions should be {a, b, C, 1}
               C is the input tensor's channels number, a and b are 2 random values, but axb should not exceed 2050x22,
               that is the maximum number of pointers

               - no_sparse_map --> it's value could be true or false, if true the sparse map is ignored/non existent

      - sparsity: after SEP there are information about input sparsity
             Input sparsity
               - true/false for enable
               - value --> float, should be in the range (0, 1)
    *******************************************************************************************************************

     2. Test expectations consist of memory value expected, computed based on wl dimensions (WHCB) and SEP :
       data_memory_samples dpu.input_0.datatype + sparsity_map_bytes + storage_elements_table_samples * pointer_size

          - data_memory_samples:
              **if sep_activators=true: is obtained by multiplying the dimensions of actual_activators_input together
              **if sep_activators=false: W*H*C*B  W,H,C,B are input tensor dimensions !!! halo influence
          !!! it's value could be *2 if tensor's data type is FLOAT16 or BFLOAT16 in both cases

          - sparsity_map_bytes can or cannot be missing; it is present if input sparsity is true or/and no_sparse_map is
          false; If input sparsity is false and no_sparse_map is true we ignore sparsity_map_bytes

          - storeage_elements_table_samples is obtained by multiplying the dimensions of storage_elements_pointers
          together
    ********************************************************************************************************************

    */

    const TestsVector tests = {
            // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
        /*  ||   workload   || sep_act |     SEP    |  activators  | no_SM ||act_sp + val||      test info     || input_0 aligned memory exp||    */

            //case when the dimensions of wl are small, so the memory value will be small
            {{wl_ref_18x18x64, {true, {18, 18, 1, 1}, {12, 10, 64, 1}, false}, true, 0.3F}, "SEP decrease memory", {16384}},
            {{wl_ref_18x18x64, {true, {18, 18, 1, 1}, {512, 6, 70, 1}, false}, true, 0.7F}, "SEP increase memory", {229376}},
            {{wl_ref_18x18x64, {true, {18, 18, 1, 1}, {1024, 12, 64, 1}, true}, false, 0.0F}, "SEP increase memory, without input sparsity", {802816}},
            {{wl_ref_18x18x64, {true, {18, 18, 1, 1}, {2000, 19, 190, 1}, false}, true, 0.2F}, "SEP increase memory", {7225344}},
            {{wl_ref_18x18x64, {false, {18, 18, 1, 1}, {67, 20, 190, 1}, false}, true, 0.8F}, "memory without SEP, but with input sparsity", {32768}},
            {{wl_ref_18x18x64, {false, {18, 18, 1, 1}, {1989, 13, 190, 1}, true}, false, 0.0F}, "memory without SEP and input sparsity", {32768}},

            //case when the dimensions of wl are neither too large nor too small
            {{wl_ref_64x64x512, {true, {64, 64, 1, 1}, {20, 11, 512, 1}, false}, false, 0.0F}, "SEP decrease memory", {393216}}, 
            {{wl_ref_64x64x512, {true, {64, 64, 1, 1}, {2000, 19, 600, 1}, true}, false, 0.0F}, "SEP increase memory, without input sparsity", {22822912}},
            {{wl_ref_64x64x512, {true, {64, 64, 1, 1}, {335, 20, 512, 1}, false}, true, 0.72F}, "SEP increase memory", {3719168}},
            {{wl_ref_64x64x512, {true, {64, 64, 1, 1}, {1234, 9, 2000, 1}, false}, true, 0.34F}, "SEP increase memory", {22495232}},
            {{wl_ref_64x64x512, {false, {64, 64, 1, 1}, {340, 15, 2000, 1}, false}, false, 0.0F}, "memory without SEP and input sparsity", {2097152}},
            {{wl_ref_64x64x512, {false, {64, 64, 1, 1}, {1999, 9, 2000, 1}, true}, true, 0.86F}, "memory without SEP, but with input sparsity", {2359296}}, 

            //case when the dimensions of wl are large, so the memory value will be large
            {{wl_ref_64x128x512, {true, {64, 128, 1, 1}, {17, 19, 512, 1}, true}, false, 0.0F}, "SEP decrease memory, without input sparsity", {212992}},
            {{wl_ref_64x128x512, {true, {64, 128, 1, 1}, {534, 7, 522, 1}, false}, true, 0.55F}, "SEP increase memory", {2523136}},
            {{wl_ref_64x128x512, {true, {64, 128, 1, 1}, {1000, 9, 512, 1}, false}, true, 0.47F}, "SEP increase memory", {5177344}},
            {{wl_ref_64x128x512, {true, {64, 128, 1, 1}, {2000, 19, 1500, 1}, true}, true, 0.39F}, "SEP increase memory", {57573376}},
            {{wl_ref_64x128x512, {false, {64, 128, 1, 1}, {678, 10, 1500, 1}, false}, false, 0.0F}, "memory without SEP and input sparsity", {4194304}},
            {{wl_ref_64x128x512, {false, {64, 128, 1, 1}, {2010, 21, 1500, 1}, true}, true, 0.91F}, "memory without SEP, but with input sparsity", {4718592}},
            // clang-format on
    };

    verify_input0_memory(tests);
}

TEST_F(DPU_OperationValidator_Test, OutputSparsity_and_SEP_memory_test) {
    const VPUNN::DPUWorkload wl_ref_18x18x64{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(18, 18, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 18, 48, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 0, 0},                                               // padding
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

    const VPUNN::DPUWorkload wl_ref_64x64x512{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {3, 3},                                                      // strides
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
            false                                                        // weight_sparsity_enabled

    };

    const VPUNN::DPUWorkload wl_ref_64x128x512{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {3, 3},                                                         // kernels
            {3, 3},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                             // execution mode
            VPUNN::ActivationFunction::NONE,                                // activation
            0.0F,                                                           // act_sparsity
            0.0F,                                                           // weight_sparsity
            {swz_def, swz_def},                                             // input_swizzling
            {swz_def},                                                      // output_swizzling
            1,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false                                                           // weight_sparsity_enabled

    };

    struct TestInput {
        VPUNN::DPUWorkload wl;        // the wl for which we compute memory
        bool output_sparsity_enable;  // activate/deactivate output sparsity; !!!affect memory
    };

    struct TestExpectation {
        long long mem_size_exp;  // memory expected; it depends on test input
    };

    struct TestCase {
        TestInput t_in;
        std::string info = "";
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    auto verify_output0_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << "Test case:"
                      << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref{t.t_in.wl};

            // output sparsity
            wl_ref.outputs[0].set_sparsity(t.t_in.output_sparsity_enable);

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.output_0, t.t_exp.mem_size_exp) << mem << std::endl << t.t_exp.mem_size_exp << std::endl;
            }
            i++;
        }
    };

    /*
    a test case in TestsVector tests is comprised of Test input and Test expectation

    **************************************************************************************************************
     1.Test input consist of a wl and output sparsity
      - sparsity: after SEP there are information about output sparsity
             Output sparsity
               - true/false for enable
    *******************************************************************************************************************

     2. Test expectations consist of memory value expected, computed based on wl dimensions (WHCB) and sparsity map
    given by output sparsity : data_memory_samples + sparsity_map_bytes

          - data_memory_samples:
              W*H*C*B --> W,H,C,B are output tensor dimensions !!! halo influence
          !!! it's value could be *2 if tensor's data type is FLOAT16 or BFLOAT16

          - sparsity_map_bytes can or cannot be missing; it is present if output sparsity is true

    ********************************************************************************************************************

    */

    const SEPModeInfo sepInfo18x18x64{
            true,              // sep on
            {18, 18, 1, 1},    // sep table,  4 bytes per element
            {512, 5, 600, 1},  // actual activator input,  same datatype as compute tensor input
    };

    const SEPModeInfo sepInfo64x64x512{
            true,              // sep on
            {64, 64, 1, 1},    // sep table,  4 bytes per element
            {128, 5, 512, 1},  // actual activator input,  same datatype as compute tensor input
    };

    const SEPModeInfo sepInfo64x128x512{
            true,                // sep on
            {64, 128, 1, 1},     // sep table,  4 bytes per element
            {1000, 15, 600, 1},  // actual activator input,  same datatype as compute tensor input
    };

    VPUNN::DPUWorkload wl_ref_18x18x64_SEP_active{wl_ref_18x18x64};
    wl_ref_18x18x64_SEP_active.sep_activators = sepInfo18x18x64;

    VPUNN::DPUWorkload wl_ref_64x64x512_SEP_active{wl_ref_64x64x512};
    wl_ref_64x64x512_SEP_active.sep_activators = sepInfo64x64x512;

    VPUNN::DPUWorkload wl_ref_64x128x512_SEP_active{wl_ref_64x128x512};
    wl_ref_64x128x512_SEP_active.sep_activators = sepInfo64x128x512;

    const TestsVector tests = {
            // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || output spars || test info || output0 memory expected ||    */
            
            //the dimensions of wl are small, so the memory will be small; when aligning, the memory value becomes the same, whether so is activated or not
            {{wl_ref_18x18x64,  true}, "output sparsity enable, data type UINT8",{16384}},
            {{wl_ref_18x18x64,  false}, "output sparsity disable, data type UINT8",{16384}},
            {{wl_ref_18x18x64_SEP_active,  true}, "output sparsity enable, data type UINT8, SEP active -- does not affect the memory",{16384}},
            {{wl_ref_18x18x64_SEP_active,  false}, "output sparsity disable, data type UINT8, SEP active -- does not affect the memory",{16384}},

            {{wl_ref_64x64x512,  false}, "output sparsity disable, data type UINT8",{229376}},
            {{wl_ref_64x64x512,  true}, "output sparsity enable, data type UINT8",{262144}},
            {{wl_ref_64x64x512_SEP_active,  true}, "output sparsity enable, data type UINT8, SEP active -- does not affect the memory",{262144}},
            {{wl_ref_64x64x512_SEP_active,  false}, "output sparsity disable, data type UINT8, SEP active -- does not affect the memory",{229376}},

            {{wl_ref_64x128x512, false}, "output sparsity disable, data type FLOAT16", {917504}},
            {{wl_ref_64x128x512, true},"output sparsity enable, data type FLOAT16",{966656}},
            {{wl_ref_64x128x512_SEP_active, true},"output sparsity enable, data type FLOAT16, SEP active -- does not affect the memory",{966656}},
            {{wl_ref_64x128x512_SEP_active, false},"output sparsity disable, data type FLOAT16, SEP active -- does not affect the memory",{917504}},
            // clang-format on
    };

    verify_output0_memory(tests);
}

TEST_F(DPU_OperationValidator_Test, Output_sparsity_memory_computation_test) {
    // we use these workloads to compute memory when we activate or not output sparsity for them
    // the last two of them have weight sparsity or input sparsity active, these wl are used to demonstrate that weight
    // and input sparsity does not affect output memory

    // this wl doesn't have any sparsity activated, small memory
    const VPUNN::DPUWorkload wl_ref_no_spars_small_mem{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(18, 18, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 18, 48, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 0, 0},                                               // padding
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

    // this wl doesn't have any sparsity activated, large memory
    const VPUNN::DPUWorkload wl_ref_no_spars_large_mem{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {3, 3},                                                         // kernels
            {3, 3},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                             // execution mode
            VPUNN::ActivationFunction::NONE,                                // activation
            0.0F,                                                           // act_sparsity
            0.0F,                                                           // weight_sparsity
            {swz_def, swz_def},                                             // input_swizzling
            {swz_def},                                                      // output_swizzling
            1,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false                                                           // weight_sparsity_enabled

    };

    // this wl have weight sparsity active, but input sparsity is not
    const VPUNN::DPUWorkload wl_ref_weight_spars_on{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {3, 3},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                          // execution mode
            VPUNN::ActivationFunction::NONE,                             // activation
            0.0F,                                                        // act_sparsity
            0.32F,                                                       // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            1,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            VPUNN::ISIStrategy::CLUSTERING,                              // isi_strategy
            true                                                         // weight_sparsity_enabled

    };

    // this wl have input sparsity active, but weight sparsity is not
    const VPUNN::DPUWorkload wl_ref_input_spars_on{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::FLOAT16, Layout::YZX, true)},  // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::FLOAT16)},                      // output dimensions
            {3, 3},                                                                            // kernels
            {3, 3},                                                                            // strides
            {0, 0, 0, 0},                                                                      // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                                // execution mode
            VPUNN::ActivationFunction::NONE,                                                   // activation
            0.57F,                                                                             // act_sparsity
            0.0F,                                                                              // weight_sparsity
            {swz_def, swz_def},                                                                // input_swizzling
            {swz_def},                                                                         // output_swizzling
            1,                                                                                 // output_write_tiles
            {0, 0, 0, 0},                                                                      // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                                    // isi_strategy
            false  // weight_sparsity_enabled

    };

    struct TestInput {
        VPUNN::DPUWorkload wl;        // the wl for which we compute memory
        bool output_sparsity_enable;  // activate/deactivate output sparsity; !!!affect output memory
    };

    struct TestExpectation {
        long long mem_size_exp;  // memory expected; it depends on test input
    };

    struct TestCase {
        TestInput t_in;
        std::string info = "";
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    // this lambda function verify if the output memory is computed correctly for different workloads that could/could
    // not have input or weight sparsity activated alongside output sparsity
    // weight and input sparsity does not affect output0 memory
    auto verify_output0_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << "Test case:"
                      << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref{t.t_in.wl};

            // set output sparsity
            wl_ref.outputs[0].set_sparsity(t.t_in.output_sparsity_enable);

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.output_0, t.t_exp.mem_size_exp) << mem << std::endl << t.t_exp.mem_size_exp << std::endl;
            }
            i++;
        }
    };

    /**************************************** HOW output0_memory is computed
    **********************************************
    !!! unaligned memory:
    sparsity_map_bytes = (W*H*C) / 8 and aligned to 16

      if data type is FLOAT16 or BFLOAT16
           unaligned_output0_memory = W*H*C * 2 + sparsity_map_bytes
      else: unaligned_output0_memory = W*H*C + sparsity_map_bytes

      W, H, C are workload's dimensions

    */
    const TestsVector tests = {
            // clang-format off
            /************************************************ TABLE HEADER ********************************************************/
            /*  || workload || output spars || test info || output0 memory expected ||    */
            
            //for wl_ref_18x18x64_no_spars you can see that output memory is the same regardless of whether output sparsity is active or not
            {{wl_ref_no_spars_small_mem,  true}, "output sparsity enable, data type UINT8",{16384}},
            {{wl_ref_no_spars_small_mem,  false}, "output sparsity disable, data type UINT8",{16384}},

            //for wl_ref_64x128x512_no_spars memory is large, so you can see a difference between memory values when output sparsity is active or not 
            {{wl_ref_no_spars_large_mem,  true}, "output sparsity enable, data type FLOAT16",{966656}},
            {{wl_ref_no_spars_large_mem,  false}, "output sparsity disable, data type FLOAT16",{917504}},

            //in this case this wl have the same dimensions as wl_ref_no_spars_large_mem 
            //as you can see output memory is the same in both cases, whether input sparsity is active or not, and whether output sparsity is active or not
            {{wl_ref_input_spars_on, false}, "output sparsity disable, input sparsity enable, data type FLOAT16", {917504}},
            {{wl_ref_input_spars_on, true},"output sparsity enable, input sparsity enable, data type FLOAT16",{966656}},

            //for this test case we have an wl with weight sparsity active 
            {{wl_ref_weight_spars_on,  false}, "output sparsity disable, weight sparsity enable, data type UINT8",{229376}},
            {{wl_ref_weight_spars_on,  true}, "output sparsity enable, weight sparsity enable, data type UINT8",{262144}},

            // clang-format on
    };

    verify_output0_memory(tests);
}

TEST_F(DPU_OperationValidator_Test, Elementwise_weightless_inplace_MemorySize_Test) {
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
TEST_F(DPU_OperationValidator_Test, elementwiseMemorySizeNoInout1ANdNoInplace_Test) {
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
    {  // no in place(due to layout) + no weights (due to layout change)
        auto wl{std::move(wl_no_in_place_layout)};
        EXPECT_TRUE(dut.is_supported(wl.device));
        long long in_mem_bytes{180 * 4 * 640 * 1};   // int8
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

TEST_F(DPU_OperationValidator_Test, convolutionMemorySize_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 0},                                                // padding
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

    const auto input_0_raw{14 * 14 * 512};
    const auto output_0_raw{7 * 7 * 512};
    const auto input_1_raw{3 * 3 * 512 * 512 + 512 * 16};

    const auto input_0{align(input_0_raw, VPUDevice::VPU_2_7)};
    const auto output_0{align(output_0_raw, VPUDevice::VPU_2_7)};
    const auto input_1{align(input_1_raw, VPUDevice::VPU_2_7)};

    EXPECT_TRUE(isAligned(input_0, VPUDevice::VPU_2_7));
    EXPECT_TRUE(isAligned(output_0, VPUDevice::VPU_2_7));
    EXPECT_TRUE(isAligned(input_1, VPUDevice::VPU_2_7));

    {  // no ISI strategy
        auto wl{wl_ref};
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
        EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        EXPECT_EQ(mem.input_1, input_1) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 + mem.output_0) << mem << std::endl;
    }

    {  // SOH-> activators are halved in contribution (NOT any more)
        auto wl{wl_ref};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
        EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        EXPECT_EQ(mem.input_1, input_1) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 / 1 + mem.input_1 + mem.output_0) << mem << std::endl;
    }
    {  // SOK-> weights are  is halved in contribution (NOT any more)
        auto wl{std::move(wl_ref)};
        wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        EXPECT_TRUE(dut.is_supported(wl.device));

        VPUNN::MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
        EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        EXPECT_EQ(mem.input_1, input_1) << mem << std::endl;
        EXPECT_EQ(mem.cmx, cmx_overhead + mem.input_0 + mem.input_1 / 1 + mem.output_0) << mem << std::endl;
    }
}

TEST_F(DPU_OperationValidator_Test, HALOconvolutionMemorySize_SmokeTest) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},    // output dimensions
            {3, 3},                                                      // kernels
            {2, 2},                                                      // strides
            {1, 0, 1, 0},                                                // padding TBLR
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

    const HaloWorkload h1_ref{
            {0, 1, 0, 0},  // HaloInfo2D input_0_halo, TBLR
            {0, 0, 0, 0},  // HaloInfo2D output_0_halo TBLR

            {0, 0, 0, 0},       // HaloInfo2D output_0_halo_broadcast_cnt TBLR
            {7, 0, 0, 0, 0, 0}  // HaloInfo3D output_0_inbound_halo TBLR FB
    };

    const HaloWorkload h1_ref2{
            {0, 1, 0, 0},  // HaloInfo2D input_0_halo, TBLR
            {1, 2, 3, 4},  // HaloInfo2D output_0_halo TBLR

            {10, 20, 30, 40},   // HaloInfo2D output_0_halo_broadcast_cnt TBLR
            {7, 0, 0, 0, 0, 0}  // HaloInfo3D output_0_inbound_halo TBLR FB
    };

    VPUNN::DPUWorkload wl_ref_halo{wl_ref};
    wl_ref_halo.halo = h1_ref;

    VPUNN::DPUWorkload wl_ref_halo2{wl_ref};
    wl_ref_halo2.halo = h1_ref2;  // same as 1 but with irrelevant data for memory in

    const auto input_0_raw{14 * 14 * 512};
    const auto output_0_raw{7 * 7 * 512};
    // const auto input_1_raw{3 * 3 * 512 * 512 + 512 * 16};

    const auto input_0{align(input_0_raw, VPUDevice::VPU_2_7)};
    const auto output_0{align(output_0_raw, VPUDevice::VPU_2_7)};
    // const auto input_1{align(input_1_raw)};

    EXPECT_TRUE(isAligned(input_0, VPUDevice::VPU_2_7));
    EXPECT_TRUE(isAligned(output_0, VPUDevice::VPU_2_7));
    //    EXPECT_TRUE(isAligned(input_1));
    ASSERT_TRUE(dut.is_supported(wl_ref.device));

    {
        MemorySize mem_clean;
        {
            MemorySize& mem{mem_clean};
            auto wl{std::move(wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
            EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        }

        MemorySize mem_H1;
        {
            MemorySize& mem{mem_H1};
            auto wl{std::move(wl_ref_halo)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_LT(mem.input_0, input_0 - (14 * 512)) << mem << std::endl;  // 1 row lwss
            EXPECT_EQ(mem.output_0, output_0 * 2) << mem << std::endl;         // doubling the dim
        }

        EXPECT_GT(mem_clean.input_0, mem_H1.input_0) << mem_clean << mem_H1 << std::endl;
        EXPECT_LT(mem_clean.output_0, mem_H1.output_0) << mem_clean << mem_H1 << std::endl;

        EXPECT_EQ(mem_clean.input_1, mem_H1.input_1) << mem_clean << mem_H1 << std::endl;

        MemorySize mem_H2;
        {
            MemorySize& mem{mem_H2};
            auto wl{std::move(wl_ref_halo2)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_LT(mem.input_0, input_0 - (14 * 512)) << mem << std::endl;  // 1 row lwss
            EXPECT_EQ(mem.output_0, output_0 * 2) << mem << std::endl;         // doubling the dim
        }

        EXPECT_EQ(mem_H2.input_0, mem_H1.input_0) << mem_clean << mem_H1 << std::endl;
        EXPECT_EQ(mem_H2.output_0, mem_H1.output_0) << mem_clean << mem_H1 << std::endl;

        EXPECT_EQ(mem_H2.input_1, mem_H1.input_1) << mem_clean << mem_H1 << std::endl;
    }
}

// test for Halo input values
// we are testing if the input memory has been calculated the right way
TEST_F(DPU_OperationValidator_Test, InputHALOTest) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
    };

    struct TestExpectation {
        long long mem_size_exp_not_aligned;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // this lambda function checks if memory is calculated correctly when using halo aspects
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << "Case " << t.test_case << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref_halo)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.input_0, align(t.t_exp.mem_size_exp_not_aligned, wl.device))
                        << mem << std::endl
                        << align(t.t_exp.mem_size_exp_not_aligned, wl.device) << std::endl;
            }
            i++;
        }
    };

    // when we test input_0_halo: output_0_halo, output_0_halo_broadcast_cnt and output_0_inbound_halo values (TBLR /
    // TBLRFB) should be 0
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};  // TBLR output_0_halo, output_0_halo_broadcast_cnt
    const HaloWorkload::HaloInfoHWC output_inbound_halo{0, 0, 0, 0, 0, 0};  // TBLRFB output_0_inbound_halo

    // here we have some test cases where halo input values (TBLR) are smaller than number of rows or columns of our
    // tensor
    //  we are testing if memory tensor is as expected
    const TestsVector tests_normal_cases = {
            {{wl_ref, {{0, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {2097152}, "No halo information"},
            {{wl_ref, {{1, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {2064384}, "Top halo"},
            {{wl_ref, {{0, 1, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {2064384}, "Btm halo"},
            {{wl_ref, {{4, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {1966080}, "Top halo"},
            {{wl_ref, {{7, 7, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {1638400}, "Top+Btm halo"},
            {{wl_ref, {{5, 5, 5, 5}, output_halo, output_halo, output_inbound_halo}},
             {1492992},
             "Top+Btm+Left+Right halo"},
            {{wl_ref, {{60, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {131072}, "Top halo"},
            {{wl_ref, {{0, 60, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {131072}, "Btm halo"},
            {{wl_ref, {{31, 31, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {65536},
             "Top+Btm halo (compute tensor is 2 rows)"},
            {{wl_ref, {{0, 0, 31, 31}, output_halo, output_halo, output_inbound_halo}},
             {65536},
             "Left+Right halo (compute tensor is 2 columns)"},
            {{wl_ref, {{0, 18, 9, 0}, output_halo, output_halo, output_inbound_halo}}, {1295360}, "Btm+Left halo"},
            {{wl_ref, {{62, 1, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {32768},
             "Top+Btm halo (compute tensor is 1 row)"},
            {{wl_ref, {{0, 0, 1, 62}, output_halo, output_halo, output_inbound_halo}},
             {32768},
             "Left+Right- halo (compute tensor is 1 column)"},
            {{wl_ref, {{63, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {32768},
             "Top halo (compute tensor is 1 row)"},
            {{wl_ref, {{0, 0, 63, 0}, output_halo, output_halo, output_inbound_halo}},
             {32768},
             "Left halo (compute tensor is 1 column)"}

    };

    // here we have some test cases where halo input values (TBLR) are negative, combinations of negative and positive
    // numbers, or bigger than number of rows or columns of our tensor when TBLR (their sum, or sum of 2 of them, or
    // just a value of one of them) are (is) equal with number of rows (TB) or columns (LR) of our tensor, the expected
    // value of memory is 0
    // when TBLR are negative values, the value of calculated memory should be larger than initial value
    // when TBLR (their sum, or sum of 2 of them, or just a value of one of them) are (is) greater than number of rows
    // (TB) or columns (LR) of our tensor, the expected value of memory is 0
    const TestsVector tests_extreme_cases = {
            {{wl_ref, {{64, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Top halo is equal with tensor height"},
            {{wl_ref, {{0, 64, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Btm halo is equal with tensor height"},
            {{wl_ref, {{0, 0, 0, 64}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Right halo is equal with tensor width"},
            {{wl_ref, {{32, 32, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Top+Btm halo, their sum is equal with tensor height"},
            {{wl_ref, {{-1, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {2129920}, "Negative Top halo"},
            {{wl_ref, {{-5, -3, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {2359296},
             "Negative Top+Btm halo"},
            {{wl_ref, {{-1, 2, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {2064384},
             "Negative Top + Positive Btm halo, but their sum is positive "},
            {{wl_ref, {{0, 0, -10, 5}, output_halo, output_halo, output_inbound_halo}},
             {2260992},
             "Negative Left + Positive Right halo, but their sum is negative"},
            {{wl_ref, {{0, 0, -2, -2}, output_halo, output_halo, output_inbound_halo}},
             {2228224},
             "Negative Left+Right halo"},
            {{wl_ref, {{0, 0, 0, -1}, output_halo, output_halo, output_inbound_halo}},
             {2129920},
             "Negative Right halo"},
            {{wl_ref, {{65, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Top halo bigger than tensor height"},  // expected memory value is 0, not negative (-32768)
            {{wl_ref, {{33, 34, 0, 0}, output_halo, {0, 0, 0, 0}, output_inbound_halo}},
             {0},
             "Top+Btm halo, their sum is bigger than tensor height"},  // expected memory value is 0, not negative
                                                                       // (-98304)
            {{wl_ref, {{0, 0, 0, 0, -1, 0}, output_halo, output_halo, output_inbound_halo}},
             {2097152 + (64 * 64)},
             "Negative channels  1 halo"},
            {{wl_ref, {{0, 0, 0, 0, -5, 0}, output_halo, output_halo, output_inbound_halo}},
             {2097152 + 5 * (64 * 64)},
             "Negative channels  1 halo"}};

    // some test cases where we have front and back values for input_0_halo
    const TestsVector tests_front_back = {
            {{wl_ref, {{0, 0, 0, 0, 1, 0}, output_halo, output_halo, output_inbound_halo}},
             {2093056},
             "Front halo information"},
            {{wl_ref, {{0, 0, 0, 0, 0, 1}, output_halo, output_halo, output_inbound_halo}},
             {2093056},
             "Back halo information"},
            {{wl_ref, {{0, 0, 0, 0, 5, 7}, output_halo, output_halo, output_inbound_halo}},
             {2048000},
             "Front and Back halo information"},
            {{wl_ref, {{0, 0, 0, 0, 10, 12}, output_halo, output_halo, output_inbound_halo}},
             {2007040},
             "Front and Back halo information"},
            {{wl_ref, {{0, 0, 0, 0, -1, 0}, output_halo, output_halo, output_inbound_halo}},
             {2101248},
             "Negative Front halo information"},
            {{wl_ref, {{0, 0, 0, 0, -2, 3}, output_halo, output_halo, output_inbound_halo}},
             {2093056},
             "Negative front and positive back halo information"},
            {{wl_ref, {{0, 0, 0, 0, -5, -5}, output_halo, output_halo, output_inbound_halo}},
             {2138112},
             "Negative front and back halo information"},
            {{wl_ref, {{64, 0, 0, 0, 0, 512}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Top halo is equal with tensor height, and back halo is equal with tensor channels"},
            {{wl_ref, {{0, 0, 0, 0, 512, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Front halo information is equal with tensor channels"},
            {{wl_ref, {{20, 40, 0, 0, 500, 10}, output_halo, output_halo, output_inbound_halo}},
             {512},
             "Top, Btm, Front and Back halo information"},
            {{wl_ref, {{0, 0, -6, -1, -12, -9}, output_halo, output_halo, output_inbound_halo}},
             {2421952},
             "Negative Left, Right, Front and Back halo information"},
            {{wl_ref, {{-2, -7, 0, 0, -100, -67}, output_halo, output_halo, output_inbound_halo}},
             {3172288},
             "Negative Top, Btm, Front and Back halo information"},
            {{wl_ref, {{0, 0, 10, 13, 27, 91}, output_halo, output_halo, output_inbound_halo}},
             {1033856},
             "Left, Right, Front and Back halo information"}

    };
    verify_memory(tests_normal_cases);
    verify_memory(tests_extreme_cases);
    verify_memory(tests_front_back);
}

// output Halo have 3 important fields: output_0_halo (TBLR), output_0_halo_broadcast_cnt (these 2 does not affect
// the value of memory) and output_0_inbound_halo (this affect the value of memory)
// here are some tests for output_0_halo and output_0_halo_broadcast_cnt, even if their TBLR values are not 0, memory
// value is the same
TEST_F(DPU_OperationValidator_Test, OutputHALOTest) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
    };

    struct TestExpectation {
        long long mem_size_exp_not_aligned;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // this lambda function checks if memory is calculated correctly when using halo aspects
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << t.test_case << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref_halo)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.output_0, align(t.t_exp.mem_size_exp_not_aligned, wl.device))
                        << mem << std::endl
                        << align(t.t_exp.mem_size_exp_not_aligned, wl.device) << std::endl;
            }
            i++;
        }
    };

    const HaloWorkload::HaloInfoHWC input_halo{0, 0, 0, 0, 0, 0};   // TBLR input_0_halo
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};  // TBLR: output_0_halo_broadcast_cnt, output_0_halo
    const HaloWorkload::HaloInfoHWC output_inbound_halo{0, 0, 0, 0, 0, 0};  // TBLRFB output_0_inbound_halo

    // here are some test cases for output halo: output_0_halo (TBLR)
    // input_0_halo, output_0_halo_broadcast_cnt and output_0_inbound_halo values (TBLR / TBLRFB) should be 0
    // you can see that for different values for TBLR our memory values are always the same, like TBLR are 0, 0, 0, 0
    const TestsVector tests_normal_cases_output_halo = {
            {{wl_ref, {input_halo, {0, 0, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case no halo information"},
            {{wl_ref, {input_halo, {1, 0, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Top halo"},
            {{wl_ref, {input_halo, {0, 1, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Btm halo"},
            {{wl_ref, {input_halo, {1, 1, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Top+Btm halo"},
            {{wl_ref, {input_halo, {0, 0, 0, 2}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Right halo"},
            {{wl_ref, {input_halo, {0, 0, 2, 3}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Left+Right halo"},
            {{wl_ref, {input_halo, {10, 10, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Top+Btm halo"},
            {{wl_ref, {input_halo, {0, 0, 10, 10}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Left+Right halo"},
            {{wl_ref, {input_halo, {2, 2, 2, 2}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Top+Btm+Left+Right halo"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 1, 1}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: channels halo"},
    };

    // here TBLR have some extreme values like negative ones or values that are bigger than number of rows or columns of
    // our tensor but the memory values are always the same, like TBLR are 0, 0, 0, 0,
    // precondition: halo_0_output>0, but negative values are treated like 0
    const TestsVector tests_extreme_cases_output_halo = {
            {{wl_ref, {input_halo, {21, 0, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Top halo is equal with tensor height"},
            {{wl_ref, {input_halo, {0, 21, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Btm halo is equal with tensor width"},
            {{wl_ref, {input_halo, {0, 0, 0, 21}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Right halo is equal with tensor height "},
            {{wl_ref, {input_halo, {10, 11, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Top+Btm halo, but their sum is equal with tensor height"},
            {{wl_ref, {input_halo, {-1, 0, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Negative Top halo"},
            {{wl_ref, {input_halo, {-1, -1, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Negative Top+Btm halo"},
            {{wl_ref, {input_halo, {-1, 3, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo:Case Negative Top + Positive Btm halo"},
            {{wl_ref, {input_halo, {0, 0, -2, -4}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Negative Left+Right halo"},
            {{wl_ref, {input_halo, {22, 0, 0, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Top halo is bigger than tensor height"},
            {{wl_ref, {input_halo, {0, 0, 22, -5}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Positive Left +Negative Right halo, but Left halo is bigger than tensor width and "
             "tehir sum is positive and smaller than tensor width"},
            {{wl_ref, {input_halo, {0, 0, 11, 11}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Case Left+Right halo, but their sum is bigger than tensor width"}};

    // some test cases where we have front and back values for output_0_halo
    // precondition: halo_0_output>0, but negative values are treated like 0
    const TestsVector tests_front_back_output_halo = {
            {{wl_ref, {input_halo, {0, 0, 0, 0, 1, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Front halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 0, 1}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 5, 7}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 10, 12}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, -1, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Negative Front halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, -2, 3}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Negative front and positive back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, -5, -5}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Negative front and back halo information"},
            {{wl_ref, {input_halo, {64, 0, 0, 0, 0, 512}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Top halo is equal with tensor height, and back halo is equal with tensor channels"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 512, 0}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Front halo information is equal with tensor channels"},
            {{wl_ref, {input_halo, {20, 40, 0, 0, 500, 10}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, -6, -1, -12, -9}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Negative Left, Right, Front and Back halo information"},
            {{wl_ref, {input_halo, {-2, -7, 0, 0, -100, -67}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Negative Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 10, 13, 27, 91}, output_halo, output_inbound_halo}},
             {225792},
             "output_0_halo: Left, Right, Front and Back halo information"}};

    verify_memory(tests_normal_cases_output_halo);
    verify_memory(tests_extreme_cases_output_halo);
    verify_memory(tests_front_back_output_halo);

    // here are some test cases for output halo: output_0_halo_broadcast_cnt
    // input_0_halo, output_0_halo and output_0_inbound_halo values (TBLR /TBLRFB) should be 0
    // you can see that for different values for TBLR our memory values are always the same, like TBLR are 0, 0, 0, 0
    const TestsVector tests_normal_cases_output_broadcast = {
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case No halo information"},
            {{wl_ref, {input_halo, output_halo, {1, 0, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top halo"},
            {{wl_ref, {input_halo, output_halo, {0, 1, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Btm halo"},
            {{wl_ref, {input_halo, output_halo, {1, 1, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 2}, output_inbound_halo}},
             {225792},
             "broadcast: Case Right halo"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 2, 3}, output_inbound_halo}},
             {225792},
             "broadcast: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, {10, 10, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 10, 10}, output_inbound_halo}},
             {225792},
             "broadcast: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, {2, 2, 2, 2}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top+Btm+Left+Right halo"}};

    // here TBLR have some extreme values like negative ones or values that are bigger than number of rows or columns of
    // our tensor but the memory values are always the same, like TBLR are 0, 0, 0, 0,
    // precondition: halo_broadcast>0, but negative values are treated like 0
    const TestsVector tests_extreme_cases_output_broadcast = {
            {{wl_ref, {input_halo, output_halo, {21, 0, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top halo is equal with tensor height"},
            {{wl_ref, {input_halo, output_halo, {0, 21, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Btm halo is equal with tensor height "},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 21}, output_inbound_halo}},
             {225792},
             "broadcast: Case  Right halo is equal with tensor width"},
            {{wl_ref, {input_halo, output_halo, {10, 11, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top+Btm halo, but their sum is equal with tensor height"},
            {{wl_ref, {input_halo, output_halo, {-1, 0, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Negative Top halo"},
            {{wl_ref, {input_halo, output_halo, {-1, -1, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Negative Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, {-1, 3, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Negative Top + Positive Btm, but their sum is positive"},
            {{wl_ref, {input_halo, output_halo, {0, 0, -2, -4}, output_inbound_halo}},
             {225792},
             "broadcast: Case Negative Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, {22, 0, 0, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Case Top halo is bigger than tensor height"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 22, -5}, output_inbound_halo}},
             {225792},
             "broadcast: Case Positive Left + Negative Right halo, but Left is bigger than halo width and their sum is "
             "positive "},
            {{wl_ref, {input_halo, output_halo, {0, 0, 11, 11}, output_inbound_halo}},
             {225792},
             "broadcast: Case Left+Right halo, but their sum is bigger than tensor width"}};

    // some test cases where we have front and back values for broadcast
    // precondition: halo_broadcast>0, but negative values are treated like 0
    const TestsVector tests_front_back_broadcast = {
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 1, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Front halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 0, 1}, output_inbound_halo}},
             {225792},
             "broadcast: Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 5, 7}, output_inbound_halo}},
             {225792},
             "broadcast: Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 10, 12}, output_inbound_halo}},
             {225792},
             "broadcast: Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, -1, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Negative Front halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, -2, 3}, output_inbound_halo}},
             {225792},
             "broadcast: Negative front and positive back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, -5, -5}, output_inbound_halo}},
             {225792},
             "broadcast: Negative front and back halo information"},
            {{wl_ref, {input_halo, output_halo, {64, 0, 0, 0, 0, 512}, output_inbound_halo}},
             {225792},
             "broadcast: Top halo is equal with tensor height, and back halo is equal with tensor channels"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 512, 0}, output_inbound_halo}},
             {225792},
             "broadcast: Front halo information is equal with tensor channels"},
            {{wl_ref, {input_halo, output_halo, {20, 40, 0, 0, 500, 10}, output_inbound_halo}},
             {225792},
             "broadcast: Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, -6, -1, -12, -9}, output_inbound_halo}},
             {225792},
             "broadcast: Negative Left, Right, Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {-2, -7, 0, 0, -100, -67}, output_inbound_halo}},
             {225792},
             "broadcast: Negative Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 10, 13, 27, 91}, output_inbound_halo}},
             {225792},
             "broadcast: Left, Right, Front and Back halo information"}};
    verify_memory(tests_normal_cases_output_broadcast);
    verify_memory(tests_extreme_cases_output_broadcast);
    verify_memory(tests_front_back_broadcast);
}

// output Halo have 3 important fields: output_0_halo (TBLR), output_0_halo_broadcast_cnt (these 2 does not affect
// the value of memory) and output_0_inbound_halo (this affect the value of memory)
// here are some tests for output_0_inbound_halo , for different values TBLRFB, memory value also varies
TEST_F(DPU_OperationValidator_Test, OutputHALO_Inbound_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
    };

    struct TestExpectation {
        long long mem_size_exp_not_aligned;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // this lambda function checks if memory is calculated correctly when using halo aspects
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << t.test_case << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref_halo)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.output_0, align(t.t_exp.mem_size_exp_not_aligned, wl.device))
                        << mem << std::endl
                        << align(t.t_exp.mem_size_exp_not_aligned, wl.device) << std::endl;
            }
            i++;
        }
    };

    const HaloWorkload::HaloInfoHWC input_halo{0, 0, 0, 0, 0, 0};   // TBLR input_0_halo
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};  // TBLR: output_0_halo_broadcast_cnt, output_0_halo

    // you can see that for different values for TBLR our calculated memory values are grater than these of the initial
    // memory
    const TestsVector tests_normal_cases_output_inbound = {
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 0, 0, 0}}},
             {225792},
             "inbound: Case No halo information"},
            {{wl_ref, {input_halo, output_halo, output_halo, {1, 0, 0, 0, 0, 0}}}, {236544}, "inbound: Case Top halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 1, 0, 0, 0, 0}}}, {236544}, "inbound: Case Btm halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {1, 1, 0, 0, 0, 0}}},
             {247296},
             "inbound: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 2, 0, 0}}},
             {247296},
             "inbound: Case Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 2, 3, 0, 0}}},
             {279552},
             "inbound: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {10, 10, 0, 0, 0, 0}}},
             {440832},
             "inbound: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 10, 10, 0, 0}}},
             {440832},
             "inbound: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {2, 2, 2, 2, 0, 0}}},
             {311808},
             "inbound: Case Top+Btm+Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {1, 2, 0, 0, 9, 0}}},
             {262602},
             "inbound: Case Top+Btm+Front halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 0, 6, 0}}},
             {228438},
             "inbound: Case Front halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 0, 12, 0}}},
             {231084},
             "inbound: Case Front halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 2, 2, 3, 2}}},
             {271005},
             "inbound: Case Left+Right+Front+Back halo"}};

    verify_memory(tests_normal_cases_output_inbound);
}

// here are some tests where wl Height and Width are not equal
TEST_F(DPU_OperationValidator_Test, InputHALOTest_different_H_and_W_wl) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::UINT8)},     // input dimensions WHCB
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
    };

    struct TestExpectation {
        long long mem_size_exp_not_aligned;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // this lambda function checks if memory is calculated correctly when using halo aspects
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << t.test_case << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref_halo)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.input_0, align(t.t_exp.mem_size_exp_not_aligned, wl.device))
                        << mem << std::endl
                        << align(t.t_exp.mem_size_exp_not_aligned, wl.device) << std::endl;
            }
            i++;
        }
    };

    // when we test input_0_halo: output_0_halo, output_0_halo_broadcast_cnt and output_0_inbound_halo values (TBLR /
    // TBLRFB) should be 0
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};  // TBLR output_0_halo, output_0_halo_broadcast_cnt
    const HaloWorkload::HaloInfoHWC output_inbound_halo{0, 0, 0, 0, 0, 0};  // TBLRFB output_0_inbound_halo

    // here we have some test cases where halo input values (TBLR) are smaller than number of rows or columns of our
    // tensor
    //  we are testing if memory tensor is as expected
    const TestsVector tests = {
            {{wl_ref, {{0, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {4191304}, "No halo information"},
            {{wl_ref, {{1, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {4161536}, "Top halo"},
            {{wl_ref, {{0, 1, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {4161536}, "Btm halo"},
            {{wl_ref, {{4, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {4063232}, "Top halo"},
            {{wl_ref, {{7, 7, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {3735552}, "Top+Btm halo"},
            {{wl_ref, {{5, 5, 5, 5}, output_halo, output_halo, output_inbound_halo}},
             {3262464},
             "Top+Btm+Left+Right halo"},
            {{wl_ref, {{120, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}}, {262144}, "Top halo"},
            {{wl_ref, {{0, 126, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {65536},
             "Btm halo (compute tensor is 2 rows)"},
            {{wl_ref, {{63, 63, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {65536},
             "Top+Btm halo (compute tensor is 2 rows)"},
            {{wl_ref, {{0, 0, 31, 31}, output_halo, output_halo, output_inbound_halo}},
             {131072},
             "Left+Right halo (compute tensor is 2 columns)"},
            {{wl_ref, {{128, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Top halo is equal with height"},
            {{wl_ref, {{126, 1, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {32768},
             "Top+Btm halo (compute tensor is 1 row)"},
            {{wl_ref, {{0, 0, 1, 62}, output_halo, output_halo, output_inbound_halo}},
             {65536},
             "Left+Right- halo (compute tensor is 1 column)"},
            {{wl_ref, {{127, 0, 0, 0}, output_halo, output_halo, output_inbound_halo}},
             {32768},
             "Top halo (compute tensor is 1 row)"},
            {{wl_ref, {{0, 0, 63, 0}, output_halo, output_halo, output_inbound_halo}},
             {65536},
             "Left halo (compute tensor is 1 column)"}

    };

    // some test cases where we have front and back values for input_0_halo
    const TestsVector tests_front_back = {
            {{wl_ref, {{0, 0, 0, 0, 1, 0}, output_halo, output_halo, output_inbound_halo}},
             {4186112},
             "Front halo information"},
            {{wl_ref, {{0, 0, 0, 0, 0, 1}, output_halo, output_halo, output_inbound_halo}},
             {4186112},
             "Back halo information"},
            {{wl_ref, {{0, 0, 0, 0, 5, 7}, output_halo, output_halo, output_inbound_halo}},
             {4096000},
             "Front and Back halo information"},
            {{wl_ref, {{0, 0, 0, 0, 10, 12}, output_halo, output_halo, output_inbound_halo}},
             {4014080},
             "Front and Back halo information"},
            {{wl_ref, {{0, 0, 0, 0, -1, 0}, output_halo, output_halo, output_inbound_halo}},
             {4202496},
             "Negative Front halo information"},
            {{wl_ref, {{0, 0, 0, 0, -2, 3}, output_halo, output_halo, output_inbound_halo}},
             {4186112},
             "Negative front and positive back halo information"},
            {{wl_ref, {{0, 0, 0, 0, -5, -5}, output_halo, output_halo, output_inbound_halo}},
             {4276224},
             "Negative front and back halo information"},
            {{wl_ref, {{64, 0, 0, 0, 0, 512}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Top halo is equal with tensor height, and back halo is equal with tensor channels"},
            {{wl_ref, {{0, 0, 0, 0, 512, 0}, output_halo, output_halo, output_inbound_halo}},
             {0},
             "Front halo information is equal with tensor channels"},
            {{wl_ref, {{20, 40, 0, 0, 500, 10}, output_halo, output_halo, output_inbound_halo}},
             {8704},
             "Top, Btm, Front and Back halo information"},
            {{wl_ref, {{0, 0, -6, -1, -12, -9}, output_halo, output_halo, output_inbound_halo}},
             {4843904},
             "Negative Left, Right, Front and Back halo information"},
            {{wl_ref, {{-2, -7, 0, 0, -100, -67}, output_halo, output_halo, output_inbound_halo}},
             {5953472},
             "Negative Top, Btm, Front and Back halo information"},
            {{wl_ref, {{0, 0, 10, 13, 27, 91}, output_halo, output_halo, output_inbound_halo}},
             {2067712},
             "Left, Right, Front and Back halo information"}};

    verify_memory(tests);
    verify_memory(tests_front_back);
}

TEST_F(DPU_OperationValidator_Test, OutputHALOTest_different_H_and_W_wl) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
    };

    struct TestExpectation {
        long long mem_size_exp_not_aligned;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // this lambda function checks if memory is calculated correctly when using halo aspects
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << t.test_case << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref_halo)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.output_0, align(t.t_exp.mem_size_exp_not_aligned, wl.device))
                        << mem << std::endl
                        << align(t.t_exp.mem_size_exp_not_aligned, wl.device) << std::endl;
            }
            i++;
        }
    };

    const HaloWorkload::HaloInfoHWC input_halo{0, 0, 0, 0, 0, 0};   // TBLR input_0_halo
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};  // TBLR: output_0_halo_broadcast_cnt, output_0_halo
    const HaloWorkload::HaloInfoHWC output_inbound_halo{0, 0, 0, 0, 0, 0};  // TBLRFB output_0_inbound_halo

    // here are some test cases for output halo: output_0_halo (TBLR)
    // input_0_halo, output_0_halo_broadcast_cnt and output_0_inbound_halo values (TBLR / TBLRFB) should be 0
    // you can see that for different values for TBLR our memory values are always the same, like TBLR are 0, 0, 0, 0
    const TestsVector tests_output_halo = {{{wl_ref, {input_halo, {0, 0, 0, 0}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case no halo information"},
                                           {{wl_ref, {input_halo, {1, 0, 0, 0}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Top halo"},
                                           {{wl_ref, {input_halo, {0, 1, 0, 0}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Btm halo"},
                                           {{wl_ref, {input_halo, {1, 1, 0, 0}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Top+Btm halo"},
                                           {{wl_ref, {input_halo, {0, 0, 0, 2}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Right halo"},
                                           {{wl_ref, {input_halo, {0, 0, 2, 3}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Left+Right halo"},
                                           {{wl_ref, {input_halo, {54, 10, 0, 0}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Top+Btm halo"},
                                           {{wl_ref, {input_halo, {0, 0, 30, 10}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Left+Right halo"},
                                           {{wl_ref, {input_halo, {102, 2, 2, 60}, output_halo, output_inbound_halo}},
                                            {451584},
                                            "output_0_halo: Case Top+Btm+Left+Right halo"}};

    // some test cases where we have front and back values for output_0_halo
    // precondition: halo_0_output>0, but negative values are treated like 0
    const TestsVector tests_front_back_output = {
            {{wl_ref, {input_halo, {0, 0, 0, 0, 1, 0}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Front halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 0, 1}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 5, 7}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 10, 12}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, -1, 0}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Negative Front halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, -2, 3}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Negative front and positive back halo information"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, -5, -5}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Negative front and back halo information"},
            {{wl_ref, {input_halo, {64, 0, 0, 0, 0, 512}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Top halo is equal with tensor height, and back halo is equal with tensor channels"},
            {{wl_ref, {input_halo, {0, 0, 0, 0, 512, 0}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Front halo information is equal with tensor channels"},
            {{wl_ref, {input_halo, {20, 40, 0, 0, 500, 10}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, -6, -1, -12, -9}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Negative Left, Right, Front and Back halo information"},
            {{wl_ref, {input_halo, {-2, -7, 0, 0, -100, -67}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Negative Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, {0, 0, 10, 13, 27, 91}, output_halo, output_inbound_halo}},
             {451584},
             "output_0_halo: Case Left, Right, Front and Back halo information"}};

    verify_memory(tests_output_halo);
    verify_memory(tests_front_back_output);

    // here are some test cases for output halo: output_0_halo_broadcast_cnt
    // input_0_halo, output_0_halo and output_0_inbound_halo values (TBLR /TBLRFB) should be 0
    // you can see that for different values for TBLR our memory values are always the same, like TBLR are 0, 0, 0, 0
    const TestsVector tests_output_broadcast = {
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case No halo information"},
            {{wl_ref, {input_halo, output_halo, {1, 0, 0, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Top halo"},
            {{wl_ref, {input_halo, output_halo, {0, 1, 0, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Btm halo"},
            {{wl_ref, {input_halo, output_halo, {1, 1, 0, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 2}, output_inbound_halo}},
             {451584},
             "broadcast: Case Right halo"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 12, 3}, output_inbound_halo}},
             {451584},
             "broadcast: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, {70, 20, 0, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 10, 43}, output_inbound_halo}},
             {451584},
             "broadcast: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, {102, 2, 52, 2}, output_inbound_halo}},
             {451584},
             "broadcast: Case Top+Btm+Left+Right halo"}};

    // some test cases where we have front and back values for broadcast
    // precondition: halo_broadcast>0, but negative values are treated like 0
    const TestsVector tests_front_back_broadcast = {
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 1, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Front halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 0, 1}, output_inbound_halo}},
             {451584},
             "broadcast: Case Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 5, 7}, output_inbound_halo}},
             {451584},
             "broadcast: Case Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 10, 12}, output_inbound_halo}},
             {451584},
             "broadcast: Case Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, -1, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Negative Front halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, -2, 3}, output_inbound_halo}},
             {451584},
             "broadcast: Case Negative front and positive back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, -5, -5}, output_inbound_halo}},
             {451584},
             "broadcast: Case Negative front and back halo information"},
            {{wl_ref, {input_halo, output_halo, {64, 0, 0, 0, 0, 512}, output_inbound_halo}},
             {451584},
             "broadcast: Case Top halo is equal with tensor height, and back halo is equal with tensor channels"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 0, 0, 512, 0}, output_inbound_halo}},
             {451584},
             "broadcast: Case Front halo information is equal with tensor channels"},
            {{wl_ref, {input_halo, output_halo, {20, 40, 0, 0, 500, 10}, output_inbound_halo}},
             {451584},
             "broadcast: Case Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, -6, -1, -12, -9}, output_inbound_halo}},
             {451584},
             "broadcast: Case Negative Left, Right, Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {-2, -7, 0, 0, -100, -67}, output_inbound_halo}},
             {451584},
             "broadcast: Case Negative Top, Btm, Front and Back halo information"},
            {{wl_ref, {input_halo, output_halo, {0, 0, 10, 13, 27, 91}, output_inbound_halo}},
             {451584},
             "broadcast: Case Left, Right, Front and Back halo information"}};

    verify_memory(tests_output_broadcast);
    verify_memory(tests_front_back_broadcast);
}

// output Halo have 3 important fields: output_0_halo (TBLR), output_0_halo_broadcast_cnt (these 2 does not affect
// the value of memory) and output_0_inbound_halo (this affect the value of memory)
// here are some tests for output_0_inbound_halo , for different values TBLRFB, memory value also varies
TEST_F(DPU_OperationValidator_Test, OutputHALO_Inbound_Test_different_H_and_W_wl) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::UINT8)},     // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
    };

    struct TestExpectation {
        long long mem_size_exp_not_aligned;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;
    // this lambda function checks if memory is calculated correctly when using halo aspects
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << t.test_case << " " << i << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref_halo)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.output_0, align(t.t_exp.mem_size_exp_not_aligned, wl.device))
                        << mem << std::endl
                        << align(t.t_exp.mem_size_exp_not_aligned, wl.device) << std::endl;
            }
            i++;
        }
    };

    const HaloWorkload::HaloInfoHWC input_halo{0, 0, 0, 0, 0, 0};   // TBLR input_0_halo
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};  // TBLR: output_0_halo_broadcast_cnt, output_0_halo

    // you can see that for different values for TBLR our calculated memory values are grater than these of the initial
    // memory
    const TestsVector tests_output_inbound = {
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 0, 0, 0}}},
             {451584},
             "inbound: Case No halo information"},
            {{wl_ref, {input_halo, output_halo, output_halo, {1, 0, 0, 0, 0, 0}}}, {462336}, "inbound: Case Top halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 1, 0, 0, 0, 0}}}, {462336}, "inbound: Case Btm halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {1, 1, 0, 0, 0, 0}}},
             {473088},
             "inbound: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 2, 0, 0}}},
             {494592},
             "inbound: Case Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 2, 3, 0, 0}}},
             {559104},
             "inbound: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {30, 10, 0, 0, 0, 0}}},
             {881664},
             "inbound: Case Top+Btm halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 10, 10, 0, 0}}},
             {881664},
             "inbound: Case Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {2, 2, 2, 2, 0, 0}}},
             {580608},
             "inbound: Case Top+Btm+Left+Right halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {1, 2, 0, 0, 9, 0}}},
             {491778},
             "inbound: Case Top+Btm+Front halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 0, 6, 0}}},
             {456876},
             "inbound: Case Front halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, 0, 12, 0}}},
             {462168},
             "inbound: Case Front halo"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 2, 2, 3, 2}}},
             {542010},
             "inbound: Case Left+Right+Front+Back halo"}};

    verify_memory(tests_output_inbound);
}

// memory tests when sparsity is greater than zero
TEST_F(DPU_OperationValidator_Test, Weight_sparsity_memory_test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            true,                                                            // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        float sparse_weight;
    };

    struct TestExpectation {
        long long mem_size_exp;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
    };

    using TestsVector = std::vector<TestCase>;

    // lambda function to test memory when weight_spar
    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << "Test case:"
                      << " " << i << "\n";

            VPUNN::DPUWorkload wl_sparse{t.t_in.wl};
            wl_sparse.weight_sparsity = t.t_in.sparse_weight;

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_sparse)};

                EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                EXPECT_EQ(mem.input_1, t.t_exp.mem_size_exp) << mem << std::endl << t.t_exp.mem_size_exp << std::endl;
            }
            i++;
        }
    };

    const TestsVector tests = {{{wl_ref, 0.00F}, {2670592}},
                               {{wl_ref, 0.01F}, {2654208}},
                               {{wl_ref, 0.05F}, {2555904}},
                               {{wl_ref, 0.30F}, {1966080}},
                               {{wl_ref, 0.44F}, {1638400}},
                               {{wl_ref, 0.56F}, {1343488}},
                               {{wl_ref, 0.23F}, {2129920}},
                               {{wl_ref, 0.65F}, {1130496}},
                               {{wl_ref, 0.77F}, {851968}},
                               {{wl_ref, 0.82F}, {737280}},
                               {{wl_ref, 0.93F}, {475136}},
                               {{wl_ref, 0.99F}, {327680}}

    };
    verify_memory(tests);
}

TEST_F(DPU_OperationValidator_Test, Check_input1_and_input0_memory_when_datatype_is_diff) {
    const VPUNN::DPUWorkload wl_wt_dtype_INT4{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
            {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
            VPUNN::DataType::INT4,                                           // input1 data type
    };

    MemorySize memory;
    {
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_wt_dtype_INT4)) << wl_wt_dtype_INT4 << std::endl;
        EXPECT_LT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0,
                  align(2097152, wl_wt_dtype_INT4.device)); /* =64*64*512*1*1(last 1 is the number of bytes for dtype */
        EXPECT_EQ(mem.input_1, align(1187840, wl_wt_dtype_INT4.device)); /* input1 shape: 1x1x4608x512 */
    }

    {
        DPUWorkload wl_wt_dtype_INT8{std::move(wl_wt_dtype_INT4)};
        wl_wt_dtype_INT8.weight_type = VPUNN::DataType::INT8;
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_wt_dtype_INT8)) << wl_wt_dtype_INT8 << std::endl;
        EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0,
                  align(2097152, wl_wt_dtype_INT8.device)); /* =64*64*512*1*1(last 1 is the number of bytes for dtype */
        EXPECT_EQ(mem.input_1, align(2367488, wl_wt_dtype_INT8.device)); /* input1 shape: 1x1x4608x512 */
    }
}

TEST_F(DPU_OperationValidator_Test, Check_Memory_size_32Bit_output_NPU40) {
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

TEST_F(DPU_OperationValidator_Test, Check_halo_inputs_test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 64, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(21, 21, 512, 1, VPUNN::DataType::UINT8)},      // output dimensions
            {3, 3},                                                          // kernels
            {3, 3},                                                          // strides
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
            true,                                                            // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    struct TestInput {
        VPUNN::DPUWorkload wl;
        HaloWorkload halo_ref;
        std::array<unsigned int, 4> padding;  // TBLR padding for kernel
    };

    struct TestExpectation {
        bool checker_clean_status;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;

    const HaloWorkload::HaloInfoHWC input_halo{0, 0, 0, 0, 0, 0};
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};

    auto verify_checker = [](const TestsVector& tests) {
        int i = 1;  // index of test cases
        std::string info;

        for (const auto& t : tests) {
            std::cout << "TEST " << i << ": " << t.test_case << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            wl_ref_halo.padding = t.t_in.padding;
            const DPUOperation dpu{wl_ref_halo};

            EXPECT_EQ(VPUNN::DPU_ConfigurableOperationValidator<OperationsBehaviour>::check_halo(dpu, info),
                      t.t_exp.checker_clean_status);
            i++;
        }
    };

    const TestsVector tests = {{{wl_ref, {{3, 0, 0, 0, 0, 0}, output_halo, output_halo, output_halo}, {2U, 0U, 0U, 0U}},
                                {false},
                                "Padding top >0 and input halo top >0"},
                               {{wl_ref, {{0, 1, 0, 0, 0, 0}, output_halo, output_halo, output_halo}, {0U, 1U, 0U, 0U}},
                                {false},
                                "Padding bottom >0 and input halo bottom >0"},
                               {{wl_ref, {{0, 0, 5, 0, 0, 0}, output_halo, output_halo, output_halo}, {0U, 0U, 2U, 0U}},
                                {false},
                                "Padding left >0 and input halo left >0"},
                               {{wl_ref, {{0, 0, 0, 3, 0, 0}, output_halo, output_halo, output_halo}, {0U, 0U, 0U, 4U}},
                                {false},
                                "Padding right >0 and input halo right >0"},
                               {{wl_ref, {{1, 0, 0, 1, 0, 0}, output_halo, output_halo, output_halo}, {0U, 1U, 1U, 0U}},
                                {true},
                                "Padding top>0 and right>0, but input halo top=0 and right=0"},
                               {{wl_ref, {{1, 2, 3, 4, 0, 0}, output_halo, output_halo, output_halo}, {1U, 1U, 1U, 1U}},
                                {false},
                                "All paddings >0 and all input halo >0"},
                               {{wl_ref, {{0, 0, 0, 0, 0, 0}, output_halo, output_halo, output_halo}, {0U, 0U, 0U, 0U}},
                                {true},
                                "No padding and no halo"},
                               {{wl_ref, {input_halo, {0, -2, 0, 0}, output_halo, output_halo}, {0U, 0U, 0U, 0U}},
                                {false},
                                "Negative btm halo output"},
                               {{wl_ref, {input_halo, output_halo, {0, 0, -9, 0}, output_halo}, {0U, 0U, 0U, 0U}},
                                {false},
                                "Negative left halo broadcast "},
                               {{wl_ref, {input_halo, output_halo, {-4, 0, 0, 0}, output_halo}, {0U, 0U, 0U, 0U}},
                                {false},
                                "Negative top halo broadcast"},
                               {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, -1, 0}}, {0U, 0U, 0U, 0U}},
                                {false},
                                "Negative left halo inbound"},
                               {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, -2}}, {0U, 0U, 0U, 0U}},
                                {false},
                                "Negative right halo inbound"},
                               {{wl_ref, {input_halo, output_halo, output_halo, {-3, 0, 0, 0}}, {0U, 0U, 0U, 0U}},
                                {false},
                                "Negative top halo inbound"},
                               {{wl_ref, {input_halo, {0, 0, 0, -7}, output_halo, output_halo}, {0U, 3U, 0U, 0U}},
                                {false},
                                "Negative right halo output and btm padding"}};

    verify_checker(tests);
}

// here we split an initial wl into 2 tensors to see if memory is calculated correctly when using halo information
TEST_F(DPU_OperationValidator_Test, Convolution_3x3_HALOTest) {
    const VPUDevice device_req{VPUDevice::VPU_2_7};
    const VPUNN::DPUWorkload wl_ref_3x3{
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 14, 512, 1, VPUNN::DataType::UINT8)},      // input dimensions
            {VPUNN::VPUTensor(7, 7, 512, 1, VPUNN::DataType::UINT8)},        // output dimensions
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {1, 0, 1, 0},                                                    // padding
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

    // divided the initial 3x3 wl in 2

    // top
    const VPUNN::DPUWorkload top_wl_ref_3x3{
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 6, 512, 1, VPUNN::DataType::UINT8)},       // input dimensions WHCB
            {VPUNN::VPUTensor(7, 3, 512, 1, VPUNN::DataType::UINT8)},        // output dimensions WHCB
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

    // bottom
    const VPUNN::DPUWorkload btm_wl_ref_3x3{
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(14, 8, 512, 1, VPUNN::DataType::UINT8)},       // input dimensions WHCB
            {VPUNN::VPUTensor(7, 4, 512, 1, VPUNN::DataType::UINT8)},        // output dimensions WHCB
            {3, 3},                                                          // kernels
            {2, 2},                                                          // strides
            {0, 0, 1, 0},                                                    // padding TBLR
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

    // align memory

    // initial wl
    const auto input_0_raw{14 * 14 * 512};
    const auto output_0_raw{7 * 7 * 512};

    const auto input_0{align(input_0_raw, device_req)};
    const auto output_0{align(output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(input_0, device_req));
    EXPECT_TRUE(isAligned(output_0, device_req));

    // top wl
    const auto top_input_0_raw{14 * 6 * 512};
    const auto top_output_0_raw{7 * 3 * 512};

    const auto top_input_0{align(top_input_0_raw, device_req)};
    const auto top_output_0{align(top_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(top_input_0, device_req));
    EXPECT_TRUE(isAligned(top_output_0, device_req));

    // bottom wl
    const auto btm_input_0_raw{14 * 8 * 512};
    const auto btm_output_0_raw{7 * 4 * 512};

    const auto btm_input_0{align(btm_input_0_raw, device_req)};
    const auto btm_output_0{align(btm_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(btm_input_0, device_req));
    EXPECT_TRUE(isAligned(btm_output_0, device_req));

    ASSERT_TRUE(dut.is_supported(wl_ref_3x3.device));
    ASSERT_TRUE(dut.is_supported(top_wl_ref_3x3.device));
    ASSERT_TRUE(dut.is_supported(btm_wl_ref_3x3.device));

    {  // verify compute memory for initial wl, no halo information
        MemorySize mem_clean;
        {
            MemorySize& mem{mem_clean};
            auto wl{std::move(wl_ref_3x3)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, input_0) << mem << std::endl;
            EXPECT_EQ(mem.output_0, output_0) << mem << std::endl;
        }

        // verify memory for top tensor, no halo information
        MemorySize top_mem_halo;
        {
            MemorySize& mem{top_mem_halo};
            auto wl{std::move(top_wl_ref_3x3)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, top_input_0) << mem << std::endl;
            EXPECT_EQ(mem.output_0, top_output_0) << mem << std::endl;
        }

        // compare values of the initial wl with those of top tensor to ensure that the initial wl memory values are not
        // smaller than those of top tensor
        EXPECT_GT(mem_clean.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_GT(mem_clean.output_0, top_mem_halo.output_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << std::endl;

        // verify memory for btm tensor, halo information
        MemorySize btm_mem_halo;
        {
            MemorySize& mem{btm_mem_halo};
            auto wl{std::move(btm_wl_ref_3x3)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_GT(mem.input_0, btm_input_0 - (14 * 512)) << mem << std::endl;  // 1 row less
            EXPECT_EQ(mem.output_0, btm_output_0) << mem << std::endl;
        }

        EXPECT_GT(btm_mem_halo.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.output_0, top_mem_halo.output_0)
                << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
    }
}

TEST_F(DPU_OperationValidator_Test, Convolution_5x5_HALOTest) {
    const VPUDevice device_req {VPUDevice::VPU_2_7};
    const VPUNN::DPUWorkload wl_ref_5x5 = {
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},   // input dimensions
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},   // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {2, 2, 2, 2},                                                    // padding TBLR
            VPUNN::ExecutionMode::CUBOID_4x16,                               // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            0,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload top_wl_ref_5x5 = {
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(128, 64, 16, 1, VPUNN::DataType::FLOAT16)},    // input dimensions WHCB
            {VPUNN::VPUTensor(128, 64, 16, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {2, 0, 2, 2},                                                    // padding TBLR
            VPUNN::ExecutionMode::CUBOID_4x16,                               // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            0,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{0, 2, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload btm_wl_ref_5x5 = {
            device_req,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(128, 64, 16, 1, VPUNN::DataType::FLOAT16)},    // input dimensions WHCB
            {VPUNN::VPUTensor(128, 64, 16, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 2, 2, 2},                                                    // padding TBLR
            VPUNN::ExecutionMode::CUBOID_4x16,                               // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.0F,                                                            // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            0,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                  // isi_strategy
            false,                                                           // weight_sparsity_enabled
            {{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    // initial wl
    const auto input_0_raw{128 * 128 * 16};
    const auto output_0_raw{128 * 128 * 16};

    const auto input_0{align(input_0_raw, device_req)};
    const auto output_0{align(output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(input_0, device_req));
    EXPECT_TRUE(isAligned(output_0, device_req));

    // top wl
    const auto top_input_0_raw{64 * 128 * 16};
    const auto top_output_0_raw{64 * 128 * 16};

    const auto top_input_0{align(top_input_0_raw, device_req)};
    const auto top_output_0{align(top_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(top_input_0, device_req));
    EXPECT_TRUE(isAligned(top_output_0, device_req));

    // bottom wl
    const auto btm_input_0_raw{64 * 128 * 16};
    const auto btm_output_0_raw{64 * 128 * 16};

    const auto btm_input_0{align(btm_input_0_raw, device_req)};
    const auto btm_output_0{align(btm_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(btm_input_0, device_req));
    EXPECT_TRUE(isAligned(btm_output_0, device_req));

    ASSERT_TRUE(dut.is_supported(wl_ref_5x5.device));
    ASSERT_TRUE(dut.is_supported(top_wl_ref_5x5.device));
    ASSERT_TRUE(dut.is_supported(btm_wl_ref_5x5.device));

    {
        MemorySize mem_clean;
        {
            MemorySize& mem{mem_clean};
            auto wl{std::move(wl_ref_5x5)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, input_0 * 2) << mem << std::endl;  // *2 because data type FLOAT16
            EXPECT_EQ(mem.output_0, output_0 * 2) << mem << std::endl;
        }

        MemorySize top_mem_halo;
        {
            MemorySize& mem{top_mem_halo};
            auto wl{std::move(top_wl_ref_5x5)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_GT(mem.input_0, top_input_0 - (2 * (128 * 16))) << mem << std::endl;
            EXPECT_EQ(mem.output_0, top_output_0 * 2) << mem << std::endl;
        }

        EXPECT_GT(mem_clean.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_GT(mem_clean.output_0, top_mem_halo.output_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << std::endl;

        MemorySize btm_mem_halo;
        {
            MemorySize& mem{btm_mem_halo};
            auto wl{std::move(btm_wl_ref_5x5)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_GT(mem.input_0, btm_input_0 - (2 * (128 * 16))) << mem << std::endl;
            EXPECT_EQ(mem.output_0, btm_output_0 * 2) << mem << std::endl;
        }

        EXPECT_EQ(btm_mem_halo.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.output_0, top_mem_halo.output_0)
                << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
    }
}
TEST_F(DPU_OperationValidator_Test, Maxpool_HALOTest) {
    const VPUDevice device_req{VPUDevice::VPU_2_7};
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,                                       ///
            {VPUNN::VPUTensor(7, 7, 32, 1, VPUNN::DataType::UINT8)},         // input dimensions
            {VPUNN::VPUTensor(6, 6, 32, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {2, 2},                                                          // kernels
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload top_wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,                                       ///
            {VPUNN::VPUTensor(7, 3, 32, 1, VPUNN::DataType::UINT8)},         // input dimensions WHCB
            {VPUNN::VPUTensor(6, 2, 32, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {2, 2},                                                          // kernels
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    const VPUNN::DPUWorkload btm_wl_ref{
            device_req,
            VPUNN::Operation::MAXPOOL,                                       ///
            {VPUNN::VPUTensor(7, 4, 32, 1, VPUNN::DataType::UINT8)},         // input dimensions
            {VPUNN::VPUTensor(6, 4, 32, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {2, 2},                                                          // kernels
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
            false,                                                           // weight_sparsity_enabled
            {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
    };

    // initial wl
    const auto input_0_raw{7 * 7 * 32};
    const auto output_0_raw{6 * 6 * 32};

    const auto input_0{align(input_0_raw, device_req)};
    const auto output_0{align(output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(input_0, device_req));
    EXPECT_TRUE(isAligned(output_0, device_req));

    // top wl
    const auto top_input_0_raw{7 * 3 * 32};
    const auto top_output_0_raw{6 * 2 * 32};

    const auto top_input_0{align(top_input_0_raw, device_req)};
    const auto top_output_0{align(top_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(top_input_0, device_req));
    EXPECT_TRUE(isAligned(top_output_0, device_req));

    // bottom wl
    const auto btm_input_0_raw{7 * 4 * 32};
    const auto btm_output_0_raw{6 * 4 * 32};

    const auto btm_input_0{align(btm_input_0_raw, device_req)};
    const auto btm_output_0{align(btm_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(btm_input_0, device_req));
    EXPECT_TRUE(isAligned(btm_output_0, device_req));

    ASSERT_TRUE(dut.is_supported(wl_ref.device));
    ASSERT_TRUE(dut.is_supported(top_wl_ref.device));
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

        EXPECT_EQ(mem_clean.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.output_0, top_mem_halo.output_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << std::endl;

        MemorySize btm_mem_halo;
        {
            MemorySize& mem{btm_mem_halo};
            auto wl{std::move(btm_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_GT(mem.input_0, btm_input_0 - (7 * 32)) << mem << std::endl;
            EXPECT_EQ(mem.output_0, btm_output_0) << mem << std::endl;
        }

        EXPECT_EQ(btm_mem_halo.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.output_0, top_mem_halo.output_0)
                << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << btm_mem_halo << std::endl;
    }
}
TEST_F(DPU_OperationValidator_Test, Maxpool_HALOTest2) {
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
            device_req,//VPUNN::VPUDevice::VPU_2_7,
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

TEST_F(DPU_OperationValidator_Test, DW_Convolution_HALOTest) {
    const VPUDevice device_req{VPUDevice::VPU_2_7};
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 17, 288, 1, VPUNN::DataType::FLOAT16)},    // input dimensions
            {VPUNN::VPUTensor(17, 17, 288, 1, VPUNN::DataType::FLOAT16)},    // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {2, 2, 2, 2},                                                    // padding
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
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };
    const VPUNN::DPUWorkload top_wl_ref = {
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 7, 288, 1, VPUNN::DataType::FLOAT16)},     // input dimensions
            {VPUNN::VPUTensor(17, 7, 288, 1, VPUNN::DataType::FLOAT16)},     // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {2, 0, 2, 2},                                                    // padding
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
            {{0, 2, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };
    const VPUNN::DPUWorkload middle_wl_ref = {
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 4, 288, 1, VPUNN::DataType::FLOAT16)},     // input dimensions
            {VPUNN::VPUTensor(17, 4, 288, 1, VPUNN::DataType::FLOAT16)},     // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 2, 2},                                                    // padding
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
            {{2, 2, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };
    const VPUNN::DPUWorkload btm_wl_ref = {
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(17, 6, 288, 1, VPUNN::DataType::FLOAT16)},     // input dimensions
            {VPUNN::VPUTensor(17, 6, 288, 1, VPUNN::DataType::FLOAT16)},     // output dimensions
            {5, 5},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 2, 2, 2},                                                    // padding
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
            {{2, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };

    // initial wl
    const auto input_0_raw{17 * 17 * 288};
    const auto output_0_raw{17 * 17 * 288};

    const auto input_0{align(input_0_raw, device_req)};
    const auto output_0{align(output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(input_0, device_req));
    EXPECT_TRUE(isAligned(output_0, device_req));

    // top wl
    const auto top_input_0_raw{17 * 7 * 288};
    const auto top_output_0_raw{17 * 7 * 288};

    const auto top_input_0{align(top_input_0_raw, device_req)};
    const auto top_output_0{align(top_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(top_input_0, device_req));
    EXPECT_TRUE(isAligned(top_output_0, device_req));

    // middle wl
    const auto mid_input_0_raw{17 * 4 * 288};
    const auto mid_output_0_raw{17 * 4 * 288};

    const auto mid_input_0{align(mid_input_0_raw, device_req)};
    const auto mid_output_0{align(mid_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(mid_input_0, device_req));
    EXPECT_TRUE(isAligned(mid_output_0, device_req));

    // bottom wl
    const auto btm_input_0_raw{17 * 6 * 288};
    const auto btm_output_0_raw{17 * 6 * 288};

    const auto btm_input_0{align(btm_input_0_raw, device_req)};
    const auto btm_output_0{align(btm_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(btm_input_0, device_req));
    EXPECT_TRUE(isAligned(btm_output_0, device_req));

    ASSERT_TRUE(dut.is_supported(wl_ref.device));
    ASSERT_TRUE(dut.is_supported(top_wl_ref.device));
    ASSERT_TRUE(dut.is_supported(middle_wl_ref.device));
    ASSERT_TRUE(dut.is_supported(btm_wl_ref.device));

    {
        MemorySize mem_clean;
        {
            MemorySize& mem{mem_clean};
            auto wl{std::move(wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
            // We multiply by two because the data type is FLOAT16
            EXPECT_LT(mem.input_0, input_0 * 2)
                    << mem << std::endl;  // mem.input_0<input_0 * 2 because the memory alignment is done as follows
                                          // mem.input_0_raw*2-alignemet and (input_0_raw-alignement)*2
            EXPECT_LT(mem.output_0, output_0 * 2) << mem << std::endl;
        }

        MemorySize top_mem_halo;
        {
            MemorySize& mem{top_mem_halo};
            auto wl{std::move(top_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_LT(mem.input_0, top_input_0 * 2 - 2 * (2 * 17 * 288))
                    << mem << std::endl;  // halo input btm is 2, so we  subtract 2 rows from the input tensor
            EXPECT_LT(mem.output_0, top_output_0 * 2) << mem << std::endl;
        }

        EXPECT_GT(mem_clean.input_0, top_mem_halo.input_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_GT(mem_clean.output_0, top_mem_halo.output_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << std::endl;

        MemorySize mid_mem_halo;
        {
            MemorySize& mem{mid_mem_halo};
            auto wl{std::move(middle_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            // case when mem.input_0 is 0 because halo input top si 2 and btm is 2, height of input tensor is 4
            EXPECT_LT(mem.input_0, mid_input_0 * 2 - 2 * (4 * 17 * 288))
                    << mem << std::endl;  // halo input top is 2 and btm is 2, so
                                          //  we subtract 4 rows from the input tensor
            EXPECT_LT(mem.output_0, mid_output_0 * 2) << mem << std::endl;
        }

        EXPECT_LT(mid_mem_halo.input_0, top_mem_halo.input_0) << mid_mem_halo << top_mem_halo << std::endl;
        EXPECT_LT(mid_mem_halo.output_0, top_mem_halo.output_0) << mid_mem_halo << top_mem_halo << std::endl;
        EXPECT_EQ(mid_mem_halo.input_1, top_mem_halo.input_1) << mid_mem_halo << top_mem_halo << std::endl;

        MemorySize btm_mem_halo;
        {
            MemorySize& mem{btm_mem_halo};
            auto wl{std::move(btm_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_GT(mem.input_0, btm_input_0 * 2 - 2 * (2 * 17 * 288)) << mem << std::endl;
            EXPECT_EQ(mem.output_0, btm_output_0 * 2) << mem << std::endl;
        }

        EXPECT_GT(btm_mem_halo.input_0, mid_mem_halo.input_0)
                << mem_clean << top_mem_halo << mid_mem_halo << btm_mem_halo << std::endl;
        EXPECT_GT(btm_mem_halo.output_0, mid_mem_halo.output_0)
                << mem_clean << top_mem_halo << mid_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.input_1, mid_mem_halo.input_1)
                << mem_clean << top_mem_halo << mid_mem_halo << btm_mem_halo << std::endl;
    }
}

TEST_F(DPU_OperationValidator_Test, DW_Convolution_HALOTest2) {
    const VPUDevice device_req{VPUDevice::VPU_2_7};
    const VPUNN::DPUWorkload wl_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::UINT8)},       // input dimensions
            {VPUNN::VPUTensor(4, 4, 64, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {8, 8},                                                          // kernels
            {8, 8},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };

    // top
    const VPUNN::DPUWorkload top_wl_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 10, 64, 1, VPUNN::DataType::UINT8)},       // input dimensions WHCB
            {VPUNN::VPUTensor(4, 1, 64, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {8, 8},                                                          // kernels
            {8, 8},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };

    //  middle
    const VPUNN::DPUWorkload middle_wl_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 10, 64, 1, VPUNN::DataType::UINT8)},       // input dimensions
            {VPUNN::VPUTensor(4, 2, 64, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {8, 8},                                                          // kernels
            {8, 8},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{2, 4, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };

    const VPUNN::DPUWorkload btm_wl_ref{
            device_req,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(32, 12, 64, 1, VPUNN::DataType::UINT8)},       // input dimensions
            {VPUNN::VPUTensor(4, 1, 64, 1, VPUNN::DataType::UINT8)},         // output dimensions
            {8, 8},                                                          // kernels
            {8, 8},                                                          // strides
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
            false,                                                           // weight_sparsity_enabled
            {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects TBLR
    };

    // initial wl
    const auto input_0_raw{32 * 32 * 64};
    const auto output_0_raw{4 * 4 * 64};

    const auto input_0{align(input_0_raw, device_req)};
    const auto output_0{align(output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(input_0, device_req));
    EXPECT_TRUE(isAligned(output_0, device_req));

    // top wl
    const auto top_input_0_raw{32 * 10 * 64};
    const auto top_output_0_raw{4 * 1 * 64};

    const auto top_input_0{align(top_input_0_raw, device_req)};
    const auto top_output_0{align(top_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(top_input_0, device_req));
    EXPECT_TRUE(isAligned(top_output_0, device_req));

    // middle wl
    const auto mid_input_0_raw{32 * 10 * 64};
    const auto mid_output_0_raw{4 * 2 * 64};

    const auto mid_input_0{align(mid_input_0_raw, device_req)};
    const auto mid_output_0{align(mid_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(mid_input_0, device_req));
    EXPECT_TRUE(isAligned(mid_output_0, device_req));

    // bottom wl
    const auto btm_input_0_raw{32 * 12 * 64};
    const auto btm_output_0_raw{4 * 1 * 64};

    const auto btm_input_0{align(btm_input_0_raw, device_req)};
    const auto btm_output_0{align(btm_output_0_raw, device_req)};

    EXPECT_TRUE(isAligned(btm_input_0, device_req));
    EXPECT_TRUE(isAligned(btm_output_0, device_req));

    ASSERT_TRUE(dut.is_supported(wl_ref.device));
    ASSERT_TRUE(dut.is_supported(top_wl_ref.device));
    ASSERT_TRUE(dut.is_supported(middle_wl_ref.device));
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
        EXPECT_EQ(mem_clean.output_0, top_mem_halo.output_0) << mem_clean << top_mem_halo << std::endl;
        EXPECT_EQ(mem_clean.input_1, top_mem_halo.input_1) << mem_clean << top_mem_halo << std::endl;

        MemorySize mid_mem_halo;
        {
            MemorySize& mem{mid_mem_halo};
            auto wl{std::move(middle_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_LT(mem.input_0, mid_input_0 - (6 * 32 * 64))
                    << mem
                    << std::endl;  // halo input top is 2 and btm is 4, so we subtract 2+4=6 rows from the input tensor
            EXPECT_EQ(mem.output_0, mid_output_0) << mem << std::endl;
        }

        EXPECT_LT(mid_mem_halo.input_0, top_mem_halo.input_0) << mid_mem_halo << top_mem_halo << std::endl;
        EXPECT_EQ(mid_mem_halo.output_0, top_mem_halo.output_0) << mid_mem_halo << top_mem_halo << std::endl;
        EXPECT_EQ(mid_mem_halo.input_1, top_mem_halo.input_1) << mid_mem_halo << top_mem_halo << std::endl;

        MemorySize btm_mem_halo;
        {
            MemorySize& mem{btm_mem_halo};
            auto wl{std::move(btm_wl_ref)};
            EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

            EXPECT_EQ(mem.input_0, btm_input_0) << mem << std::endl;
            EXPECT_EQ(mem.output_0, btm_output_0) << mem << std::endl;
        }

        EXPECT_GT(btm_mem_halo.input_0, mid_mem_halo.input_0)
                << mem_clean << top_mem_halo << mid_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.output_0, mid_mem_halo.output_0)
                << mem_clean << top_mem_halo << mid_mem_halo << btm_mem_halo << std::endl;
        EXPECT_EQ(btm_mem_halo.input_1, mid_mem_halo.input_1)
                << mem_clean << top_mem_halo << mid_mem_halo << btm_mem_halo << std::endl;
    }
}

TEST_F(DPU_OperationValidator_Test, VPU40_presence_Test) {
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
