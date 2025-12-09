// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "dpu_validator.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DPU_OperationValidator_TestNPU5x : public DPU_OperationValidator_Test {};

TEST_F(DPU_OperationValidator_TestNPU5x, Compute_memory_subByte_dtypes) {
    class Builder {
    public:
        static DPUWorkload makeWL_with_new_DataTypes(DataType Tin, DataType Tout) {
            return DPUWorkload{
                    VPUDevice::NPU_5_0,
                    Operation::CONVOLUTION,
                    {VPUTensor(15, 49, 256, 1, Tin)},   // input dimensions
                    {VPUTensor(15, 49, 256, 1, Tout)},  // output dimensions
                    {1, 1},                             // kernels
                    {1, 1},                             // strides
                    {0, 0, 0, 0},                       // padding
                    ExecutionMode::CUBOID_16x16,        // execution mode
                    ActivationFunction::NONE,           // activation
                    0.F,                                // act sparsity
                    0.F,                                // weight_sparsity
                    {swz_def, swz_def},                 // input_swizzling
                    {swz_def},                          // output_swizzling
                    1,                                  // owtiles
                    {0, 0, 0, 0},                       // offsets,
                    ISIStrategy::CLUSTERING,            // isi_strategy
                    false,                              // weight_sparsity_enabled
            };
        }
    };
    const DPUWorkload wl_int4{Builder::makeWL_with_new_DataTypes(DataType::INT4, DataType::INT4)};
    const DPUWorkload wl_int2{Builder::makeWL_with_new_DataTypes(DataType::INT2, DataType::INT2)};
    const DPUWorkload wl_int1{Builder::makeWL_with_new_DataTypes(DataType::INT1, DataType::INT1)};
    const DPUWorkload wl_float16_hf8{Builder::makeWL_with_new_DataTypes(DataType::FLOAT16, DataType::HF8)};
    const DPUWorkload wl_bfloat16_uint8{Builder::makeWL_with_new_DataTypes(DataType::BFLOAT16, DataType::UINT8)};

    std::vector<DPUWorkload> workloads = {wl_int4, wl_int2, wl_int1, wl_float16_hf8, wl_bfloat16_uint8};

    struct TestInput {
        DPUWorkload wl;  // the wl for which we compute memory
    };

    struct TestExpectation {
        long long in0_mem_size_exp;   // input0 unaligned memory expected
        long long out0_mem_size_exp;  // output0 unaligned memory expected
        bool expect_throw = false;    // we expect to throw exception for sub-byte dtypes when computing memory for
                                      // input_0 and output_0 tensors
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string info = "";
    };
    using TestsVector = std::vector<TestCase>;

    auto verify_memory = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        for (const auto& t : tests) {
            std::cout << "Test case:"
                      << " " << i << " , " << t.info << "\n";

            DPUWorkload wl_ref{t.t_in.wl};

            MemorySize memory;
            {
                MemorySize& mem{memory};
                auto wl{std::move(wl_ref)};

                if (t.t_exp.expect_throw) {
                    EXPECT_THROW(mem = dut.compute_wl_memory(wl), std::runtime_error) << wl << std::endl;
                } else {
                    EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
                    EXPECT_EQ(mem.input_0, align(t.t_exp.in0_mem_size_exp, wl.device))
                            << mem << std::endl
                            << align(t.t_exp.in0_mem_size_exp, wl.device) << std::endl;
                    EXPECT_EQ(mem.output_0, align(t.t_exp.out0_mem_size_exp, wl.device))
                            << mem << std::endl
                            << align(t.t_exp.out0_mem_size_exp, wl.device) << std::endl;
                }
            }
            i++;
        }
    };

    // old test cases, now for datatypes<8 bits we expect to throw exception when we try to compute input_0 and output_0
    // memory const TestsVector tests = {
    //        // clang-format off
    //        {{std::move(wl_int4)}, {94080 /* (15*49*256*4(=bits))/8 */, 94080 /* (15*49*256*4(=bits))/8 */}, "in/out
    //        dtype INT4"},
    //        {{std::move(wl_int2)}, {47040 /* (15*49*256*2(=bits))/8 */, 47040 /* (15*49*256*2(=bits))/8 */}, "in/out
    //        dtype INT2"},
    //        {{std::move(wl_int1)}, {23520 /* (15*49*256*1(=bits))/8 */, 23520 /* (15*49*256*1(=bits))/8 */}, "in/out
    //        dtype INT1"},
    //        {{std::move(wl_float16_hf8)}, {376320 /* (15*49*252*16(=bits))/8 */, 188160  /* (15*49*252*8(=bits))/8
    //        */}, "in dtype FLOAT16, out dtype HF8"},
    //        {{std::move(wl_bfloat16_uint8)}, {376320 /* (15*49*252*16(=bits))/8 */, 188160  /* (15*49*252*8(=bits))/8
    //        */}, "in dtype BFLOAT16, out dtype UINT8"},
    //        // clang-format on
    //};

    const TestsVector tests = {
            // clang-format off
            {{std::move(wl_int4)}, {0, 0, true}, "in/out dtype INT4"},
            {{std::move(wl_int2)}, {0, 0, true}, "in/out dtype INT2"},
            {{std::move(wl_int1)}, {0, 0, true}, "in/out dtype INT1"},
            {{std::move(wl_float16_hf8)}, {376320 /* (15*49*252*16(=bits))/8 */, 188160  /* (15*49*252*8(=bits))/8 */}, "in dtype FLOAT16, out dtype HF8"},
            {{std::move(wl_bfloat16_uint8)}, {376320 /* (15*49*252*16(=bits))/8 */, 188160  /* (15*49*252*8(=bits))/8 */}, "in dtype BFLOAT16, out dtype UINT8"},
            // clang-format on
    };
    verify_memory(tests);
}

TEST_F(DPU_OperationValidator_TestNPU5x, Check_Memory_size_32Bit_output_NPU50) {
    const VPUDevice device_req{VPUDevice::NPU_5_0};
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
        EXPECT_EQ(mem.input_0, align(262144, device_req));        /* */
        EXPECT_EQ(mem.input_1, align(293632, device_req)) << mem; /* 1*1*(512-16)*576+((512-16)*16)(<-weight_table) */
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
        EXPECT_EQ(mem.input_1, align(303104, device_req)) << mem;            /* 1*1*512*576+(512*16)(<-weight_table) */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    const DPUWorkload wl_4{
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

        const DPUWorkload wl_5{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 - 16, 1, DataType::INT32)},             // output dimensions
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
        DPUWorkload wl_x{std::move(wl_5)};
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_x)) << wl_x << std::endl;
        // EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(262144, device_req));        /* */
        EXPECT_EQ(mem.input_1, align(293632, device_req)) << mem; /* 1*1*(512-16)*576+((512-16)*16)(<-weight_table) */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512 - 16) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }

    const DPUWorkload wl_6{
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
        DPUWorkload wl_x{std::move(wl_6)};
        MemorySize& mem{memory};
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl_x)) << wl_x << std::endl;
        // EXPECT_GT(mem.input_1, mem.input_0) << mem << std::endl;
        EXPECT_EQ(mem.input_0, align(262144, device_req));                   /* */
        EXPECT_EQ(mem.input_1, align(303104, device_req)) << mem;            /* 1*1*512*576+(512*16)(<-weight_table) */
        EXPECT_EQ(mem.output_0, align(21 * 21 * (512) * 1 * 4, device_req)); /* 21, 21, 512, 1 */
    }
}

TEST_F(DPU_OperationValidator_TestNPU5x, NPU50_presence_Test) {
    const VPUNN::DPUWorkload wl_ref{
            VPUNN::VPUDevice::NPU_5_0,
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
        wl.device = VPUDevice::NPU_5_0;
        EXPECT_TRUE(dut.is_supported(wl.device)) << " device: \t" << (int)wl.device << " : "
                                                 << VPUDevice_ToText.at(static_cast<int>(wl.device)) << " ;\n";
    }
    {  // no ISI strategy
        auto wl{std::move(wl_ref)};
        wl.device = VPUDevice::NPU_RESERVED;
        EXPECT_TRUE(dut.is_supported(wl.device)) << " device: \t" << (int)wl.device << " : "
                                                 << VPUDevice_ToText.at(static_cast<int>(wl.device)) << " ;\n";
    }
}

TEST_F(DPU_OperationValidator_TestNPU5x, Output_tensor_memory_computation_test_for_different_innermost_dim_NPU5) {
    auto mk_wl = [](const unsigned int w, const unsigned int h, const unsigned int c, Layout layout) -> DPUWorkload {
        return VPUNN::DPUWorkload{
                VPUNN::VPUDevice::NPU_5_0,
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
            
            //innermost dim is C and is divisible by 16 
            {{mk_wl(32, 18, 64, Layout::ZXY)},{36864}}, //32*18*64
            {{mk_wl(32, 18, 32, Layout::ZYX)},{18432}}, //32*18*32

            //innermost dim is C and is NOT divisible by 16
            {{mk_wl(32, 18, 63, Layout::ZXY)},{36864}}, //32*18*64 -> 63 aligned to 64
            {{mk_wl(32, 18, 67, Layout::ZYX)},{46080}}, //32*18*80 -> 67 aligned to 80

            //innermost dim is W and is divisible by 16
            {{mk_wl(32, 18, 65, Layout::XYZ)}, {37440 }}, //32*18*65
            {{mk_wl(128, 18, 65, Layout::XZY)},{149760}}, //128*18*65

            //innermost dim is W and is NOT divisible by 16
            {{mk_wl(33, 18, 65, Layout::XZY)}, {56160}}, //48**18*65 -> 33 aligned to 48
            {{mk_wl(113, 18, 65, Layout::XYZ)},{149760}}, //128*18*65 -> 113 aligned to 128

            //innermost dim is H and is divisible by 16
            {{mk_wl(33, 32, 64, Layout::YXZ)},{67584 }}, //33*32*64
            {{mk_wl(32, 64, 65, Layout::YZX)},{133120}}, //32*64*65

            //innermost dim is H and is NOT divisible by 16
            {{mk_wl(32, 18, 64, Layout::YXZ)},{65536}}, //32*32*64 -> 18 aligned to 32
            {{mk_wl(32, 37, 64, Layout::YZX)},{98304}}, //32*48*64 -> 37 aligned to 48

            // clang-format on
    };

    verify_output0_memory(tests);
}

TEST_F(DPU_OperationValidator_TestNPU5x, elementwiseMemorySizeNoInout1ANdNoInplace_Test_NPU5) {
    VPUDevice device{VPUDevice::NPU_5_0};
    const DPUWorkload wl_ref_full{
            device,
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
            device,
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
            device,
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
        long long in_mem_bytes{180 * 4 * 640 * 2};                                       // float 16
        long long out_mem_bytes{180 * 16 /*innermost dim is aligned to 16*/ * 640 * 1};  // int8

        EXPECT_TRUE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_TRUE(isAligned(out_mem_bytes, wl.device));
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
        long long in_mem_bytes{180 * 4 * 640 * 1};                                       // int8
        long long out_mem_bytes{180 * 16 /*innermost dim is aligned to 16*/ * 640 * 1};  // int8

        EXPECT_TRUE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_TRUE(isAligned(out_mem_bytes, wl.device));
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

        EXPECT_TRUE(isAligned(in_mem_bytes, wl.device));
        auto alignedInMemory{align(in_mem_bytes, wl.device)};
        EXPECT_TRUE(isAligned(alignedInMemory, wl.device));

        EXPECT_TRUE(isAligned(out_mem_bytes, wl.device));
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

}  // namespace VPUNN_unit_tests