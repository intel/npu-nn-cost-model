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

class TestCostModelNPU4x : public TestCostModel {
public:
protected:
    const VPUDevice device{VPUDevice::VPU_4_0};
    const VPUDevice device_req{device};

    void basicTest(const DPUWorkload& workload, VPUCostModel& crt_model, std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(!Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }
    void basicNOKTest(const DPUWorkload& workload, VPUCostModel& crt_model, const CyclesInterfaceType errcode_expected,
                      std::string info = "") {
        DPUWorkload wl{workload};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        std::stringstream buffer;
        buffer << "\nDetails : " << info << " Op:" << Operation_ToText.at(static_cast<int>(wl.op))
               << "\n  NN cyc:" << cost_cyc << " : " << Cycles::toErrorText(cost_cyc) << "\n ";
        std::string details = buffer.str();

        EXPECT_TRUE(Cycles::isErrorCode(cost_cyc)) << info << wl << errInfo << details;
        EXPECT_EQ(errcode_expected, cost_cyc) << info << wl << errInfo << details;

        std::cout << details
                  << "-X-------------------------------------------------------------------------------------------"
                     "-\n";
    }

private:
};

TEST_F(TestCostModelNPU4x, Test_Serializer_for_DPU_vec_function) {
    VPUNN::DPUWorkload wl0 = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::DPUWorkload wl1 = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 40, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 40, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::DPUWorkload wl2 = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(112, 12, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(112, 12, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                      // kernels
            {1, 1},                                                      // strides
            {1, 1, 1, 1},                                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16                           // execution mode
    };

    std::vector<VPUNN::DPUWorkload> workloads = {std::move(wl0), std::move(wl1), std::move(wl2)};

    VPUNN::VPUCostModel model_path{VPU_4_0_MODEL_PATH};

    // run all in one vector
    std::vector<VPUNN::CyclesInterfaceType> cycles;
    ASSERT_NO_THROW(cycles = model_path.DPU(workloads));
}

TEST_F(TestCostModelNPU4x, Test_Serializer_for_DPU_function) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::VPUCostModel model40{VPU_4_0_MODEL_PATH};
    EXPECT_TRUE(model40.nn_initialized());

    auto cycles = model40.DPU(std::move(wl));
    EXPECT_FALSE(Cycles::isErrorCode(cycles)) << cycles;
}

TEST_F(TestCostModelNPU4x, Test_Serializer_for_NNCostProvider) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    // uncomment to generate the csv from NNCostProvider
    // set_env_var("ENABLE_VPUNN_CACHE_MISS_DATA_SERIALIZATION", "TRUE");
    VPUNN::VPUCostModel model40{VPU_4_0_MODEL_PATH};
    EXPECT_TRUE(model40.nn_initialized());

    auto cycles = model40.DPU(std::move(wl));
    EXPECT_FALSE(Cycles::isErrorCode(cycles)) << cycles;
}

TEST_F(TestCostModelNPU4x, Mock_Legacy159_40_DPU) {
    std::string mroot{NameHelperNN::get_model_root()};
    std::filesystem::path models_root{mroot};

    const std::filesystem::path stricti89_02Ver{std::filesystem::path(models_root) /= "vpu_40_159_strict.vpunn"};
    const std::filesystem::path i11_02Ver{std::filesystem::path(std::move(models_root)) /= "vpu_40_159.vpunn"};
    {
        VPUNN::VPUCostModel model_strict{stricti89_02Ver.string()};
        EXPECT_TRUE(model_strict.nn_initialized());
    }
    {
        VPUNN::VPUCostModel model_Nostrict{i11_02Ver.string()};
        EXPECT_TRUE(model_Nostrict.nn_initialized());
    }

    {
        VPUNN::VPUCostModel model_strict{stricti89_02Ver.string()};
        EXPECT_TRUE(model_strict.nn_initialized());
        VPUNN::VPUCostModel model_nostrict{i11_02Ver.string()};
        EXPECT_TRUE(model_nostrict.nn_initialized());

        {
            DPUWorkload wl_strict = wl_glob_40;
            DPUWorkload wl_nostrict = wl_glob_40;

            const auto cycles_strict = model_strict.DPU(std::move(wl_strict));
            const auto cycles_Nostrict = model_nostrict.DPU(std::move(wl_nostrict));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_strict));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Nostrict));

            EXPECT_EQ(cycles_strict, cycles_Nostrict);
            EXPECT_GT(cycles_strict, 0);
        }
        {  // swizzling  0 will pass(strict) and not pass to the NN ()
            DPUWorkload wl_strict = wl_glob_40;
            wl_strict.input_swizzling = {Swizzling::KEY_0, Swizzling::KEY_0};
            wl_strict.output_swizzling = {Swizzling::KEY_0};
            DPUWorkload wl_nostrict = wl_strict;

            const auto cycles_strict = model_strict.DPU(std::move(wl_strict));
            const auto cycles_Nostrict = model_nostrict.DPU(std::move(wl_nostrict));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_strict));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Nostrict));

            EXPECT_EQ(cycles_strict, cycles_Nostrict);
            EXPECT_GT(cycles_strict, 0);
        }
        {  // sSOK + owt ;will limit owt to 2 or not (strict)
            DPUWorkload wl_strict = wl_glob_40;
            wl_strict.isi_strategy = ISIStrategy::SPLIT_OVER_K;
            wl_strict.output_write_tiles = 6;
            DPUWorkload wl_nostrict = wl_strict;

            const auto cycles_strict = model_strict.DPU(std::move(wl_strict));
            const auto cycles_Nostrict = model_nostrict.DPU(std::move(wl_nostrict));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_strict));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Nostrict));

            EXPECT_GE(cycles_strict, cycles_Nostrict);
            EXPECT_LE(cycles_strict - 300, cycles_Nostrict);
            EXPECT_GT(cycles_strict, 0);
        }
    }
}

TEST_F(TestCostModelNPU4x, Sanitization_with_stride_and_kernel_0) {
    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(56, 6, 64, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUNN::VPUTensor(56, 6, 64, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // output dimensions
            {0, 0},                                                                 // kernels
            {0, 0},                                                                 // strides
            {0, 0, 0, 0},                                                           // padding
            VPUNN::ExecutionMode::CUBOID_8x16,                                      // execution mode
            VPUNN::ActivationFunction::NONE,                                        // activation
            0.0F,                                                                   // act_sparsity
            0.0F,                                                                   // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                                   // input_swizzling
            {Swizzling::KEY_5},                                                     // output_swizzling
            1,                                                                      // output_write_tiles
            {0, 0, 0, 0},                                                           // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                         // isi_strategy
            true,                                                                   // weight_sparsity_enabled

    };

    const std::string modelFile{VPU_4_0_MODEL_PATH};

    VPUNN::VPUCostModel test_model{modelFile};
    EXPECT_TRUE(test_model.nn_initialized());

    std::string info = "";
    auto cycles = test_model.DPU(std::move(wl), info);

    EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles)) << info;
}

TEST_F(TestCostModelNPU4x, Wl_with_extreme_values) {
    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(0, 0, 0, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUNN::VPUTensor(0, 0, 0, 1, VPUNN::DataType::UINT8, Layout::ZXY)},  // output dimensions
            {0, 0},                                                               // kernels
            {0, 0},                                                               // strides
            {0, 0, 0, 0},                                                         // padding
            VPUNN::ExecutionMode::CUBOID_8x16,                                    // execution mode
            VPUNN::ActivationFunction::NONE,                                      // activation
            0.0F,                                                                 // act_sparsity
            0.0F,                                                                 // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},                                 // input_swizzling
            {Swizzling::KEY_5},                                                   // output_swizzling
            0,                                                                    // output_write_tiles
            {0, 0, 0, 0},                                                         // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                       // isi_strategy
            true,                                                                 // weight_sparsity_enabled

    };

    const std::string modelFile{VPU_4_0_MODEL_PATH};

    VPUNN::VPUCostModel test_model{modelFile};
    EXPECT_TRUE(test_model.nn_initialized());

    std::string info = "";
    auto cycles = test_model.DPU(std::move(wl), info);

    EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles)) << info;
}

TEST_F(TestCostModelNPU4x, Weigths_types_NPU40_test) {
    constexpr int kw{3};
    constexpr int kh{3};
    constexpr int in_ch = 64;
    constexpr int o_ch = 64;

    constexpr VPUDevice npu{VPUDevice::VPU_4_0};

    VPUCostModel& model_x{cost_models.getModel(npu)};

    DPU_OperationValidator dut;
    class Builder {
    public:
        static DPUWorkload makeWL(Operation op, unsigned int in_ch, unsigned int o_ch, DataType Tin, DataType Tout,
                                  unsigned int kw, unsigned int kh, DataType wts_t, unsigned int padd) {
            return DPUWorkload{
                    VPUDevice::VPU_4_0,
                    op,
                    {VPUTensor(28, 28, in_ch, 1, Tin, Layout::ZXY)},  // input dimensions
                    {VPUTensor(28, 28, o_ch, 1, Tout, Layout::ZXY)},  // output dimensions
                    {kw, kh},                                         // kernels
                    {1, 1},                                           // strides
                    {padd, padd, padd, padd},                         // padding
                    ExecutionMode::CUBOID_16x16,                      // execution mode //original wl have CUBOID_16X16
                    ActivationFunction::NONE,                         // activation
                    0.0F,                                             // act_sparsity
                    0.0F,                                             // weight_sparsity
                    {swz_def, swz_def},                               // input_swizzling
                    {swz_def},                                        // output_swizzling
                    1,                                                // output_write_tiles
                    {0, 0, 0, 0},                                     // offsets
                    ISIStrategy::CLUSTERING,                          // isi_strategy
                    false,                                            // weight_sparsity_enabled
                    {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},  // halo aspects
                    {false, {0, 0, 0, 0}, {0, 0, 0, 0}, false},                      // SEP
                    wts_t,
            };
        }
    };
    const DPUWorkload wl_DWCONV_8_4{Builder::makeWL(Operation::DW_CONVOLUTION, in_ch, o_ch, DataType::INT8,
                                                    DataType::INT8, kw, kh, DataType::INT4, 1U)};
    const DPUWorkload wl_DWCONV_16_4{Builder::makeWL(Operation::DW_CONVOLUTION, in_ch, o_ch, DataType::FLOAT16,
                                                     DataType::FLOAT16, kw, kh, DataType::INT4, 1U)};
    const DPUWorkload wl_ELT_8_4{
            Builder::makeWL(Operation::ELTWISE, in_ch, o_ch, DataType::INT8, DataType::INT8, 1, 1, DataType::INT4, 0U)};
    const DPUWorkload wl_ELT_16_4{Builder::makeWL(Operation::ELTWISE, in_ch, o_ch, DataType::FLOAT16, DataType::FLOAT16,
                                                  1, 1, DataType::INT4, 0U)};

    std::vector<DPUWorkload> workloads = {std::move(wl_DWCONV_8_4), std::move(wl_DWCONV_16_4), std::move(wl_ELT_8_4),
                                          std::move(wl_ELT_16_4)};

    auto input1_volume = [](const DPUWorkload& wl) -> int {
        int wl_wts_size{0};
        int wl_wts_kernel{0};
        int wts_table_size = wl.outputs[0].get_shape()[2] /*output channels*/ * 16;  // is it 16 also for NPU4+?

        if (wl.op == Operation::DW_CONVOLUTION) {
            wl_wts_kernel = (wl.kernels[0] * wl.kernels[1]) + 32 - ((wl.kernels[0] * wl.kernels[1]) % 32);
        }

        if (wl.op == Operation::ELTWISE) {
            wl_wts_kernel = wl.inputs[0].get_shape()[0] * wl.inputs[0].get_shape()[1] * wl.inputs[0].get_shape()[2];
        }

        wl_wts_size = compute_size_in_bytes((wl_wts_kernel + wts_table_size), wl.weight_type.value());
        EXPECT_EQ(wl_wts_kernel % 32, 0) << "wl_wts_kernel not 32B aligned";

        return wl_wts_size;
    };

    auto verify = [&model_x, &dut, input1_volume](const DPUWorkload& wl) {
        constexpr int num16KB{16384};

        // text will be something like input:dtype, wts:dtype
        std::string ttl = "input:" + DataType_ToText.at(static_cast<int>(wl.inputs[0].get_dtype())) + ", wts:" +
                          (wl.weight_type.has_value() ? DataType_ToText.at(static_cast<int>(wl.weight_type.value()))
                                                      : DataType_ToText.at(static_cast<int>(wl.inputs[0].get_dtype())));

        int wts_table_size = wl.outputs[0].get_shape()[2] /*output channels*/ * 16;  // is it 16 also for NPU4+?
        int wl_wts_size = input1_volume(wl);
        int wl_wts_full_size = wl_wts_size + wts_table_size;  // must be aligned to 16KB

        int padding{num16KB - (wl_wts_full_size % num16KB)};
        int wl_wts_align_full_size = wl_wts_full_size + padding;

        EXPECT_EQ(wl_wts_align_full_size % num16KB, 0) << "wl_wts_size not 16K aligned";

        std::string msg;
        auto cycles = model_x.DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << " Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    };

    for (const auto& wl : workloads) {
        verify(wl);
    }
}

TEST_F(TestCostModelNPU4x, Weigths_types_CONV_NPU40_test) {
    const HaloWorkload halo{};
    const SEPModeInfo sep_activators{};

    std::optional<DataType> wts_t{};

    constexpr int num16KB{16384};

    constexpr int kw{4};
    constexpr int kh{3};
    constexpr int in_ch = ((16 * 2) * 1024 / 32) / (4);
    constexpr int o_ch = 32 * 5;

    constexpr VPUDevice npu{VPUDevice::VPU_4_0};

    VPUCostModel& model_x{cost_models.getModel(npu)};

    DPU_OperationValidator dut;  //

    const DPUWorkload base_wl8_8{
            npu,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, in_ch, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(27, 28, o_ch, 1, DataType::UINT8, Layout::ZXY)},   // output dimensions
            {kw, kh},                                                     // kernels
            {1, 1},                                                       // strides
            {1, 1, 1, 1},                                                 // padding
            ExecutionMode::CUBOID_16x16,  // execution mode //original wl have CUBOID_16X16
            ActivationFunction::NONE,     // activation
            0.0F,                         // act_sparsity
            0.0F,                         // weight_sparsity
            {swz_def, swz_def},           // input_swizzling
            {swz_def},                    // output_swizzling
            1,                            // output_write_tiles
            {0, 0, 0, 0},                 // offsets
            ISIStrategy::CLUSTERING,      // isi_strategy
            false,                        // weight_sparsity_enabled
            halo,                         // halo aspects
            sep_activators,               // sep
            wts_t,                        // optional empty => UINT8 (as input)
    };

    // 8 bit data
    {
        const std::string ttl{"input:INT8, wts: same :"};
        constexpr int wl_wts_kernel = kw * kh * in_ch;  // 32B alignment?
        constexpr int wts_table_size = o_ch * 16;       // is it 16 also for NPU4+?

        constexpr int wl_wts_size = wl_wts_kernel * o_ch;
        constexpr int wl_wts_full_size = wl_wts_size + wts_table_size;  // must be aligned to 16KB
        constexpr int padding{num16KB - (wl_wts_full_size % num16KB)};
        constexpr int wl_wts_align_full_size = wl_wts_full_size + padding;

        EXPECT_EQ(wl_wts_kernel % 32, 0) << "wl_wts_kernel not 32B aligned";
        EXPECT_EQ(wl_wts_size % num16KB, 0) << "wl_wts_size not 16K aligned";
        EXPECT_EQ(wl_wts_align_full_size % num16KB, 0) << "wl_8x8_wts_size not 16K aligned";

        const DPUWorkload& wl{base_wl8_8};
        std::string msg;
        auto cycles = model_x.DPU(wl, msg);

        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    const DPUWorkload base_wl16_16{
            npu,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(27, 28, o_ch, 1, DataType::UINT8, Layout::ZXY)},     // output dimensions
            {kw, kh},                                                       // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1},                                                   // padding
            ExecutionMode::CUBOID_16x16,  // execution mode //original wl have CUBOID_16X16
            ActivationFunction::NONE,     // activation
            0.0F,                         // act_sparsity
            0.0F,                         // weight_sparsity
            {swz_def, swz_def},           // input_swizzling
            {swz_def},                    // output_swizzling
            1,                            // output_write_tiles
            {0, 0, 0, 0},                 // offsets
            ISIStrategy::CLUSTERING,      // isi_strategy
            false,                        // weight_sparsity_enabled
            halo,                         // halo aspects
            sep_activators,               // sep
            wts_t,                        // optional empty => UINT8 (as input)
    };
    // 16 bit data in out
    {
        const std::string ttl{"input:FP16, wts: same :"};
        constexpr int wl_wts_kernel = kw * kh * in_ch * 2;  // 32B alignment?
        constexpr int wts_table_size = o_ch * 16;           // is it 16 also for NPU4+?

        constexpr int wl_wts_size = wl_wts_kernel * o_ch;
        constexpr int wl_wts_full_size = wl_wts_size + wts_table_size;  // must be aligned to 16KB
        constexpr int padding{num16KB - (wl_wts_full_size % num16KB)};
        constexpr int wl_wts_align_full_size = wl_wts_full_size + padding;

        EXPECT_EQ(wl_wts_kernel % 32, 0) << "wl_wts_kernel not 32B aligned";
        EXPECT_EQ(wl_wts_size % num16KB, 0) << "wl_wts_size not 16K aligned";
        EXPECT_EQ(wl_wts_align_full_size % num16KB, 0) << "wl_wts_size not 16K aligned";

        const DPUWorkload& wl{base_wl16_16};
        std::string msg;
        auto cycles = model_x.DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    // int8 in - INT4 wts
    const DPUWorkload base_wl8_4{
            npu,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, in_ch, 1, DataType::INT8, Layout::ZXY)},  // input dimensions
            {VPUTensor(27, 28, o_ch, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
            {kw, kh},                                                    // kernels
            {1, 1},                                                      // strides
            {1, 1, 1, 1},                                                // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode //original wl have CUBOID_16X16
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
            halo,                                          // halo aspects
            sep_activators,                                // sep
            std::make_optional<DataType>(DataType::INT4),  // dedicated
    };
    {
        const std::string ttl{"input:INT8, wts: INT4 :"};
        constexpr int wl_wts_kernel = (kw * kh * in_ch + 1) / 2;  // 32B alignment?
        constexpr int wts_table_size = o_ch * 16;                 // is it 16 also for NPU4+?

        constexpr int wl_wts_size = wl_wts_kernel * o_ch;
        constexpr int wl_wts_full_size = wl_wts_size + wts_table_size;  // must be aligned to 16KB
        constexpr int padding{num16KB - (wl_wts_full_size % num16KB)};
        constexpr int wl_wts_align_full_size = wl_wts_full_size + padding;

        EXPECT_EQ(wl_wts_kernel % 32, 0) << "wl_wts_kernel not 32B aligned";
        EXPECT_EQ(wl_wts_size % num16KB, 0) << "wl_wts_size not 16K aligned";
        EXPECT_EQ(wl_wts_align_full_size % num16KB, 0) << "wl_wts_size not 16K aligned";

        const DPUWorkload& wl{base_wl8_4};
        std::string msg;
        auto cycles = model_x.DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    // FP16 in - INT4 wts
    const DPUWorkload base_wl16_4{
            npu,
            Operation::CONVOLUTION,
            {VPUTensor(28, 28, in_ch, 1, DataType::FLOAT16, Layout::ZXY)},  // input dimensions
            {VPUTensor(27, 28, o_ch, 1, DataType::UINT8, Layout::ZXY)},     // output dimensions
            {kw, kh},                                                       // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1},                                                   // padding
            ExecutionMode::CUBOID_16x16,                   // execution mode //original wl have CUBOID_16X16
            ActivationFunction::NONE,                      // activation
            0.0F,                                          // act_sparsity
            0.0F,                                          // weight_sparsity
            {swz_def, swz_def},                            // input_swizzling
            {swz_def},                                     // output_swizzling
            1,                                             // output_write_tiles
            {0, 0, 0, 0},                                  // offsets
            ISIStrategy::CLUSTERING,                       // isi_strategy
            false,                                         // weight_sparsity_enabled
            halo,                                          // halo aspects
            sep_activators,                                // sep
            std::make_optional<DataType>(DataType::INT4),  // dedicated
    };
    {
        const std::string ttl{"input:FP16, wts: INT4 :"};
        constexpr int wl_wts_kernel = (kw * kh * in_ch + 1) / 2;  // 32B alignment?
        constexpr int wts_table_size = o_ch * 16;                 // is it 16 also for NPU4+?

        constexpr int wl_wts_size = wl_wts_kernel * o_ch;
        constexpr int wl_wts_full_size = wl_wts_size + wts_table_size;  // must be aligned to 16KB
        constexpr int padding{num16KB - (wl_wts_full_size % num16KB)};
        constexpr int wl_wts_align_full_size = wl_wts_full_size + padding;

        EXPECT_EQ(wl_wts_kernel % 32, 0) << "wl_wts_kernel not 32B aligned";
        EXPECT_EQ(wl_wts_size % num16KB, 0) << "wl_wts_size not 16K aligned";
        EXPECT_EQ(wl_wts_align_full_size % num16KB, 0) << "wl_wts_size not 16K aligned";

        const DPUWorkload& wl{base_wl8_4};
        std::string msg;
        auto cycles = model_x.DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    // check runtime equivalence
    {
        EXPECT_EQ(model_x.DPU(base_wl8_4), model_x.DPU(std::move(base_wl8_8)));
        EXPECT_EQ(model_x.DPU(base_wl16_4), model_x.DPU(std::move(base_wl16_16)));

        DPUWorkload base_wl16_8{base_wl16_4};
        base_wl16_8.weight_type = DataType::INT8;

        EXPECT_EQ(model_x.DPU(std::move(base_wl16_4)), model_x.DPU(std::move(base_wl16_8)));

        DPUWorkload base_wl8_16{base_wl8_4};
        base_wl8_16.weight_type = DataType::FLOAT16;
        EXPECT_EQ(model_x.DPU(std::move(base_wl8_4)), model_x.DPU(std::move(base_wl8_16)));
    }

    // EXPECT_TRUE(false);
}

TEST_F(TestCostModelNPU4x, DISABLED_Compressed_CONV_EquivPostprocessing_test_VPU40) {
    // constructs wl with changed output channels
    auto make_CM_oc = [](const DPUWorkload& wl, int ic, int oc) {
        DPUWorkload wl_{wl};
        {
            std::array<unsigned int, 4> new_shape{wl.outputs[0].get_shape()};
            new_shape[2] = oc;  // set output channels
            VPUTensor out{new_shape, wl.outputs[0]};
            wl_.outputs[0] = out;
        }
        {
            std::array<unsigned int, 4> new_shape{wl.inputs[0].get_shape()};
            new_shape[2] = ic;  // set input channels
            VPUTensor im{new_shape, wl.inputs[0]};
            wl_.inputs[0] = im;
        }
        return wl_;
    };
    const DPUWorkload wl_CM_4 = {
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(16, 16, 4, 1, DataType::INT8)},   // input dimensions
            {VPUTensor(16, 16, 64, 1, DataType::INT8)},  // output dimensions
            {3, 3},                                      // kernels
            {1, 1},                                      // strides
            {1, 1, 1, 1},                                // padding
            ExecutionMode::CUBOID_16x16,                 // execution mode
            ActivationFunction::NONE,                    // activation
            0.0F,                                        // act_sparsity
            0.0F,                                        // weight_sparsity
            {swz_def, swz_def},                          // input_swizzling
            {swz_def},                                   // output_swizzling
            1,                                           // output_write_tiles
            {0, 0, 0, 0},                                // offsets
            ISIStrategy::CLUSTERING,                     // isi_strategy
            false,                                       // weight_sparsity_enabled

    };
    const DPUWorkload wl_CM_4F = {
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(16, 16, 4, 1, DataType::FLOAT16)},   // input dimensions
            {VPUTensor(16, 16, 64, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                         // kernels
            {1, 1},                                         // strides
            {1, 1, 1, 1},                                   // padding
            ExecutionMode::CUBOID_16x16,                    // execution mode
            ActivationFunction::NONE,                       // activation
            0.0F,                                           // act_sparsity
            0.0F,                                           // weight_sparsity
            {swz_def, swz_def},                             // input_swizzling
            {swz_def},                                      // output_swizzling
            1,                                              // output_write_tiles
            {0, 0, 0, 0},                                   // offsets
            ISIStrategy::CLUSTERING,                        // isi_strategy
            false,                                          // weight_sparsity_enabled

    };

    auto make_eqiv = [](const DPUWorkload& wl) {
        DPUWorkload wl_equiv{wl};
        std::array<unsigned int, 4> new_shape{wl.inputs[0].get_shape()};
        new_shape[2] = 16;  // 16 channels
        VPUTensor im_conv{new_shape, wl.inputs[0]};
        wl_equiv.inputs[0] = im_conv;
        return wl_equiv;
    };
    // const DPUWorkload wl_CMequiv{make_eqiv(wl_CM_4)};

    {
        VPUCostModel model_{VPU_4_0_MODEL_PATH};
        const auto [v_in, v_out] = model_.get_NN_cost_provider().getNNVersion();

        /* coverity[pass_by_value] */
        auto test_exec = [&](DPUWorkload wl /*, DPUWorkload wl_equiv*/, float factor, std::string info) {
            auto wl_equiv{make_eqiv(wl)};
            std::string info_raw, info_equiv;

            auto cycles_raw = model_.DPU(wl, info_raw);  // will change
            auto cycles_equiv = model_.DPU(wl_equiv, info_equiv);

            EXPECT_FALSE(Cycles::isErrorCode(cycles_raw))
                    << info << "ERROR code received: " << cycles_raw << " : " << Cycles::toErrorText(cycles_raw)
                    << "\n INFO: " << wl << info_raw << std::endl;
            EXPECT_FALSE(Cycles::isErrorCode(cycles_equiv))
                    << info << "ERROR code received: " << cycles_equiv << " : " << Cycles::toErrorText(cycles_equiv)
                    << "\n INFO: " << wl_equiv << info_equiv << std::endl;
            ;

            // EXPECT_EQ(cycles_raw, cycles_equiv) << info;
            EXPECT_NEAR(cycles_raw, cycles_equiv * factor, 100) << cycles_equiv << " : " << info << wl;
            EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0)) << info;
        };
        // test for combinations of input channels and output channels
        if ((int)NNOutputVersions::OUT_CYCLES_NPU40_DEV == v_out) {  // execute only if we are postprocessing CM conv
            test_exec(wl_CM_4, 1.0f / 3.0f, " NOM<INAL ");

            test_exec(make_CM_oc(wl_CM_4, 4, 16), 1.0f, " 16 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 32), 2.0f / 3.0f, " 32 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 48), 2.0f / 3.0f, " 48 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 64), 1.0f / 3.0f, " 64 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 80), 1.0f / 3.0f, " 80 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 96), 1.0f / 3.0f, " 96 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 112), 1.0f / 3.0f, " 112 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 4, 256), 1.0f / 3.0f, " 256 o channels , 4 in ch: ");

            test_exec(make_CM_oc(wl_CM_4, 3, 16), 1.0f / 1.0f, " 16 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 32), 2.0f / 3.0f, " 32 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 48), 2.0f / 3.0f, " 48 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 64), 1.0f / 3.0f, " 64 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 80), 1.0f / 3.0f, " 80 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 96), 1.0f / 3.0f, " 96 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 112), 1.0f / 3.0f, " 112 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 3, 256), 1.0f / 3.0f, " 256 o channels , 3 in ch: ");

            test_exec(make_CM_oc(wl_CM_4, 2, 16), 1.0f / 1.0f, " 16 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 32), 2.0f / 3.0f, " 32 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 48), 2.0f / 3.0f, " 48 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 64), 1.0f / 3.0f, " 64 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 80), 1.0f / 3.0f, " 80 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 96), 1.0f / 3.0f, " 96 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 112), 1.0f / 3.0f, " 112 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 2, 256), 1.0f / 3.0f, " 256 o channels , 2 in ch: ");

            test_exec(make_CM_oc(wl_CM_4, 1, 16), 1.0f / 1.0f, " 16 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 32), 2.0f / 3.0f, " 32 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 48), 2.0f / 3.0f, " 48 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 64), 1.0f / 3.0f, " 64 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 80), 1.0f / 3.0f, " 80 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 96), 1.0f / 3.0f, " 96 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 112), 1.0f / 3.0f, " 112 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 1, 256), 1.0f / 3.0f, " 256 o channels , 1 in ch: ");

            test_exec(make_CM_oc(wl_CM_4, 5, 16), 1.0f, " 16 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 32), 1.0f, " 32 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 48), 1.0f, " 48 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 64), 1.0f, " 64 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 80), 1.0f, " 80 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 96), 1.0f, " 96 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 112), 1.0f, " 112 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 5, 256), 1.0f, " 256 o channels , 5 in ch: ");

            test_exec(make_CM_oc(wl_CM_4, 15, 16), 1.0f, " 16 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 32), 1.0f, " 32 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 48), 1.0f, " 48 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 64), 1.0f, " 64 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 80), 1.0f, " 80 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 96), 1.0f, " 96 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 112), 1.0f, " 112 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4, 15, 256), 1.0f, " 256 o channels , 15 in ch: ");
        }

        // float
        if ((int)NNOutputVersions::OUT_CYCLES_NPU40_DEV == v_out) {  // execute only if we are postprocessing CM conv
            constexpr float o1 = 0.3f;                               // offset for small channels
            constexpr float o2 = 0.22f;                              // offset for larger channels

            test_exec(wl_CM_4F, 1.0f / 3.0f + 0.22f, " NOM<INAL ");

            test_exec(make_CM_oc(wl_CM_4F, 4, 16), 1.0f, " 16 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 32), 2.00f / 3.0f + o1, " 32 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 48), 2.00f / 3.0f + o1, " 48 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 64), 1.00f / 3.0f + o2, " 64 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 80), 1.00f / 3.0f + o2, " 80 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 96), 1.00f / 3.0f + o2, " 96 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 112), 1.0f / 3.0f + o2, " 112 o channels , 4 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 4, 256), 1.0f / 3.0f + o2, " 256 o channels , 4 in ch: ");

            test_exec(make_CM_oc(wl_CM_4F, 3, 16), 1.0f / 1.0f, " 16 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 32), 2.00f / 3.0f + o1, " 32 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 48), 2.00f / 3.0f + o1, " 48 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 64), 1.00f / 3.0f + o2, " 64 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 80), 1.00f / 3.0f + o2, " 80 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 96), 1.00f / 3.0f + o2, " 96 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 112), 1.0f / 3.0f + o2, " 112 o channels , 3 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 3, 256), 1.0f / 3.0f + o2, " 256 o channels , 3 in ch: ");

            test_exec(make_CM_oc(wl_CM_4F, 2, 16), 1.00f / 1.0f, " 16 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 32), 2.00f / 3.0f + o1, " 32 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 48), 2.00f / 3.0f + o1, " 48 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 64), 1.00f / 3.0f + o2, " 64 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 80), 1.00f / 3.0f + o2, " 80 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 96), 1.00f / 3.0f + o2, " 96 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 112), 1.0f / 3.0f + o2, " 112 o channels , 2 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 2, 256), 1.0f / 3.0f + o2, " 256 o channels , 2 in ch: ");

            test_exec(make_CM_oc(wl_CM_4F, 1, 16), 1.00f / 1.0f, " 16 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 32), 2.00f / 3.0f + o1, " 32 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 48), 2.00f / 3.0f + o1, " 48 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 64), 1.00f / 3.0f + o2, " 64 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 80), 1.00f / 3.0f + o2, " 80 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 96), 1.00f / 3.0f + o2, " 96 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 112), 1.0f / 3.0f + o2, " 112 o channels , 1 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 1, 256), 1.0f / 3.0f + o2, " 256 o channels , 1 in ch: ");

            test_exec(make_CM_oc(wl_CM_4F, 5, 16), 1.00f, " 16 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 32), 1.00f, " 32 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 48), 1.00f, " 48 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 64), 1.00f, " 64 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 80), 1.00f, " 80 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 96), 1.00f, " 96 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 112), 1.0f, " 112 o channels , 5 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 5, 256), 1.0f, " 256 o channels , 5 in ch: ");

            test_exec(make_CM_oc(wl_CM_4F, 15, 16), 1.00f, " 16 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 32), 1.00f, " 32 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 48), 1.00f, " 48 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 64), 1.00f, " 64 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 80), 1.00f, " 80 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 96), 1.00f, " 96 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 112), 1.0f, " 112 o channels , 15 in ch: ");
            test_exec(make_CM_oc(wl_CM_4F, 15, 256), 1.0f, " 256 o channels , 15 in ch: ");
        }
    }
}

TEST_F(TestCostModelNPU4x, FP32_output_basic) {
    VPUCostModel crt_model{VPU_4_0_MODEL_PATH};

    const DPUWorkload wl_ref_less{
            device_req,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 + 32, 1, DataType::FLOAT32)},             // output dimensions
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

    const DPUWorkload wl_ref2{
            device_req,
            Operation::ELTWISE,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 64, 64, 1, DataType::FLOAT32)},                   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
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
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };

    //{
    //    DPUWorkload wl{wl_ref_less};
    //    basicNOKTest(wl, crt_model, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    //}
    //{
    //    DPUWorkload wl{wl_ref2};
    //    basicNOKTest(wl, crt_model, Cycles::ERROR_INVALID_INPUT_CONFIGURATION);
    //}

    {
        DPUWorkload wl{wl_ref_less};
        basicTest(wl, crt_model, " COnv should fit in memory.");
    }
    {
        DPUWorkload wl{wl_ref2};
        basicTest(wl, crt_model, "Elemwise should fit into memory.");
    }

    auto checkSame = [&crt_model](const DPUWorkload& w1, const DPUWorkload& w2, std::string info = "") {
        DPUWorkload wl{w1};

        std::string errInfo;
        unsigned cost_cyc{};
        ASSERT_NO_THROW(cost_cyc = crt_model.DPU(wl, errInfo)) << info << wl;

        DPUWorkload wl_2{w2};

        unsigned cost_cyc2{};
        ASSERT_NO_THROW(cost_cyc2 = crt_model.DPU(wl_2, errInfo)) << info << wl_2;

        EXPECT_EQ(cost_cyc, cost_cyc2) << info << wl << wl_2;
    };

    const DPUWorkload wl_ref_less_16{
            device,
            Operation::CONVOLUTION,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(21, 21, 512 + 32, 1, DataType::FLOAT16)},             // output dimensions
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

    const DPUWorkload wl_ref2_16{
            device,
            Operation::ELTWISE,
            {VPUTensor(64, 64, 64, 1, DataType::UINT8)},                     // input dimensions
            {VPUTensor(64, 64, 64, 1, DataType::FLOAT16)},                   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
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
            "",                                                              // layer_info
            false,                                                           ///< operation does not have weights
            false                                                            // in_place_output_memory{};
    };

    {
        checkSame(wl_ref_less, wl_ref_less_16, "Test 1, convs with FP32 and FP16 output should be equal");
        checkSame(wl_ref2, wl_ref2_16, "Test 2, elemnwise with FP32 and FP16 output should be equal");
    }
}

}  // namespace VPUNN_unit_tests