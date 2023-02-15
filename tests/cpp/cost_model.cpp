// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>
#include "vpu/compatibility/types01.h"
#include "vpu/shave/activation.h"
#include "vpu/shave/data_movement.h"
#include "vpu/shave/elementwise.h"
#include "vpu_cost_model.h"

#ifndef VPU_2_7_MODEL_PATH
#define VPU_2_7_MODEL_PATH "../../../models/vpu_2_7.vpunn"
#endif

#ifndef VPU_2_0_MODEL_PATH
#define VPU_2_0_MODEL_PATH "../../../models/vpu_2_0.vpunn"
#endif

static auto model = VPUNN::VPUCostModel();
static auto model_2_7 = VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH);
static auto model_2_0 = VPUNN::VPUCostModel(VPU_2_0_MODEL_PATH);

TEST(LoadModels, BasicAssertions) {
    EXPECT_EQ(model_2_7.nn_initialized(), true);
    EXPECT_EQ(model_2_0.nn_initialized(), true);
}

TEST(ArchTest2_0, BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_0;

    EXPECT_EQ(input_channels_mac(device), 1u);
    EXPECT_EQ(get_nr_ppe(device), 16u);
    EXPECT_EQ(get_nr_macs(device), 256u);
    EXPECT_EQ(get_dpu_fclk(device), 700u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    700.0f / 20000.0f);
}

TEST(ArchTest2_1, BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_1;

    EXPECT_EQ(input_channels_mac(device), 1u);
    EXPECT_EQ(get_nr_ppe(device), 16u);
    EXPECT_EQ(get_nr_macs(device), 256u);
    EXPECT_EQ(get_dpu_fclk(device), 850u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    850.0f / 20000.0f);
}

TEST(ArchTest2_7, BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;

    EXPECT_EQ(input_channels_mac(device), 8u);
    EXPECT_EQ(get_nr_ppe(device), 64u);
    EXPECT_EQ(get_nr_macs(device), 2048u);
    EXPECT_EQ(get_dpu_fclk(device), 1300u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    1300.0f / 27000.0f);
}

TEST(ArchTest4_0, BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_4_0;

    EXPECT_EQ(input_channels_mac(device), 8u);
    EXPECT_EQ(get_nr_ppe(device), 64u);
    EXPECT_EQ(get_nr_macs(device), 2048u);
    EXPECT_EQ(get_dpu_fclk(device), 1700u);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8), device,
                                                   VPUNN::MemoryLocation::DRAM),
                    1700.0f / 45000.0f);
}

TEST(BITC2_7, BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;
    const auto tensor = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(tensor, device, VPUNN::MemoryLocation::CMX, false),
                    2 * get_bandwidth_cycles_per_bytes(tensor, device, VPUNN::MemoryLocation::CMX, true));
}

TEST(Permute_2_7, BasicAssertions) {
    const auto device = VPUNN::VPUDevice::VPU_2_7;
    const auto tensor_fp16 = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::FLOAT16);
    const auto tensor_uint8 = VPUNN::VPUTensor({56, 56, 64, 1}, VPUNN::DataType::UINT8);
    // BITC enabled takes has a smaller cycles/byte bw
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(tensor_fp16, device, VPUNN::MemoryLocation::CMX, false, true),
                    0.5f * 1300.0f / 975.0f);
    EXPECT_FLOAT_EQ(get_bandwidth_cycles_per_bytes(tensor_uint8, device, VPUNN::MemoryLocation::CMX, false, true),
                    1.0f * 1300.0f / 975.0f);
}

// Demonstrate some basic assertions.
TEST(BatchTestVPUNN, BasicAssertions_NN2_7) {
    // Dummy WL
    VPUNN::DPUWorkload wl0 = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    VPUNN::DPUWorkload wl1 = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            //{VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_8x16                           // execution mode
    };

    std::vector<VPUNN::DPUWorkload> workloads = {wl0, wl1, wl1, wl0};

    VPUNN::Preprocessing_Interface10<float>
            preprocessing;  // this has to be in sync with what model we will use in the test
    auto input_size = preprocessing.output_size();
    auto batch_size = (unsigned int)workloads.size();

    auto runtime_model = VPUNN::Runtime(VPU_2_7_MODEL_PATH, 1);
    auto model_batched = VPUNN::Runtime(VPU_2_7_MODEL_PATH, batch_size);

    ASSERT_EQ(runtime_model.model_version_info().get_input_interface_version(), preprocessing.interface_version())
            << "Runtime expected input version and preprocessing version must be the same" << std::endl;
    ASSERT_EQ(model_batched.model_version_info().get_input_interface_version(), preprocessing.interface_version())
            << "Runtime expected input version and preprocessing version must be the same" << std::endl;

    auto input_shape = runtime_model.input_tensors()[0]->shape();
    auto output_shape = runtime_model.output_tensors()[0]->shape();
    auto input_shape_batched = model_batched.input_tensors()[0]->shape();

    // Expect proper batch
    EXPECT_EQ(input_shape[0], 1u);
    EXPECT_EQ(input_shape_batched[0], batch_size);

    // Expect model size to match
    EXPECT_EQ(input_size, input_shape_batched[1]);
    EXPECT_EQ(input_shape[1], input_shape_batched[1]);

    // Run inference separately on each batch
    std::vector<float*> inference = std::vector<float*>(batch_size);
    for (unsigned int idx = 0; idx < batch_size; idx++) {
        const float* const data = &(preprocessing.transform(workloads[idx])[0]);
        // Create a new pointer otherwise it will get overwritten every time
        inference[idx] = new float[output_shape[1]];
        memcpy((void*)inference[idx], runtime_model.predict(data, input_size), sizeof(float) * output_shape[1]);
    }

    // Expect output 0 and 2 and 1 and 3 to be the same
    EXPECT_EQ(inference.size(), 4);
    for (unsigned int data_idx = 0; data_idx < output_shape[1]; data_idx++) {
        EXPECT_FLOAT_EQ(inference[0][data_idx], inference[3][data_idx]);
        EXPECT_FLOAT_EQ(inference[1][data_idx], inference[2][data_idx]);

        EXPECT_NE(inference[0][data_idx], inference[1][data_idx]);
        EXPECT_NE(inference[2][data_idx], inference[3][data_idx]);
    }

    // Duplicate batched data
    auto batched_data = preprocessing.transform(workloads);
    float* inference_batched = (float*)model_batched.predict(batched_data.data(), input_size * batch_size);

    for (unsigned int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (unsigned int data_idx = 0; data_idx < output_shape[1]; data_idx++) {
            EXPECT_FLOAT_EQ(inference[batch_idx][data_idx], inference_batched[batch_idx * output_shape[1] + data_idx]);
        }
    }

    // Delete heap array to avoid memory leaks
    for (unsigned int idx = 0; idx < batch_size; idx++) {
        delete[] inference[idx];
    }
}

// Demonstrate some basic assertions.
TEST(BatchTestVPUNNCostModel_VPUNN_2_0, BasicAssertions) {
    // due to the random workloads this test sometimes fails. Reason: the epsilon = 0.001 is slightly overshoot
    const std::string model_path = VPU_2_0_MODEL_PATH;
    const VPUNN::VPUDevice device_version = VPUNN::VPUDevice::VPU_2_0;

    for (auto batch_size : {1, 2, 4, 5, 7, 11, 13, 17, 30, 40, 100}) {
        for (auto n_workloads : {1, 2, 4, 5, 7, 11, 13, 17, 30, 40, 100}) {
            VPUNN::VPUCostModel batched_model = VPUNN::VPUCostModel(model_path, false, 0, batch_size);
            VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path, false, 0, 1);

            // Generate a bunch of random workloads
            auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
            std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));

            // Batched Inference
            std::vector<float> batched_hw_overhead = batched_model.compute_hw_overhead(workloads);
            // Standard Inference
            for (unsigned int idx = 0; idx < workloads.size(); idx++) {
                auto hw_overhead = vpunn_model.compute_hw_overhead(workloads[idx]);
                EXPECT_LE(std::abs(batched_hw_overhead[idx] - hw_overhead) / hw_overhead, 0.001);
            }

            // Batched Inference for cycles
            std::vector<unsigned int> batched_cycles = batched_model.DPU(workloads);
            // Standard Inference for cycles
            for (unsigned int idx = 0; idx < workloads.size(); idx++) {
                unsigned int cycles = vpunn_model.DPU(workloads[idx]);
                EXPECT_LE(abs((float)batched_cycles[idx] - (float)cycles) / float(cycles), 0.001);
            }
        }
    }
}

// Demonstrate some basic assertions.
TEST(SmokeTestDPU, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::MATRIX                                // execution mode
    };

    auto dpu_cycles = model.DPU(wl);

    // Expect equality.
    EXPECT_EQ(dpu_cycles, static_cast<unsigned int>((56 / 4) * (56 / 4) * (16 / 16) * 3 * 3 * 16));
}

// Demonstrate some basic assertions.
TEST(SmokeTestDPUVPU_2_0Model, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            //{VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions 1
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 1, 1, 1},                                                 // padding
            VPUNN::ExecutionMode::VECTOR_FP16                             // execution mode
    };

    float overhead = static_cast<float>(model_2_0.compute_hw_overhead(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(overhead > 1);
}

TEST(SmokeTestDPUVPU_2_0Model_Eltwise, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            //{VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::VECTOR                                // execution mode
    };

    float cycles = static_cast<float>(model_2_0.DPU(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(cycles > (16 / 4) * (16 / 4) * (64 / 16));
}

TEST(SmokeTestDPUVPU27Model_Eltwise, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            //{VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    float cycles = static_cast<float>(model_2_7.DPU(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(cycles > (16 / 4) * (16 / 4) * (64 / 16));
}

// Demonstrate some basic assertions.
TEST(SmokeTestDPUVPU27Model, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            //{VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    float overhead = static_cast<float>(model_2_7.compute_hw_overhead(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(overhead > 1);
}

TEST(SmokeTestDMA, BasicAssertions) {
    auto dma_cycles = model.DMA(VPUNN::VPUDevice::VPU_2_7, VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8),
                                VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                                VPUNN::MemoryLocation::CMX);

    auto dma_cycles_model =
            model.DMA(VPUNN::VPUDevice::VPU_2_7, VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8),
                      VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                      VPUNN::MemoryLocation::CMX);

    // Expect equality.
    EXPECT_EQ(dma_cycles, 950 + static_cast<unsigned int>(ceil(56 * 56 * 16 * 1300 / 27000.0f)));
    EXPECT_EQ(dma_cycles, dma_cycles_model);
}

TEST(SmokeTestCompressedDMA, BasicAssertions) {
    // Compressed DMA with 50% CR
    auto dma_cycles = model.DMA(VPUNN::VPUDevice::VPU_2_7, VPUNN::VPUTensor(25088, 1, 1, 1, VPUNN::DataType::UINT8),
                                VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                                VPUNN::MemoryLocation::CMX);

    // Expect equality.
    EXPECT_EQ(dma_cycles, 950 + static_cast<unsigned int>(ceil(25088 * 1300 / 27000.0f)));
}

TEST(SmokeTestPermutedDMA, BasicAssertions) {
    // DMA + Permute
    auto dma_cycles_fp16 = model.DMA(VPUNN::VPUDevice::VPU_2_7,
                                     VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::FLOAT16, VPUNN::Layout::CMAJOR),
                                     VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::FLOAT16, VPUNN::Layout::ZMAJOR),
                                     VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX);

    auto dma_cycles_uint8 = model.DMA(VPUNN::VPUDevice::VPU_2_7,
                                      VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::CMAJOR),
                                      VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::ZMAJOR),
                                      VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX);

    // Expect equality.
    unsigned int tensor_size_fp16 = 56 * 56 * 16 * 2;
    unsigned int tensor_size_uint8 = 56 * 56 * 16;
    float fclk_ratio = 1300.0f / 975.0f;
    EXPECT_EQ(dma_cycles_fp16, 950 + static_cast<unsigned int>(ceil(tensor_size_fp16 / 2.0f * fclk_ratio)));
    EXPECT_EQ(dma_cycles_uint8, 950 + static_cast<unsigned int>(ceil(tensor_size_uint8 / 1.0f * fclk_ratio)));
}

/// @brief Tests that the Shave objects can be created. Not covering every functionality/shave
class TestSHAVE : public ::testing::Test {
public:
protected:
    VPUNN::VPUCostModel empty_model{VPUNN::VPUCostModel()};

    const VPUNN::VPUTensor input_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};   // input dimensions
    const VPUNN::VPUTensor output_0{56, 56, 16, 1, VPUNN::DataType::FLOAT16};  // output dimensions

    void SetUp() override {
    }

private:
};

/// @brief tests that an activation can be instantiated
TEST_F(TestSHAVE, BasicAssertionsActivationCategory) {
    constexpr unsigned int efficiencyx1K{2000};
    constexpr unsigned int latency{1000};
    auto swwl = VPUNN::SHVActivation<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7,
                                                             input_0,  // input dimensions
                                                             output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Element wise can be instantiated
TEST_F(TestSHAVE, BasicAssertionsELMWiseCategory) {
    constexpr unsigned int efficiencyx1K{800};
    constexpr unsigned int latency{1300};
    auto swwl = VPUNN::SHVElementwise<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7, {input_0},  // input dimensions
                                                              output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Data Movement can be instantiated
TEST_F(TestSHAVE, BasicAssertionsDataMovementCategory) {
    constexpr unsigned int efficiencyx1K{2050};
    constexpr unsigned int latency{3000};
    auto swwl = VPUNN::SHVDataMovement<efficiencyx1K, latency>(VPUNN::VPUDevice::VPU_2_7,
                                                               input_0,  // input dimensions
                                                               output_0  // output dimensions
    );
    // init part
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    // behavioral part
    EXPECT_NEAR(swwl.getKernelEfficiency(), efficiencyx1K / 1000.0F, 0.001F);
    EXPECT_EQ(swwl.getLatency(), latency);

    const auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    const auto op_count = swwl.outputs[0].size();

    EXPECT_NEAR(shave_cycles_sigmoid, round(op_count / swwl.getKernelEfficiency()) + swwl.getLatency(), 1.0F);
}

/// @brief tests that an Sigmoid can be instantiated
TEST_F(TestSHAVE, BasicAssertionsActivationSigmoid) {
    auto swwl = VPUNN::SHVSigmoid(VPUNN::VPUDevice::VPU_2_7,
                                  input_0,  // input dimensions
                                  output_0  // output dimensions
    );
    EXPECT_EQ(swwl.device, VPUNN::VPUDevice::VPU_2_7);
    ASSERT_EQ(swwl.inputs.size(), 1);
    EXPECT_EQ(swwl.inputs[0].size(), input_0.size()) << " clearly they are not the same" << std::endl;

    ASSERT_EQ(swwl.outputs.size(), 1);
    EXPECT_EQ(swwl.outputs[0].size(), output_0.size()) << " clearly they are not the same" << std::endl;

    auto shave_cycles_sigmoid = empty_model.SHAVE(swwl);
    // Expect equality.
    EXPECT_GE(shave_cycles_sigmoid, 0u);
}

class TestCostModel : public ::testing::Test {
public:
protected:
    VPUNN::DPUWorkload wl{VPUNN::VPUDevice::VPU_2_7,
                          VPUNN::Operation::CONVOLUTION,
                          {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                          //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
                          {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                          {3, 3},                                                     // kernels
                          {1, 1},                                                     // strides
                          {1, 1},                                                     // padding
                          VPUNN::ExecutionMode::CUBOID_16x16};

    void SetUp() override {
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

private:
};

// Demonstrate some basic assertions.
TEST_F(TestCostModel, InitAspects) {
    {  // 20
        const std::string model_path = VPU_2_0_MODEL_PATH;
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());

        const auto file_content{read_a_file(model_path)};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(file_content.data(), file_content.size(), true));
        auto vpunn_model_buf = VPUNN::VPUCostModel(file_content.data(), file_content.size(), true);
        EXPECT_TRUE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(file_content.data(), file_content.size(), false));
        auto vpunn_model_buf_copy = VPUNN::VPUCostModel(file_content.data(), file_content.size(), false);
        EXPECT_TRUE(vpunn_model_buf_copy.nn_initialized());
    }
    {  // 27
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        EXPECT_TRUE(vpunn_model.nn_initialized());

        const auto file_content{read_a_file(model_path)};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(file_content.data(), file_content.size(), true));
        auto vpunn_model_buf = VPUNN::VPUCostModel(file_content.data(), file_content.size(), true);
        EXPECT_TRUE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(file_content.data(), file_content.size(), false));
        auto vpunn_model_buf_copy = VPUNN::VPUCostModel(file_content.data(), file_content.size(), false);
        EXPECT_TRUE(vpunn_model_buf_copy.nn_initialized());
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        EXPECT_FALSE(vpunn_model.nn_initialized());

        const decltype(read_a_file("")) file_content{'M', 'u', 's', 't', 'h', 'a', 'v', 'e', ' ', '0', '1'};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(file_content.data(), file_content.size(), true));
        auto vpunn_model_buf = VPUNN::VPUCostModel(file_content.data(), file_content.size(), true);
        EXPECT_FALSE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(file_content.data(), file_content.size(), false));
        auto vpunn_model_buf_copy = VPUNN::VPUCostModel(file_content.data(), file_content.size(), false);
        EXPECT_FALSE(vpunn_model_buf_copy.nn_initialized());

        EXPECT_FLOAT_EQ(vpunn_model_buf.compute_hw_overhead(wl), 1.0F);
        EXPECT_FLOAT_EQ(vpunn_model_buf_copy.compute_hw_overhead(wl), 1.0F);
    }
}

TEST_F(TestCostModel, DISABLED_ComaparativeRuns) {
    const std::string model_path = VPU_2_0_MODEL_PATH;

    auto modelRun = [](const std::string& model_path, VPUNN::DPUWorkload& wld) {
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        std::cout << model_path << " : initialized: " << vpunn_model.nn_initialized() << std::endl;

        std::cout << "compute_hw_overhead(wl)   : " << vpunn_model.compute_hw_overhead(wld) << std::endl;
        std::cout << "DPU(wl)   : " << vpunn_model.DPU(wld) << std::endl;
        std::cout << "hw_utilization(wl)   : " << vpunn_model.hw_utilization(wld) << std::endl;
    };

    std::cout << "----------------------------------------------------------\n";
    modelRun(VPU_2_0_MODEL_PATH, wl);
    modelRun(VPU_2_0_MODEL_PATH, wl);
    std::cout << "----------------------------------------------------------\n";
    modelRun(VPU_2_7_MODEL_PATH, wl);
    modelRun(VPU_2_7_MODEL_PATH, wl);
    std::cout << "----------------------------------------------------------\n";
    // modelRun("c:/gitwrk/libraries.performance.modeling.vpu.nn-cost-model/models/torch.nn.vpunn", wl);
    // modelRun("c:/gitwrk/libraries.performance.modeling.vpu.nn-cost-model/models/torch.nn.vpunn", wl);
    // std::cout << "----------------------------------------------------------\n";
    // modelRun("c:/gitwrk/libraries.performance.modeling.vpu.nn-cost-model/models/torch.nn00.vpunn", wl);
    // modelRun("c:/gitwrk/libraries.performance.modeling.vpu.nn-cost-model/models/torch.nn00.vpunn", wl);
    std::cout << "----------------------------------------------------------\n";
    // c:\gitwrk\libraries.performance.modeling.vpu.nn-cost-model\models\torch.nn.vpunn
    // c:\gitwrk\libraries.performance.modeling.vpu.nn-cost-model\models\torch.nn00.vpunn
}

/// @brief tests that show backwards compatibility with NN with different versions
class TestNNModelCompatibility : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    std::string get_model_root() const {
        const std::string m{VPU_2_0_MODEL_PATH};
        std::string model_root = m.substr(0, m.find_last_of('/') + 1);

        // std::cout << "Root model is :" << model_root << std::endl;
        return model_root;
    }
    const std::string VPU20_default_{get_model_root() +
                                   "../tests/cpp/nn_model_versions/vpu_2_0-default_initial.vpunn"};
    const std::string VPU27_default_{get_model_root() +
                                   "../tests/cpp/nn_model_versions/vpu_2_7-default_initial.vpunn"};
    const std::string VPU27_10_2_{get_model_root() + "../tests/cpp/nn_model_versions/vpu_2_7_v-10-2.vpunn"};

    VPUNN::DPUWorkload wl_20{VPUNN::VPUDevice::VPU_2_0,
                             VPUNN::Operation::CONVOLUTION,
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                             //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                             {3, 3},                                                     // kernels
                             {1, 1},                                                     // strides
                             {1, 1},                                                     // padding
                             VPUNN::ExecutionMode::CUBOID_16x16};

    VPUNN::DPUWorkload wl_27{VPUNN::VPUDevice::VPU_2_7,
                             VPUNN::Operation::CONVOLUTION,
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                             //{VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions 1
                             {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                             {3, 3},                                                     // kernels
                             {1, 1},                                                     // strides
                             {1, 1},                                                     // padding
                             VPUNN::ExecutionMode::CUBOID_16x16};

    const float epsilon{0.01F};

    VPUNN::VPUCostModel ideal_model = VPUNN::VPUCostModel("");  // model without NN, output is interface 1-hw overhead

private:
};

/// test ideal model, no NN, just theoretical aspects
TEST_F(TestNNModelCompatibility, Ideal_Empty_Model) {
    {
        VPUNN::DPUWorkload wl{wl_20};
        float theoretical_nn_out1 = ideal_model.compute_hw_overhead(wl);  // VPU20 has mode hw overhead
        unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);        // will be computed based o
        float theoretical_hw_util = ideal_model.hw_utilization(wl);       // a float
        ASSERT_EQ(theoretical_nn_out1, 1.0F);
        EXPECT_NEAR(theoretical_hw_util, 1.0F, epsilon);

        EXPECT_EQ(theoretical_dpu_cycles, 28224);
    }
    {
        VPUNN::DPUWorkload wl{wl_27};
        float theoretical_nn_out1 = ideal_model.compute_hw_overhead(wl);  // VPU20 has mode hw overhead
        unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);        // will be computed based o
        float theoretical_hw_util = ideal_model.hw_utilization(wl);       // a float
        ASSERT_EQ(theoretical_nn_out1, 1.0F);
        EXPECT_NEAR(theoretical_hw_util, 1.0F, epsilon);

        EXPECT_EQ(theoretical_dpu_cycles, 22896);
    }
}

/// Test VPU20 exists
TEST_F(TestNNModelCompatibility, VPU20_default_01) {
    VPUNN::DPUWorkload wl{wl_20};
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU20_default_);
    ASSERT_TRUE(vpunn_model.nn_initialized())
            << "Model not loaded, might be due to file location: " << VPU20_default_ << std::endl;

    float nn_out1 = vpunn_model.compute_hw_overhead(wl); 
    unsigned int dpu_cycles = vpunn_model.DPU(wl);
    float hw_util = vpunn_model.hw_utilization(wl);  // a float

    // this version uses hw_overhead at exit, limits it to 1.0
    EXPECT_GE(nn_out1, 1.0F);
    EXPECT_NEAR(nn_out1, 1 / hw_util, epsilon);
    EXPECT_GE(dpu_cycles, 20000);

    unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);  // will be computed based o

    EXPECT_NEAR(dpu_cycles / (theoretical_dpu_cycles / hw_util), 1.0F, epsilon)
            << " info: " << dpu_cycles << " " << theoretical_dpu_cycles << " " << hw_util << std::endl;

    std::cout << "VPU20 original, default wl. "
              << "NN output/hw overhead: " << nn_out1 << ", DPU cycles: " << dpu_cycles
              << ", hw_utilization: " << hw_util << ", Theoretical DPU cycles: " << theoretical_dpu_cycles << std::endl;
}

/// Test VPU27 1 exists
TEST_F(TestNNModelCompatibility, VPU27_default_01) {
    VPUNN::DPUWorkload wl{wl_27};
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU27_default_);
    ASSERT_TRUE(vpunn_model.nn_initialized())
            << "Model not loaded, might be due to file location: " << VPU27_default_ << std::endl;

    float nn_out1 = vpunn_model.compute_hw_overhead(wl);  
    unsigned int dpu_cycles = vpunn_model.DPU(wl);
    float hw_util = vpunn_model.hw_utilization(wl);  // a float

    // this version uses hw_overhead at exit, limits it to 1.0
    EXPECT_GE(nn_out1, 1.0F);
    EXPECT_NEAR(nn_out1, 1 / hw_util, epsilon);
    EXPECT_GE(dpu_cycles, 20000);

    unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);  // will be computed based o

    EXPECT_NEAR(dpu_cycles / (theoretical_dpu_cycles / hw_util), 1.0F, epsilon)
            << " info: " << dpu_cycles << " " << theoretical_dpu_cycles << " " << hw_util << std::endl;

    std::cout << "VPU27 original, default wl. "
              << "NN output/hw overhead: " << nn_out1 << ", DPU cycles: " << dpu_cycles
              << ", hw_utilization: " << hw_util << ", Theoretical DPU cycles: " << theoretical_dpu_cycles << std::endl;
}

/// Test VPU27 10 exists
TEST_F(TestNNModelCompatibility, VPU27_10_2) {
    VPUNN::DPUWorkload wl{wl_27};
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(VPU27_10_2_);
    ASSERT_TRUE(vpunn_model.nn_initialized())
            << "Model not loaded, might be due to file location: " << VPU27_10_2_ << std::endl;

    float nn_out1 = vpunn_model.compute_hw_overhead(wl); 
    unsigned int dpu_cycles = vpunn_model.DPU(wl);
    float hw_util = vpunn_model.hw_utilization(wl);  // a float

    // this version uses cycles at exit
    EXPECT_GE(nn_out1, 10000.0F);  // is 10023.5
    EXPECT_GE(dpu_cycles, 10024);
    EXPECT_NEAR(dpu_cycles, nn_out1, 1.0F);

    EXPECT_GE(hw_util, 2.0F);  // Strange but is like that

    unsigned int theoretical_dpu_cycles = ideal_model.DPU(wl);  // will be computed based o

    EXPECT_NEAR(dpu_cycles / (theoretical_dpu_cycles / hw_util), 1.0F, epsilon)
            << " info: " << dpu_cycles << " " << theoretical_dpu_cycles << " " << hw_util << std::endl;

    std::cout << "VPU27 10 2, default wl. "
              << "NN output/hw overhead: " << nn_out1 << ", DPU cycles: " << dpu_cycles
              << ", hw_utilization: " << hw_util << ", Theoretical DPU cycles: " << theoretical_dpu_cycles << std::endl;
}
