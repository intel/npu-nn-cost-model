// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "costmodel/cost_model.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestCostModelVPU2x : public TestCostModel {
protected:

    const VPUDevice device20{VPUDevice::VPU_2_0};
    const VPUDevice device27{VPUDevice::VPU_2_7};

    const std::string model20_path{VPU_2_0_MODEL_PATH};
    const std::string model27_path{VPU_2_7_MODEL_PATH};
};

TEST_F(TestCostModelVPU2x, MAXPOOL_172_VPU27_NoGT) {
    DPUWorkload tst_wl{
            device27,                                   // dev
            Operation::MAXPOOL,                         // op
            {VPUTensor(7, 7, 64, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(1, 1, 64, 1, DataType::UINT8)},  // output dimensions
            {7, 7},                                     // kernels
            {1, 1},                                     // strides
            {0, 0, 0, 0},                               // padding
            ExecutionMode::CUBOID_16x16,                // execution mode
            ActivationFunction::NONE,                   // activation
            0.0F,                                       // act_sparsity
            0.0F,                                       // weight_sparsity
            {swz_def, swz_def},                         // input_swizzling
            {swz_def},                                  // output_swizzling
            1,                                          // output_write_tiles
            {0, 0, 0, 0},                               // offsets
            ISIStrategy::CLUSTERING,                    // isi_strategy
            false,                                      // weight_sparsity_enabled
    };
    auto wl_owt2{tst_wl};
    wl_owt2.output_write_tiles = 2;

    auto wl_sok2{wl_owt2};
    wl_sok2.isi_strategy = ISIStrategy::SPLIT_OVER_K;

    const std::vector<GTestCase> tests{
            {{std::move(tst_wl)},
             {NO_ERROR_EXPECTED, false, 1500, 1500 + 250},  // 1659    GT:1660
             "T1"},
            {{std::move(wl_owt2)},
             {NO_ERROR_EXPECTED, false, 1450, 1450 + 250},  // 1689 NOK GT: NA  cannot do! //v150: 1468
             "T2 clu owt=2"},
            {{std::move(wl_sok2)},
             {NO_ERROR_EXPECTED, false, 1450, 1450 + 250},  // 1689  GT: NA     //v150: 1468
             " SOK owt=2"},

    };

    executeTests(tests);
}

TEST_F(TestCostModelVPU2x, MAXPOOL_Example_NoGT) {
    // slow: 17015 fast : 5250
    const DPUWorkload wl1{
            device27,                                                    // dev
            VPUNN::Operation::MAXPOOL,                                   // op
            {VPUNN::VPUTensor(43, 181, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(42, 180, 32, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {2, 2},                                                      // kernels
            {1, 1},                                                      // strides
            {0, 0, 0, 0},                                                // padding
            ExecutionMode::CUBOID_16x16,                                 // execution mode
            ActivationFunction::NONE,                                    // activation
            0.0F,                                                        // act_sparsity
            0.0F,                                                        // weight_sparsity
            {swz_def, swz_def},                                          // input_swizzling
            {swz_def},                                                   // output_swizzling
            2,                                                           // output_write_tiles
            {0, 0, 0, 0},                                                // offsets
            ISIStrategy::SPLIT_OVER_K,                                   // isi_strategy
            false,                                                       // weight_sparsity_enabled
    };

    auto wl1_halfsok{wl1};
    wl1_halfsok.inputs[0] = VPUTensor(43, 181, 32 / 2, 1, DataType::UINT8);
    wl1_halfsok.outputs[0] = VPUTensor(42, 180, 32 / 2, 1, DataType::UINT8);

    auto wl2{wl1};
    wl2.isi_strategy = ISIStrategy::CLUSTERING;
    wl2.output_write_tiles = 1;

    auto wl3{wl2};
    wl2.isi_strategy = ISIStrategy::CLUSTERING;
    wl2.output_write_tiles = 2;

    auto wl4{wl2};
    wl4.isi_strategy = ISIStrategy::SPLIT_OVER_H;
    wl4.output_write_tiles = 2;

    auto wl5{wl4};
    wl5.output_write_tiles = 1;

    auto wl5_halfsoh{wl5};
    wl5_halfsoh.inputs[0] = VPUTensor(43, 91, 32, 1, DataType::UINT8);
    wl5_halfsoh.outputs[0] = VPUTensor(42, 90, 32, 1, DataType::UINT8);

    // EXPECT_TRUE(false);
    const std::vector<GTestCase> tests{
            {{std::move(wl1_halfsok)},
             {NO_ERROR_EXPECTED, false, 9600,
              9600 + 1100},  //  v16: xxx  v17:10558    // v1.5.9 9635  GT: 10388 assumed for x16
             "T1 HALF SOK"},
            {{std::move(wl1)},
             {NO_ERROR_EXPECTED, false, 12000,
              12000 + 2000},  //  v16: 13471  v17:13486    // v1.5.9 12897  GT: 12506,  14399 cc if no swizzling
             "T1 SOK"},
            {{std::move(wl2)},
             {NO_ERROR_EXPECTED, false, 11900,
              11900 + 2000},  //  v16: 13634  v17:13559  (much! vs GT)  // v1.5.9 13089   ,  GT 11959  CLU
             "T2 CLU owt1"},
            {{std::move(wl3)},
             {NO_ERROR_EXPECTED, false, 12000,
              12000 + 2000},  // OK vs GTclu, NOK vs NN clu: v16: 12044  v17:12437 , v1.5.9 11233  should be =
                              // SOK, and larger than owt=1  (CLU is 11959) GT NA
             "T3 CLU owt2"},
            {{std::move(wl4)},
             {NO_ERROR_EXPECTED, false, 13000, 13000 + 2000},  // OK  v16: 14028 v17: 13665// v1.5.9 13218   GT ???
             "T4 SOH H owt2"},
            {{std::move(wl5)},
             {NO_ERROR_EXPECTED, false, 12000, 12000 + 2000},  // v16: 12244  NOK v17: 12604 nok should be larger than
                                                               // CLU (is larger than GT CLU still) // v1.5.9
                                                               // 11515   //GT 12484cc
             "T5h SOH H owt1"},

            {{std::move(wl5_halfsoh)},
             {NO_ERROR_EXPECTED, false, 6000, 6000 + 600},  // v16: 6499  NOK v17: x  //GT 6162  aprox
             "T5h SOH H half owt1"},

    };

    executeTests(tests);
}

TEST_F(TestCostModelVPU2x, LoadModels_BasicAssertions) {
    {  // 2_0
        const std::string model_path = VPU_2_0_MODEL_PATH;
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }

    {  // 2_7
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }

    {  // 2_0 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH)};
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }

    {  // 2_7 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH)};
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }
}

TEST_F(TestCostModelVPU2x, LoadModels_NN_Valid_Interval) {
    float down_exp = 0.0F;
    float up_exp = 4000000000.0F;

    {  // empty models

        ASSERT_FALSE(empty_model.nn_initialized());
        auto minmax = empty_model.get_NN_cost_provider().get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_0
        const std::string model_path = VPU_2_0_MODEL_PATH;
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_cost_provider().get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_7
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_cost_provider().get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_0 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH)};
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_cost_provider().get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_7 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH)};
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model_path});
        VPUNN::VPUCostModel vpunn_model{model_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_cost_provider().get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }
}

// Demonstrate some basic assertions.
TEST_F(TestCostModelVPU2x, BatchTestVPUNN_NN2_7) {
    // Dummy WL
    VPUNN::DPUWorkload wl0 = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    VPUNN::DPUWorkload wl1 = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_8x16                           // execution mode
    };

    std::vector<VPUNN::DPUWorkload> workloads = {wl0, wl1, wl1, wl0};

    VPUNN::Preprocessing_Interface11<float>
            preprocessing;  // this has to be in sync with what model we will use in the test
    auto input_size = preprocessing.output_size();
    auto batch_size = (unsigned int)workloads.size();

    const auto runtime_model = VPUNN::Runtime(model27_path);
    InferenceExecutionData runtime_buffer_data{runtime_model.createNewInferenceExecutionData(1)};

    const auto model_batched = VPUNN::Runtime(model27_path);
    InferenceExecutionData batched_buffer_data{model_batched.createNewInferenceExecutionData(batch_size)};

    ASSERT_EQ(runtime_model.model_version_info().get_input_interface_version(), preprocessing.interface_version())
            << "Runtime expected input version and preprocessing version must be the same" << std::endl;
    ASSERT_EQ(model_batched.model_version_info().get_input_interface_version(), preprocessing.interface_version())
            << "Runtime expected input version and preprocessing version must be the same" << std::endl;

    // const auto& input_shape = runtime_model.input_tensors()[0]->shape();
    // const auto& output_shape = runtime_model.output_tensors()[0]->shape();
    // const auto& input_shape_batched = model_batched.input_tensors()[0]->shape();

    const std::vector<unsigned int> input_shape{runtime_buffer_data.input_shapes()[0]};
    const std::vector<unsigned int> output_shape{runtime_buffer_data.output_shapes()[0]};
    const std::vector<unsigned int> input_shape_batched{batched_buffer_data.input_shapes()[0]};

    // Expect proper batch
    EXPECT_EQ(input_shape[0], 1u);
    EXPECT_EQ(input_shape_batched[0], batch_size);

    // Expect model size to match
    EXPECT_EQ(input_size, input_shape_batched[1]);
    EXPECT_EQ(input_shape[1], input_shape_batched[1]);

    // Run inference separately on each batch
    std::vector<float*> inference = std::vector<float*>(batch_size);
    std::vector<std::vector<float>> inference_vector = std::vector<std::vector<float>>(batch_size);
    for (unsigned int idx = 0; idx < batch_size; idx++) {
        const std::vector<float> data = preprocessing.transformSingle(workloads[idx]);
        // Create a new pointer otherwise it will get overwritten every time
        inference[idx] = new float[output_shape[1]];
        memcpy((void*)inference[idx], runtime_model.predict(&(data[0]), input_size, runtime_buffer_data),
               sizeof(float) * output_shape[1]);
        // Compute inference using the vector interface
        inference_vector[idx] = runtime_model.predict<float>(data, runtime_buffer_data);
    }

    // Expect output 0 and 2 and 1 and 3 to be the same
    EXPECT_EQ(inference.size(), 4);
    EXPECT_EQ(inference_vector.size(), 4);
    for (unsigned int data_idx = 0; data_idx < output_shape[1]; data_idx++) {
        EXPECT_FLOAT_EQ(inference[0][data_idx], inference[3][data_idx]);
        EXPECT_FLOAT_EQ(inference[1][data_idx], inference[2][data_idx]);

        EXPECT_NE(inference[0][data_idx], inference[1][data_idx]);
        EXPECT_NE(inference[2][data_idx], inference[3][data_idx]);

        EXPECT_FLOAT_EQ(inference_vector[0][data_idx], inference_vector[3][data_idx]);
        EXPECT_FLOAT_EQ(inference_vector[1][data_idx], inference_vector[2][data_idx]);

        EXPECT_NE(inference_vector[0][data_idx], inference_vector[1][data_idx]);
        EXPECT_NE(inference_vector[2][data_idx], inference_vector[3][data_idx]);

        // Both interface should lead the same result
        EXPECT_FLOAT_EQ(inference_vector[0][data_idx], inference[0][data_idx]);
        EXPECT_FLOAT_EQ(inference_vector[1][data_idx], inference[1][data_idx]);
        EXPECT_FLOAT_EQ(inference_vector[2][data_idx], inference[2][data_idx]);
        EXPECT_FLOAT_EQ(inference_vector[3][data_idx], inference[3][data_idx]);
    }

    // Duplicate batched data
    const auto batched_data{preprocessing.transformBatch(workloads)};
    float* inference_batched =
            (float*)model_batched.predict(batched_data.data(), input_size * batch_size, batched_buffer_data);

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

/// Demonstrate outputs from batch are the same as from single runs. Random data
TEST_F(TestCostModelVPU2x, DISABLED_BatchTestVPUNNCostModel_VPUNN_2_0_stochastic) {
    // due to the random workloads this test sometimes fails. Reason: the epsilon = 0.001 is slightly overshoot

    for (unsigned int batch_size : {1, 2, 3, 5, 8, 13, 17, 30, 40, 100}) {
        for (auto n_workloads : {1, 2, 3, 5, 8, 13, 17, 30, 40, 100}) {
            VPUNN::VPUCostModel batched_model{model20_path, false, 0, batch_size};
            VPUNN::VPUCostModel vpunn_model{model20_path, false, 0, 1};

            // Generate a bunch of random workloads
            auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
            std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device20));

            {
                // Batched Inference for cycles.
                std::vector<VPUNN::CyclesInterfaceType> batched_cycles = batched_model.DPU(workloads);
                // Standard Inference for cycles
                for (unsigned int idx = 0; idx < workloads.size(); idx++) {
                    VPUNN::CyclesInterfaceType cycles_s = vpunn_model.DPU(workloads[idx]);
                    VPUNN::CyclesInterfaceType cycles_b = batched_cycles[idx];

                    if (VPUNN::Cycles::isErrorCode(cycles_s) ||
                        VPUNN::Cycles::isErrorCode(cycles_b)) {  // one is error expect exact eq
                        EXPECT_EQ(cycles_b, cycles_s)
                                << "batched value: " << cycles_b
                                << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_b)
                                << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_b) << std::endl
                                << "single run value: " << cycles_s
                                << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_s)
                                << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_s) << std::endl
                                << " , idx: " << idx << ", batch_size : " << batch_size
                                << " n_workloads : " << n_workloads << std::endl
                                << workloads[idx] << std::endl;
                    } else {  // regular cycles value
                        const VPUNN::CyclesInterfaceType delta{delta_cycles(cycles_b, cycles_s)};
                        const VPUNN::CyclesInterfaceType tolerance{max_tolerance_cycles(cycles_b, cycles_s, 2)};
                        EXPECT_LE(delta, tolerance)
                                << "batched value: " << cycles_b << " ,single run value: " << cycles_s
                                << " , idx: " << idx << ", batch_size : " << batch_size
                                << " n_workloads : " << n_workloads << std::endl
                                << workloads[idx] << std::endl;
                    }
                }
            }
        }
    }
}

/// Demonstrate outputs from batch are the same as from single runs. Random data
TEST_F(TestCostModelVPU2x, DISABLED_BatchTestVPUNNCostModel_VPUNN_2_7F_stochastic) {
    const std::string model_path = the_NN_models.fast_model_paths[1].first;
    const VPUNN::VPUDevice device_version = the_NN_models.fast_model_paths[1].second;

    for (unsigned int batch_size : {1, 2, 3, 5, 8, 13, 17, 30, 40, 100}) {
        for (auto n_workloads : {1, 2, 3, 5, 8, 13, 17, 30, 40, 100}) {
            VPUNN::VPUCostModel batched_model{model_path, false, 0, batch_size};
            VPUNN::VPUCostModel vpunn_model{model_path, false, 0, 1};

            // Generate a bunch of random workloads
            auto workloads = std::vector<VPUNN::DPUWorkload>(n_workloads);
            std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));

            {
                // Batched Inference for cycles.
                std::vector<VPUNN::CyclesInterfaceType> batched_cycles = batched_model.DPU(workloads);
                // Standard Inference for cycles
                for (unsigned int idx = 0; idx < workloads.size(); ++idx) {
                    VPUNN::CyclesInterfaceType cycles_s = vpunn_model.DPU(workloads[idx]);
                    VPUNN::CyclesInterfaceType cycles_b = batched_cycles[idx];

                    if (VPUNN::Cycles::isErrorCode(cycles_s) ||
                        VPUNN::Cycles::isErrorCode(cycles_b)) {  // one is error expect exact eq
                        EXPECT_EQ(cycles_b, cycles_s)
                                << "batched value: " << cycles_b
                                << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_b)
                                << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_b) << std::endl
                                << "single run value: " << cycles_s
                                << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_s)
                                << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_s) << std::endl
                                << " , idx: " << idx << ", batch_size : " << batch_size
                                << " n_workloads : " << n_workloads << std::endl
                                << workloads[idx] << std::endl;
                    } else {  // regular cycles value
                        const VPUNN::CyclesInterfaceType delta{delta_cycles(cycles_b, cycles_s)};
                        const VPUNN::CyclesInterfaceType tolerance{max_tolerance_cycles(cycles_b, cycles_s)};
                        EXPECT_LE(delta, tolerance)
                                << "batched value: " << cycles_b << " ,single run value: " << cycles_s
                                << " , idx: " << idx << ", batch_size : " << batch_size
                                << " n_workloads : " << n_workloads << std::endl
                                << workloads[idx] << std::endl;
                    }
                }
            }
        }
    }
}

TEST_F(TestCostModelVPU2x, BatchTestVPUNNCostModel_VPUNN_2_7F_Particular1) {
    const std::string model_path = the_NN_models.fast_model_paths[1].first;
    const VPUNN::VPUDevice device_version = the_NN_models.fast_model_paths[1].second;
    unsigned int batch_size = 1;
    const int n_workloads = 1;

    VPUNN::VPUCostModel batched_model{model_path, false, 0, batch_size};
    VPUNN::VPUCostModel vpunn_model{model_path, false, 0, 1};

    VPUNN::DPUWorkload wl_ref = {
            device_version,
            VPUNN::Operation::DW_CONVOLUTION,
            {VPUNN::VPUTensor(1278, 3, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(1273, 2, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {8, 8},                                                      // kernels
            {1, 1},                                                      // strides
            {3, 3, 1, 1},                                                // padding
            VPUNN::ExecutionMode::CUBOID_8x16                            // execution mode
    };
    wl_ref.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;

    // Generate a bunch of random workloads
    auto workloads = std::vector<VPUNN::DPUWorkload>(1);
    // std::generate_n(workloads.begin(), n_workloads, VPUNN::randDPUWorkload(device_version));
    workloads[0] = std::move(wl_ref);

    const auto& w0{workloads[0]};

    {
        // Batched Inference for cycles.
        std::vector<VPUNN::CyclesInterfaceType> batched_cycles = batched_model.DPU(workloads);

        const auto& w01{workloads[0]};
        EXPECT_EQ(w0, w01) << w0 << w01;

        // Standard Inference for cycles
        unsigned int idx = 0;
        {
            VPUNN::CyclesInterfaceType cycles_s = vpunn_model.DPU(workloads[idx]);
            VPUNN::CyclesInterfaceType cycles_b = batched_cycles[idx];

            EXPECT_EQ(w0, workloads[idx]);

            if (VPUNN::Cycles::isErrorCode(cycles_s) ||
                VPUNN::Cycles::isErrorCode(cycles_b)) {  // one is error expect exact eq
                EXPECT_EQ(cycles_b, cycles_s)
                        << "batched value: " << cycles_b << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_b)
                        << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_b) << std::endl
                        << "single run value: " << cycles_s << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_s)
                        << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_s) << std::endl
                        << " , idx: " << idx << ", batch_size : " << batch_size << " n_workloads : " << n_workloads
                        << std::endl
                        << workloads[idx] << std::endl;
            } else {  // regular cycles value
                const VPUNN::CyclesInterfaceType delta{delta_cycles(cycles_b, cycles_s)};
                const VPUNN::CyclesInterfaceType tolerance{max_tolerance_cycles(cycles_b, cycles_s)};
                EXPECT_LE(delta, tolerance)
                        << "batched value: " << cycles_b << " ,single run value: " << cycles_s << " , idx: " << idx
                        << ", batch_size : " << batch_size << " n_workloads : " << n_workloads << std::endl
                        << workloads[idx] << std::endl;
            }
        }
    }
}

// Demonstrate some basic assertions.
TEST_F(TestCostModelVPU2x, BatchTest_SanitizedWorkloadsEquivalence) {
    // Dummy WL
    VPUNN::DPUWorkload wl0 = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    // Int8 type, should be replace at sanitization
    VPUNN::DPUWorkload wl1 = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::INT8)},  // input dimensions
            {VPUNN::VPUTensor(32, 32, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {1, 1},                                                    // kernels
            {1, 1},                                                    // strides
            {0, 0},                                                    // padding
            VPUNN::ExecutionMode::CUBOID_8x16                          // execution mode
    };

    // input output channels must be equal.
    const VPUNN::DPUWorkload wl2 = {
            device27,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(16, 16, 30, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
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
    };

    // avepool is not supported, but should be transformed to an equivalent
    const VPUNN::DPUWorkload wl3 = {
            device27,
            VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
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

    };

    std::vector<VPUNN::DPUWorkload> workloads = {std::move(wl0), std::move(wl1), std::move(wl2), std::move(wl3)};

    auto batch_size = (unsigned int)workloads.size();

    const std::string model_path = the_NN_models.fast_model_paths[1].first;  // fast nn2.7
    VPUNN::VPUCostModel batched_model{model_path, false, 0, batch_size};
    VPUNN::VPUCostModel vpunn_model{model_path, false, 0, 1};

    {  // one by one batch, so if one list throws it will not affect the rest of wl in vector
        for (unsigned int idx = 0; idx < workloads.size(); idx++) {
            std::vector<VPUNN::DPUWorkload> workloads_one{workloads[idx]};
            std::vector<VPUNN::CyclesInterfaceType> batched_cycles{0};           // remains zero in case it throws next
            EXPECT_NO_THROW(batched_cycles = batched_model.DPU(workloads_one));  // only one element in vector

            const VPUNN::CyclesInterfaceType cycles_s = vpunn_model.DPU(workloads_one[0]);
            const VPUNN::CyclesInterfaceType cycles_b = batched_cycles[0];
            EXPECT_EQ(cycles_b, cycles_s)
                    << "batched value: " << cycles_b << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_b)
                    << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_b) << std::endl
                    << "single run value: " << cycles_s << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_s)
                    << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_s) << std::endl
                    << "idx: " << idx << ", batch_size : " << batch_size << std::endl
                    << workloads_one[0] << std::endl
                    << std::endl;
        }
    }

    {  // run all in one vector
        // Batched Inference for cycles
        std::vector<VPUNN::CyclesInterfaceType> batched_cycles;  // = batched_model.DPU(workloads);
        ASSERT_NO_THROW(batched_cycles = batched_model.DPU(workloads));
        // Standard Inference for cycles
        for (unsigned int idx = 0; idx < workloads.size(); idx++) {
            const VPUNN::CyclesInterfaceType cycles_s = vpunn_model.DPU(workloads[idx]);
            const VPUNN::CyclesInterfaceType cycles_b = batched_cycles[idx];
            EXPECT_EQ(cycles_b, cycles_s)
                    << "batched value: " << cycles_b << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_b)
                    << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_b) << std::endl
                    << "single run value: " << cycles_s << " , is_error: " << VPUNN::Cycles::isErrorCode(cycles_s)
                    << " , err_code: " << VPUNN::Cycles::toErrorText(cycles_s) << std::endl
                    << "idx: " << idx << ", batch_size : " << batch_size << std::endl
                    << workloads[idx] << std::endl
                    << std::endl;
        }
    }
}

// Demonstrate some basic assertions.
TEST_F(TestCostModelVPU2x, DISABLED_SmokeTestDPU) {
    VPUNN::DPUWorkload wl = {
            device20,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::MATRIX                                // execution mode
    };

    auto dpu_cycles = empty_model.DPU(std::move(wl));

    // Expect equality.
    EXPECT_EQ(dpu_cycles, static_cast<unsigned int>((56 / 4) * (56 / 4) * (16 / 16) * 3 * 3 * 16));
}

// Demonstrate some basic assertions.
TEST_F(TestCostModelVPU2x, SmokeTestDPUVPU_2_0Model) {
    VPUNN::DPUWorkload wl = {
            device20,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 1, 1, 1},                                                 // padding
            VPUNN::ExecutionMode::VECTOR_FP16                             // execution mode
    };

    auto overhead = cost_models.getModel(device20).DPU(std::move(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(overhead > 1);
    EXPECT_FALSE(Cycles::isErrorCode(overhead));
}

TEST_F(TestCostModelVPU2x, SmokeTestDPUVPU_2_0Model_Eltwise) {
    VPUNN::DPUWorkload wl = {
            device20,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::VECTOR                                // execution mode
    };
    float cycles = static_cast<float>(cost_models.getModel(device20).DPU(std::move(wl)));

    // Expect hw overhead to be valid
    EXPECT_TRUE(cycles > (16 / 4) * (16 / 4) * (64 / 16));
}

TEST_F(TestCostModelVPU2x, SmokeTestDPUVPU27Model_Eltwise) {
    VPUNN::DPUWorkload wl = {
            device27,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    float cycles = static_cast<float>(cost_models.getModel(device27).DPU(std::move(wl)));

    // Expect hw overhead to be valid
    EXPECT_TRUE(cycles > (16 / 4) * (16 / 4) * (64 / 16));
}

// Demonstrate some basic assertions.
TEST_F(TestCostModelVPU2x, SmokeTestDPUVPU27Model) {
    VPUNN::DPUWorkload wl = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    EXPECT_TRUE(cost_models.getModel(device27).nn_initialized());

    auto overhead = cost_models.getModel(device27).DPU(std::move(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(overhead > 1);
    EXPECT_FALSE(Cycles::isErrorCode(overhead)) << overhead;
}

TEST_F(TestCostModelVPU2x, SmokeTestDMA) {
    auto dma_cycles = empty_model.DMA(device27, VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8),
                                VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                                VPUNN::MemoryLocation::CMX);

    auto dma_cycles_model =
            empty_model.DMA(device27, VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8),
                      VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                      VPUNN::MemoryLocation::CMX);

    // Expect equality.
    EXPECT_EQ(dma_cycles, 1242 + static_cast<unsigned int>(std::ceil(56 * 56 * 16 * 1300 / 27000.0f)));
    EXPECT_EQ(dma_cycles, dma_cycles_model);
}

TEST_F(TestCostModelVPU2x, SmokeTestCompressedDMA) {
    // Compressed DMA with 50% CR
    auto dma_cycles = empty_model.DMA(device27, VPUNN::VPUTensor(25088, 1, 1, 1, VPUNN::DataType::UINT8),
                                VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                                VPUNN::MemoryLocation::CMX);

    // Expect equality.
    EXPECT_EQ(dma_cycles, 1242 + static_cast<unsigned int>(std::ceil(25088 * 1300 / 27000.0f)));
}

TEST_F(TestCostModelVPU2x, SmokeTestPermutedDMA) {
    // DMA + Permute
    auto dma_cycles_fp16 = empty_model.DMA(device27,
                                     VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::FLOAT16, VPUNN::Layout::CMAJOR),
                                     VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::FLOAT16, VPUNN::Layout::ZMAJOR),
                                     VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX);

    auto dma_cycles_uint8 = empty_model.DMA(device27,
                                      VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::CMAJOR),
                                      VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8, VPUNN::Layout::ZMAJOR),
                                      VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX);

    // Expect equality.
    unsigned int tensor_size_fp16 = 56 * 56 * 16 * 2;
    unsigned int tensor_size_uint8 = 56 * 56 * 16;
    float fclk_ratio = 1300.0f / 975.0f;
    EXPECT_EQ(dma_cycles_fp16, 1242 + static_cast<unsigned int>(std::ceil(tensor_size_fp16 / 2.0f * fclk_ratio)));
    EXPECT_EQ(dma_cycles_uint8, 1242 + static_cast<unsigned int>(std::ceil(tensor_size_uint8 / 1.0f * fclk_ratio)));
}

TEST_F(TestCostModelVPU2x, Special_Tests_DPU_MAXPOOL_VPU_2_0_1_96_96_9_9_1_2_VALID_FLOAT16_MATRIX) {
    VPUNN::DPUWorkload wl = {
            device20,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(9, 9, 96, 1, VPUNN::DataType::FLOAT16)},  // input dimensions WHCB
            {VPUNN::VPUTensor(9, 9, 96, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::VECTOR_FP16                           // execution mode
    };
    VPUNN::DPUWorkload wl_smaller = {
            device20,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(9, 9 / 5, 96, 1, VPUNN::DataType::FLOAT16)},  // input dimensions WHCB
            {VPUNN::VPUTensor(9, 9 / 5, 96, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::VECTOR_FP16                               // execution mode
    };

    struct TestCycles {
        unsigned int cycles{};
        unsigned int cycles_smaller{};
    };

    {
        TestCycles cycles_slow{cost_models.getModel(device20).DPU(wl), cost_models.getModel(device20).DPU(wl_smaller)};

        VPUNN::VPUCostModel model_fast{the_NN_models.fast_model_paths[0].first};
        TestCycles cycles_fast{model_fast.DPU(wl), model_fast.DPU(wl_smaller)};

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_slow.cycles_smaller))
                << VPUNN::Cycles::toErrorText(cycles_slow.cycles_smaller);
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_slow.cycles)) << VPUNN::Cycles::toErrorText(cycles_slow.cycles);

        EXPECT_LE(cycles_slow.cycles_smaller, cycles_slow.cycles)
                << " Model 2.0 slow \n"
                << wl << wl_smaller << " Other Model Times: \nsmaller: " << cycles_slow.cycles_smaller
                << "\nnormal: " << cycles_slow.cycles << std::endl;

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_fast.cycles_smaller))
                << VPUNN::Cycles::toErrorText(cycles_fast.cycles_smaller);
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_fast.cycles)) << VPUNN::Cycles::toErrorText(cycles_fast.cycles);

        EXPECT_LE(cycles_fast.cycles_smaller, cycles_fast.cycles)
                << " Model 2.0 fast \n"
                << wl << wl_smaller << " Other Model Times: \nsmaller: " << cycles_fast.cycles_smaller
                << "\nnormal: " << cycles_fast.cycles << std::endl;

        // not clear what is the difference versus CI, what WL is there, but we get on CI
        // 680(full)   5992(height /5) for slow model
    }
}

TEST_F(TestCostModelVPU2x, AVEPOOL_equivalence_test_27) {
    const VPUNN::DPUWorkload wl_avgpool_ref = {
            device27,
            VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
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

    };
    const auto equivalent_op{VPUNN::Operation::DW_CONVOLUTION};

    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = cost_models.getModel(device27).DPU(std::move(wl_avgpool));
            auto cycles_equiv = cost_models.getModel(device27).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);

            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }

    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        wl_avgpool.execution_order = VPUNN::ExecutionMode::CUBOID_8x16;
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = cost_models.getModel(device27).DPU(std::move(wl_avgpool));
            auto cycles_equiv = cost_models.getModel(device27).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }

    {
        VPUNN::DPUWorkload wl_avgpool{std::move(wl_avgpool_ref)};
        wl_avgpool.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl_avgpool.output_write_tiles = 2;
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = cost_models.getModel(device27).DPU(std::move(wl_avgpool));
            auto cycles_equiv = cost_models.getModel(device27).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }
}

TEST_F(TestCostModelVPU2x, AVEPOOL_equivalence_test_20) {
    const VPUNN::DPUWorkload wl_avgpool_ref = {
            device20,
            VPUNN::Operation::AVEPOOL,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::VECTOR,                               // execution mode
            VPUNN::ActivationFunction::NONE,                            // activation
            0.0F,                                                       // act_sparsity
            0.0F,                                                       // weight_sparsity
            {swz_def, swz_def},                                         // input_swizzling
            {swz_def},                                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled

    };
    const auto equivalent_op{VPUNN::Operation::DW_CONVOLUTION};

    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = cost_models.getModel(device20).DPU(std::move(wl_avgpool));
            auto cycles_equiv = cost_models.getModel(device20).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }

    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        wl_avgpool.execution_order = VPUNN::ExecutionMode::MATRIX;
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = cost_models.getModel(device20).DPU(std::move(wl_avgpool));
            auto cycles_equiv = cost_models.getModel(device20).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }

    {
        VPUNN::DPUWorkload wl_avgpool{std::move(wl_avgpool_ref)};
        wl_avgpool.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // not supported on 2,.0
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = cost_models.getModel(device20).DPU(std::move(wl_avgpool));
            auto cycles_equiv = cost_models.getModel(device20).DPU(std::move(wl_equiv));

            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }
}

TEST_F(TestCostModelVPU2x, Datatype_Sanity_test_VPU27) {
    const VPUNN::DPUWorkload wl_ref_int = {
            device27,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
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
    const VPUNN::DataType normalized_data_int{VPUNN::DataType::UINT8};

    const VPUNN::DPUWorkload wl_ref_flt = {
            device27,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::BFLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::BFLOAT16)},  // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {1, 1, 1, 1},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.0F,                                                          // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            false,                                                         // weight_sparsity_enabled

    };
    const VPUNN::DataType normalized_data_flt{VPUNN::DataType::FLOAT16};

    {
        {
            VPUNN::DPUWorkload wl{std::move(wl_ref_int)};
            VPUNN::DPUWorkload wl_equiv{wl};
            wl_equiv.inputs[0].change_datatype_superficial(normalized_data_int);
            wl_equiv.outputs[0].change_datatype_superficial(normalized_data_int);

            auto cycles_raw = cost_models.getModel(device27).DPU(std::move(wl));  // will change
            auto cycles_equiv = cost_models.getModel(device27).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_raw));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_raw, cycles_equiv);
            EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
        }

        {
            VPUNN::DPUWorkload wl{std::move(wl_ref_flt)};
            VPUNN::DPUWorkload wl_equiv{wl};
            wl_equiv.inputs[0].change_datatype_superficial(normalized_data_flt);
            wl_equiv.outputs[0].change_datatype_superficial(normalized_data_flt);

            auto cycles_raw = cost_models.getModel(device27).DPU(std::move(wl));  // will change
            auto cycles_equiv = cost_models.getModel(device27).DPU(std::move(wl_equiv));

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_raw));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_raw, cycles_equiv);
            EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
        }
    }
}

// Demonstrate some basic assertions.
TEST_F(TestCostModelVPU2x, InitAspects) {
    {  // 20
        EXPECT_NO_THROW(VPUNN::VPUCostModel x{model20_path});
        VPUNN::VPUCostModel vpunn_model{model20_path};
        ASSERT_TRUE(vpunn_model.nn_initialized());

        const auto file_content{read_a_file(std::move(model20_path))};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(VPUNN::VPUCostModel x(file_content.data(), file_content.size(), true));
        VPUNN::VPUCostModel vpunn_model_buf{file_content.data(), file_content.size(), true};
        EXPECT_TRUE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(VPUNN::VPUCostModel x(file_content.data(), file_content.size(), false));
        VPUNN::VPUCostModel vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_TRUE(vpunn_model_buf_copy.nn_initialized());
    }
    {  // 27
        EXPECT_NO_THROW(VPUNN::VPUCostModel x(model27_path));
        VPUNN::VPUCostModel vpunn_model(model27_path);
        EXPECT_TRUE(vpunn_model.nn_initialized());

        const auto file_content{read_a_file(model27_path)};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(VPUNN::VPUCostModel x(file_content.data(), file_content.size(), true));
        VPUNN::VPUCostModel vpunn_model_buf(file_content.data(), file_content.size(), true);
        EXPECT_TRUE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(VPUNN::VPUCostModel x(file_content.data(), file_content.size(), false));
        VPUNN::VPUCostModel vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_TRUE(vpunn_model_buf_copy.nn_initialized());
    }
    {  // garbage data/no file
        const std::string model_path = "NoFileHere.vpunn";
        EXPECT_NO_THROW(VPUNN::VPUCostModel x(model_path));
        VPUNN::VPUCostModel vpunn_model(model_path);
        EXPECT_FALSE(vpunn_model.nn_initialized());

        const decltype(read_a_file("")) file_content{'M', 'u', 's', 't', 'h', 'a', 'v', 'e', ' ', '0', '1'};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        EXPECT_NO_THROW(VPUNN::VPUCostModel x(file_content.data(), file_content.size(), true));
        VPUNN::VPUCostModel vpunn_model_buf(file_content.data(), file_content.size(), true);
        EXPECT_FALSE(vpunn_model_buf.nn_initialized());

        EXPECT_NO_THROW(VPUNN::VPUCostModel x(file_content.data(), file_content.size(), false));
        VPUNN::VPUCostModel vpunn_model_buf_copy(file_content.data(), file_content.size(), false);
        EXPECT_FALSE(vpunn_model_buf_copy.nn_initialized());

        auto cycles_20 = vpunn_model_buf.DPU(wl_glob_20);
        auto cycles_27 = vpunn_model_buf.DPU(wl_glob_27);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_20));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_27));

        EXPECT_EQ(cycles_20, 27556);  // theoretical value, before fixing padding skip was 28224
        EXPECT_EQ(cycles_27, 3445);   // theoretical values,  before fixing padding skip was 3528
    }
}

TEST_F(TestCostModelVPU2x, ComaparativeRuns) {
    auto modelRun = [](const std::string& model_path, VPUNN::DPUWorkload& wld) {
        VPUNN::VPUCostModel vpunn_model(model_path);
        const IEnergy& my_energy = vpunn_model.getEnergyInterface();
        std::cout << model_path << " : initialized: " << vpunn_model.nn_initialized() << std::endl;

        // std::cout << "run_NN(wl)   : " << vpunn_model.run_NN(wld) << std::endl;
        std::cout << "DPU(wl)   : " << vpunn_model.DPU(wld) << std::endl;
        std::cout << "hw_utilization(wl)   : " << my_energy.hw_utilization(wld) << std::endl;
    };

    std::cout << "----------------------------------------------------------\n";
    modelRun(model20_path, wl_glob_20);
    // modelRun(VPU_2_0_MODEL_PATH, wl_glob_20);
    std::cout << "----------------------------------------------------------\n";
    modelRun(model27_path, wl_glob_27);
    // modelRun(VPU_2_7_MODEL_PATH, wl_glob_27);
    std::cout << "----------------------------------------------------------\n";
    modelRun(NameHelperNN::make_fast_version(model20_path), wl_glob_20);
    // modelRun(NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH), wl_glob_20);
    std::cout << "----------------------------------------------------------\n";
    modelRun(NameHelperNN::make_fast_version(model27_path), wl_glob_27);
    // modelRun(NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH), wl_glob_27);
    std::cout << "----------------------------------------------------------\n";
}

TEST_F(TestCostModelVPU2x, OutputWriteTiles_multiple) {
    const VPUNN::DPUWorkload wl_ref_1x1_f = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                         // kernels
            {1, 1},                                                         // strides
            {0, 0, 0, 0},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                // activation
            0.0F,                                                           // act_sparsity
            0.0F,                                                           // weight_sparsity
            {swz_def, swz_def},                                             // input_swizzling
            {swz_def},                                                      // output_swizzling
            1,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false,                                                          // weight_sparsity_enabled

    };
    const VPUNN::DPUWorkload wl_ref_5x5_f = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(128, 128, 16, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {5, 5},                                                         // kernels
            {1, 1},                                                         // strides
            {2, 2, 2, 2},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                // activation
            0.0F,                                                           // act_sparsity
            0.0F,                                                           // weight_sparsity
            {swz_def, swz_def},                                             // input_swizzling
            {swz_def},                                                      // output_swizzling
            0,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false,                                                          // weight_sparsity_enabled

    };

    VPUNN::DPU_OperationValidator ops;  ///< sanitizer mechanisms
    const VPUNN::IDeviceValidValues& cfg{ops.get_config(device27)};
    auto owt_list = cfg.get_output_write_tile_options();
    std::sort(owt_list.begin(), owt_list.end());

    auto run_test_1owt = [this](const VPUNN::DPUWorkload& wl, const std::string& h) {
        auto cycles = cost_models.getModel(device27).DPU(wl);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles))
                << h << "err_code: " << VPUNN::Cycles::toErrorText(cycles) << std::endl;

        std::cout << h << "\t :output_write_tiles: " << wl.output_write_tiles << ",  cyc: " << cycles << std::endl;
    };

    auto run_test_all_owt = [&run_test_1owt, &owt_list](const VPUNN::DPUWorkload& wlref, const std::string& h) {
        VPUNN::DPUWorkload wl{wlref};
        for (const auto& owt : owt_list) {
            wl.output_write_tiles = owt;
            run_test_1owt(wl, h);
        }
        std::cout << std::endl;
    };
    {
        bool expectationToForce{true};
        {
            VPUNN::DPUWorkload wl{std::move(wl_ref_1x1_f)};
            EXPECT_TRUE(expectationToForce) << wl;

            wl.isi_strategy = VPUNN::ISIStrategy::CLUSTERING;
            run_test_all_owt(wl, "Clustering: ");

            wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
            run_test_all_owt(wl, "SOH: ");

            wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
            // run_test_all_owt(wl, "SOK: ");// not possible owt =1
        }
        {
            VPUNN::DPUWorkload wl{std::move(wl_ref_5x5_f)};
            EXPECT_TRUE(expectationToForce) << wl;

            wl.isi_strategy = VPUNN::ISIStrategy::CLUSTERING;
            run_test_all_owt(wl, "Clustering: ");

            wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
            run_test_all_owt(wl, "SOH: ");

            wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
            // run_test_all_owt(wl, "SOK: ");// not possible owt =1
        }
    }
}

TEST_F(TestCostModelVPU2x, SmokeTests_DPUInfo_20) {
    const VPUNN::DPUWorkload wl_special = {
            device20,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(9, 10, 2048, 1, VPUNN::DataType::FLOAT16, Layout::ZMAJOR)},  // input dimensions
            {VPUNN::VPUTensor(9, 10, 256, 1, VPUNN::DataType::UINT8, Layout::ZMAJOR)},     // output dimensions
            {1, 3},                                                                        // kernels
            {1, 1},                                                                        // strides
            {1, 1, 0, 0},                                                                  // padding
            VPUNN::ExecutionMode::MATRIX,                                                  // execution mode
            VPUNN::ActivationFunction::NONE,                                               // activation
            0.0F,                                                                          // act_sparsity
            0.99739581346511841F,                                                          // weight_sparsity
            {swz_def, swz_def},                                                            // input_swizzling
            {swz_def},                                                                     // output_swizzling
            1,                                                                             // output_write_tiles
            {0, 0, 0, 0},                                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                                // isi_strategy
            true,                                                                          // weight_sparsity_enabled

    };
    {  // 20
        const DPUWorkload wl{std::move(wl_special)};

        EXPECT_TRUE(cost_models.getModel(device20).nn_initialized());
        auto cycles_dpu = cost_models.getModel(device20).DPU(wl);

        DPUInfoPack cycles_Pack;
        ASSERT_NO_THROW(cycles_Pack = cost_models.getModel(device20).DPUInfo(wl)) << wl;

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_dpu));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_Pack.DPUCycles));
        EXPECT_EQ(cycles_dpu, cycles_Pack.DPUCycles) << wl;
    }
}

TEST_F(TestCostModelVPU2x, TestDPUVPU27ModelIC_4_16_32) {
    VPUNN::DPUWorkload wl0_prototype = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::DPUWorkload wl1_prototype = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::DPUWorkload wl2_prototype = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    VPUNN::DPUWorkload wl0{std::move(wl0_prototype)};
    VPUNN::DPUWorkload wl1{std::move(wl1_prototype)};
    VPUNN::DPUWorkload wl2{std::move(wl2_prototype)};

    ASSERT_TRUE(cost_models.getModel(device27).nn_initialized())
            << "Model not loaded, might be due to file location: " << model27_path << std::endl;

    // wl0 < wl1 && wl0 < wl2

    {
        CyclesInterfaceType wl0_cycles = cost_models.getModel(device27).DPU(wl0);
        EXPECT_FALSE(Cycles::isErrorCode(wl0_cycles));
        CyclesInterfaceType wl1_cycles = cost_models.getModel(device27).DPU(wl1);
        EXPECT_FALSE(Cycles::isErrorCode(wl1_cycles));
        CyclesInterfaceType wl2_cycles = cost_models.getModel(device27).DPU(wl2);
        EXPECT_FALSE(Cycles::isErrorCode(wl2_cycles));

        EXPECT_LT(wl0_cycles, wl1_cycles);
        EXPECT_LT(wl0_cycles, wl2_cycles);

        std::cout << "On Execution Mode Cuboid 16x16\nIC = 4:" << wl0_cycles << "\nIC = 16:" << wl1_cycles
                  << "\nIC = 32:" << wl2_cycles;
    }

    wl0.execution_order = wl1.execution_order = wl2.execution_order = VPUNN::ExecutionMode::CUBOID_8x16;

    {
        CyclesInterfaceType wl0_cycles = cost_models.getModel(device27).DPU(wl0);
        EXPECT_FALSE(Cycles::isErrorCode(wl0_cycles));
        CyclesInterfaceType wl1_cycles = cost_models.getModel(device27).DPU(wl1);
        EXPECT_FALSE(Cycles::isErrorCode(wl1_cycles));
        CyclesInterfaceType wl2_cycles = cost_models.getModel(device27).DPU(wl2);
        EXPECT_FALSE(Cycles::isErrorCode(wl2_cycles));

        EXPECT_LT(wl0_cycles, wl1_cycles);
        EXPECT_LT(wl0_cycles, wl2_cycles);

        std::cout << "On Execution Mode Cuboid 8x16\nIC = 4:" << wl0_cycles << "\nIC = 16:" << wl1_cycles
                  << "\nIC = 32:" << wl2_cycles;
    }
    wl0.execution_order = wl1.execution_order = wl2.execution_order = VPUNN::ExecutionMode::CUBOID_4x16;

    {
        CyclesInterfaceType wl0_cycles = cost_models.getModel(device27).DPU(std::move(wl0));
        EXPECT_FALSE(Cycles::isErrorCode(wl0_cycles));
        CyclesInterfaceType wl1_cycles = cost_models.getModel(device27).DPU(std::move(wl1));
        EXPECT_FALSE(Cycles::isErrorCode(wl1_cycles));
        CyclesInterfaceType wl2_cycles = cost_models.getModel(device27).DPU(std::move(wl2));
        EXPECT_FALSE(Cycles::isErrorCode(wl2_cycles));

        EXPECT_LT(wl0_cycles, wl1_cycles);
        EXPECT_LT(wl0_cycles, wl2_cycles);

        std::cout << "On Execution Mode Cuboid 4x16\nIC = 4:" << wl0_cycles << "\nIC = 16:" << wl1_cycles
                  << "\nIC = 32:" << wl2_cycles;
    }
}

///@todo: continue from here
TEST_F(TestCostModelVPU2x, Compressed_CONV_Sanity_test_VPU27) {
    const VPUNN::DPUWorkload wl_ref = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 4, 1, VPUNN::DataType::INT8)},   // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
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
    const VPUNN::DPUWorkload wl_ref_equiv = {
            device27,
            VPUNN::Operation::CM_CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 4, 1, VPUNN::DataType::INT8)},   // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
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

    {
        VPUNN::DPUWorkload wl{std::move(wl_ref)};
        VPUNN::DPUWorkload wl_equiv{std::move(wl_ref_equiv)};

        std::string info_raw, info_equiv;

        auto cycles_raw = cost_models.getModel(device27).DPU(wl, info_raw);  // will change
        auto cycles_equiv = cost_models.getModel(device27).DPU(wl_equiv, info_equiv);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_raw))
                << "ERROR code received: " << cycles_raw << " : " << Cycles::toErrorText(cycles_raw)
                << "\n INFO: " << wl << info_raw << std::endl;
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv))
                << "ERROR code received: " << cycles_equiv << " : " << Cycles::toErrorText(cycles_equiv)
                << "\n INFO: " << wl_equiv << info_equiv << std::endl;
        ;

        EXPECT_EQ(cycles_raw, cycles_equiv);
        EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
    }
}

/// Expecting that also CompressCOnv for IC=1 is accepted. This Is not obvious from the beginning
TEST_F(TestCostModelVPU2x, Compressed_CONV_Sanity_test_VPU27_IC1_special) {
    const VPUNN::DPUWorkload wl_ref = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 1, 1, VPUNN::DataType::INT8)},   // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
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
    const VPUNN::DPUWorkload wl_ref_equiv = {
            device27,
            VPUNN::Operation::CM_CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 1, 1, VPUNN::DataType::INT8)},   // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
            VPUNN::ActivationFunction::NONE,                           // activation
            0.0F,                                                      // act_sparsity
            0.0F,                                                      // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzlingg
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled

    };


    {
        VPUNN::DPUWorkload wl{std::move(wl_ref)};
        VPUNN::DPUWorkload wl_equiv{std::move(wl_ref_equiv)};

        std::string info_raw, info_equiv;

        auto cycles_raw = cost_models.getModel(device27).DPU(wl, info_raw);  // will change
        auto cycles_equiv = cost_models.getModel(device27).DPU(wl_equiv, info_equiv);

        EXPECT_FALSE(Cycles::isErrorCode(cycles_raw))
                << "ERROR code received: " << cycles_raw << " : " << Cycles::toErrorText(cycles_raw)
                << "\n INFO: " << wl << info_raw << std::endl;
        EXPECT_FALSE(Cycles::isErrorCode(cycles_equiv))
                << "ERROR code received: " << cycles_equiv << " : " << Cycles::toErrorText(cycles_equiv)
                << "\n INFO: " << wl_equiv << info_equiv << std::endl;
        ;

        EXPECT_EQ(cycles_raw, cycles_equiv) << wl << wl_equiv << std::endl;
        EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
    }
}

TEST_F(TestCostModelVPU2x, Compressed_CONV_Sanity_test_VPU27_sparse) {
    const VPUNN::DPUWorkload wl_ref = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 4, 1, VPUNN::DataType::INT8)},   // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
            VPUNN::ActivationFunction::NONE,                           // activation
            0.0F,                                                      // act_sparsity
            0.44F,                                                     // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            true,                                                      // weight_sparsity_enabled

    };
    const VPUNN::DPUWorkload wl_ref_equiv = {
            device27,
            VPUNN::Operation::CM_CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 4, 1, VPUNN::DataType::INT8)},   // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
            {3, 3},                                                    // kernels
            {1, 1},                                                    // strides
            {1, 1, 1, 1},                                              // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
            VPUNN::ActivationFunction::NONE,                           // activation
            0.0F,                                                      // act_sparsity
            0.44F,                                                     // weight_sparsity
            {swz_def, swz_def},                                        // input_swizzling
            {swz_def},                                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            true,                                                      // weight_sparsity_enabled

    };

    {
        VPUNN::DPUWorkload wl{std::move(wl_ref)};
        VPUNN::DPUWorkload wl_equiv{std::move(wl_ref_equiv)};

        std::string info_raw, info_equiv;

        auto cycles_raw = cost_models.getModel(device27).DPU(wl, info_raw);  // will change
        auto cycles_equiv = cost_models.getModel(device27).DPU(wl_equiv, info_equiv);

        EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_raw))
                << "ERROR code received: " << cycles_raw << " : " << Cycles::toErrorText(cycles_raw)
                << "\n INFO: " << wl << info_raw << std::endl;
        EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_equiv))
                << "ERROR code received: " << cycles_equiv << " : " << Cycles::toErrorText(cycles_equiv)
                << "\n INFO: " << wl_equiv << info_equiv << std::endl;
        ;

        EXPECT_EQ(cycles_raw, V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION));
        EXPECT_EQ(cycles_raw, cycles_equiv);

        EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
    }
}

// Compressed Conv experiment to establish ratio  between Conv16 and CM_conv 4 input ch
TEST_F(TestCostModelVPU2x, DISABLED_Compressed_CONV_Sweep_log_NPU27_EISXW_103713) {
    // const VPUNN::DPUWorkload wl_ref = {
    //         device27,
    //         VPUNN::Operation::CONVOLUTION,
    //         {VPUNN::VPUTensor(16, 16, 4, 1, VPUNN::DataType::INT8)},   // input dimensions
    //         {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
    //         {3, 3},                                                    // kernels
    //         {1, 1},                                                    // strides
    //         {1, 1, 1, 1},                                              // padding
    //         VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
    //         VPUNN::ActivationFunction::NONE,                           // activation
    //         0.0F,                                                      // act_sparsity
    //         0.0F,                                                      // weight_sparsity
    //{swz_def, swz_def},  // input_swizzling
    //        {swz_def},   // output_swizzling
    //         1,                                                         // output_write_tiles
    //         {0, 0, 0, 0},                                              // offsets
    //         VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
    //         false,                                                     // weight_sparsity_enabled

    //};
    // const VPUNN::DPUWorkload wl_ref_equiv = {
    //        device27,
    //        VPUNN::Operation::CM_CONVOLUTION,
    //        {VPUNN::VPUTensor(16, 16, 4, 1, VPUNN::DataType::INT8)},   // input dimensions
    //        {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::INT8)},  // output dimensions
    //        {3, 3},                                                    // kernels
    //        {1, 1},                                                    // strides
    //        {1, 1, 1, 1},                                              // padding
    //        VPUNN::ExecutionMode::CUBOID_16x16,                        // execution mode
    //        VPUNN::ActivationFunction::NONE,                           // activation
    //        0.0F,                                                      // act_sparsity
    //        0.0F,                                                      // weight_sparsity
    //                    {swz_def, swz_def},                                            // input_swizzling
    //{swz_def},  // output_swizzling
    //        1,                                                         // output_write_tiles
    //        {0, 0, 0, 0},                                              // offsets
    //        VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
    //        false,                                                     // weight_sparsity_enabled
    //};

    auto costructCompressConv50x50 = [this](const unsigned int InpC, const unsigned int OutC) {
        const VPUNN::DPUWorkload wl{
                device27,
                VPUNN::Operation::CM_CONVOLUTION,
                {VPUNN::VPUTensor(50, 50, InpC, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
                {VPUNN::VPUTensor(50, 50, OutC, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
                {5, 5},                                                         // kernels
                {1, 1},                                                         // strides
                {2, 2, 2, 2},                                                   // padding
                VPUNN::ExecutionMode::CUBOID_16x16,                             // execution mode
                VPUNN::ActivationFunction::NONE,                                // activation
                0.0F,                                                           // act_sparsity
                0.0F,                                                           // weight_sparsity
                {swz_def, swz_def},                                             // input_swizzling
                {swz_def},                                                      // output_swizzling
                1,                                                              // output_write_tiles
                {0, 0, 0, 0},                                                   // offsets
                VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
                false,                                                          // weight_sparsity_enabled
        };
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return wl;
    };

    auto costructCompressConv321x46 = [this](const unsigned int InpC, const unsigned int OutC) {
        const VPUNN::DPUWorkload wl{
                device27,
                VPUNN::Operation::CM_CONVOLUTION,
                {VPUNN::VPUTensor(321, 46, InpC, 1, VPUNN::DataType::INT8)},  // input dimensions
                {VPUNN::VPUTensor(320, 45, OutC, 1, VPUNN::DataType::INT8)},  // output dimensions
                {3, 3},                                                       // kernels
                {1, 1},                                                       // strides
                {0, 1, 0, 1},                                                 // padding
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
        };
        // clang and gcc does not support to use std::move here, so we need suppression
        /* coverity[copy_instead_of_move] */
        return wl;
    };

    // auto costructCompressConv321x46_CONV = [](const unsigned int InpC, const unsigned int OutC) {
    //     const VPUNN::DPUWorkload wl{
    //             device27,
    //             VPUNN::Operation::CONVOLUTION,
    //             {VPUNN::VPUTensor(321, 46, InpC, 1, VPUNN::DataType::INT8)},  // input dimensions
    //             {VPUNN::VPUTensor(320, 45, OutC, 1, VPUNN::DataType::INT8)},  // output dimensions
    //             {3, 3},                                                       // kernels
    //             {1, 1},                                                       // strides
    //             {0, 1, 0, 1},                                                 // padding
    //             VPUNN::ExecutionMode::CUBOID_16x16,                           // execution mode
    //             VPUNN::ActivationFunction::NONE,                              // activation
    //             0.0F,                                                         // act_sparsity
    //             0.0F,                                                         // weight_sparsity
    //             {swz_def, swz_def},                                           // input_swizzling
    //             {swz_def},                                                    // output_swizzling
    //             1,                                                            // output_write_tiles
    //             {0, 0, 0, 0},                                                 // offsets
    //             VPUNN::ISIStrategy::CLUSTERING,                               // isi_strategy
    //             false,                                                        // weight_sparsity_enabled
    //     };
    //     return wl;
    // };

    struct TestIn {
        DPUWorkload wl;
        std::string test_name;
    };
    struct TestCase {
        TestIn t_in;
    };

    std::vector<TestCase> tests;
    std::vector<unsigned int> oc_s{16, 32, 48, 64, 80, 96, 112, 128, 256};
    for (auto oc : oc_s) {
        // const unsigned int oc{32};

        std::stringstream buffer;
        for (unsigned int i = 1; i < 16; ++i) {  // generate simple tests
            buffer.str("");
            buffer << "CompressConv 50x50 , IC=" << i << ", OC= " << oc;
            const std::string testName = buffer.str();
            TestCase t{{costructCompressConv50x50(i, oc), std::move(testName)}};
            tests.push_back(std::move(t));
        }

        {  // CONV
            int i = 16;
            buffer.str("");
            buffer << "CONV 50x50 , IC=" << i << ", OC= " << oc;
            const std::string testName = buffer.str();
            auto w{costructCompressConv50x50(i, oc)};
            w.op = Operation::CONVOLUTION;
            TestCase t{{std::move(w), std::move(testName)}};
            tests.push_back(std::move(t));
        }

        if (false) {                                 // another larger workload
            for (unsigned int i = 1; i < 16; ++i) {  // generate sweep from tehreal example
                buffer.str("");
                buffer << "CompressConv original , IC=" << i << ", OC= " << oc;
                const std::string testName = buffer.str();
                TestCase t{{costructCompressConv321x46(i, oc), std::move(testName)}};
                tests.push_back(std::move(t));
            }
            {  // conv
                int i = 16;
                buffer.str("");
                buffer << "Conv original , IC=" << i << ", OC= " << oc;
                const std::string testName = buffer.str();
                auto w{costructCompressConv321x46(i, oc)};
                w.op = Operation::CONVOLUTION;
                TestCase t{{std::move(w), std::move(testName)}};
                tests.push_back(std::move(t));
            }
        }

        // for (unsigned int i = 1; i <= 16; ++i) {  // generate sweep from thereal example

        //    buffer.str("");
        //    buffer << "CompressConv original CONV , IC=" << i << ", OC= " << oc;
        //    const std::string testName = buffer.str();
        //    TestCase t{{costructCompressConv321x46_CONV(i, oc), testName}};
        //    tests.push_back(t);
        //}
    }

    EXPECT_EQ(1, 0);

    {
        for (const auto& test : tests) {
            DPUWorkload wl{test.t_in.wl};

            std::string info_raw;

            auto cycles_raw = cost_models.getModel(device27).DPU(wl, info_raw);

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_raw))
                    << "ERROR code received: " << cycles_raw << " : " << Cycles::toErrorText(cycles_raw)
                    << "\n TEST: " << test.t_in.test_name << "\n INFO: " << wl << info_raw << std::endl;

            std::cout << "\n TEST: " << test.t_in.test_name << " :\t Value : " << cycles_raw;
        }
    }
}

TEST_F(TestCostModelVPU2x, Check_Wl_halo_data_test) {
    const VPUNN::DPUWorkload wl_ref{
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(18, 18, 64, 1, VPUNN::DataType::UINT8)},       // input dimensions
            {VPUNN::VPUTensor(16, 18, 48, 1, VPUNN::DataType::UINT8)},       // output dimensions
            {3, 3},                                                          // kernels
            {1, 1},                                                          // strides
            {1, 1, 0, 0},                                                    // padding
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
        CyclesInterfaceType cycles;
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;

    const HaloWorkload::HaloInfoHWC input_halo{0, 0, 0, 0, 0, 0};
    const HaloWorkload::HaloInfoHWC output_halo{0, 0, 0, 0, 0, 0};

    // static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};
    static constexpr unsigned int ERROR_EXPECTED{VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION};

    EXPECT_TRUE(cost_models.getModel(device27).nn_initialized());

    std::string info = "";
    CyclesInterfaceType cyc = cost_models.getModel(device27).DPU(wl_ref, info);
    EXPECT_FALSE(is_error_code(cyc)) << cyc << "\n";

    auto verify_wl_halo = [this](const TestsVector& tests) {
        int i = 1;  // index of test cases
        std::string info;

        for (const auto& t : tests) {
            std::cout << "TEST " << i << ": " << t.test_case << "\n";

            VPUNN::DPUWorkload wl_ref_halo{t.t_in.wl};
            wl_ref_halo.halo = t.t_in.halo_ref;

            const DPUOperation dpu{wl_ref_halo};

            CyclesInterfaceType cyc = cost_models.getModel(device27).DPU(std::move(wl_ref_halo), info);

            if (!is_error_code(t.t_exp.cycles)) {
                ASSERT_FALSE(is_error_code(cyc));
            } else {
                ASSERT_TRUE(is_error_code(cyc));
                ASSERT_EQ(cyc, t.t_exp.cycles);
            }

            i++;
        }
    };

    const TestsVector tests = {
            {{wl_ref, {{0, 0, 0, 0, 0, 0}, output_halo, output_halo, output_halo}},
             {NO_ERROR_EXPECTED},
             "No halo input"},
            {{wl_ref, {{0, 1, 0, 0, 0, 0}, output_halo, output_halo, output_halo}},
             {ERROR_EXPECTED},
             "Input halo bottom, but wl have TB padding "},
            {{wl_ref, {{0, 0, 5, 0, 0, 0}, output_halo, output_halo, output_halo}},
             {NO_ERROR_EXPECTED},
             "Input halo left"},
            {{wl_ref, {{0, 0, 0, 3, 0, 0}, output_halo, output_halo, output_halo}},
             {NO_ERROR_EXPECTED},
             "Input halo right"},
            {{wl_ref, {{1, 0, 0, 1, 0, 0}, output_halo, output_halo, output_halo}},
             {ERROR_EXPECTED},
             "Input halo top and right , but wl have TB padding"},
            {{wl_ref, {{1, 2, 3, 4, 0, 0}, output_halo, output_halo, output_halo}},
             {ERROR_EXPECTED},
             "All input halo, but wl have TB padding "},
            {{wl_ref, {{0, 0, 0, 0, 0, 0}, output_halo, output_halo, output_halo}}, {NO_ERROR_EXPECTED}, "No halo"},
            {{wl_ref, {input_halo, {0, -2, 0, 0}, output_halo, output_halo}}, {ERROR_EXPECTED}, "Btm halo output"},
            {{wl_ref, {input_halo, output_halo, {0, 0, -9, 0}, output_halo}}, {ERROR_EXPECTED}, "Left halo broadcast "},
            {{wl_ref, {input_halo, output_halo, {-4, 0, 0, 0}, output_halo}},
             {ERROR_EXPECTED},
             "Negative top halo broadcast"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, -1, 0}}},
             {ERROR_EXPECTED},
             "Negative left halo inbound"},
            {{wl_ref, {input_halo, output_halo, output_halo, {0, 0, 0, -2}}},
             {ERROR_EXPECTED},
             "Negative right halo inbound"},
            {{wl_ref, {input_halo, output_halo, output_halo, {-3, 0, 0, 0}}},
             {ERROR_EXPECTED},
             "Negative top halo inbound"},
            {{wl_ref, {input_halo, {0, 0, 0, -7}, output_halo, output_halo}},
             {ERROR_EXPECTED},
             "Negative right halo output"}};

    verify_wl_halo(tests);
}

// tests that sparsity is allowed for output
TEST_F(TestCostModelVPU2x, Output_sparsity_enabled_validation_test_103894) {
    const DPUWorkload wl_ref_cnv = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUTensor(7, 7, 512, 1, DataType::INT8, Layout::ZXY, true)},  // input dimensions
            {VPUTensor(7, 7, 128, 1, DataType::INT8, Layout::ZXY, true)},  // output dimensions
            {3, 3},                                                        // kernels
            {1, 1},                                                        // strides
            {1, 1, 1, 1},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                             // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.1F,                                                          // act_sparsity
            0.627848F,                                                     // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            true,                                                          // weight_sparsity_enabled

    };

    const DPUWorkload wl_ref_avpool = {
            device27,
            VPUNN::Operation::AVEPOOL,
            {VPUTensor(7, 7, 64, 1, DataType::INT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(1, 1, 64, 1, DataType::INT8, Layout::ZXY, true)},   // output dimensions
            {7, 7},                                                        // kernels
            {1, 1},                                                        // strides
            {0, 0, 0, 0},                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                            // execution mode
            VPUNN::ActivationFunction::NONE,                               // activation
            0.0F,                                                          // act_sparsity
            0.F,                                                           // weight_sparsity
            {swz_def, swz_def},                                            // input_swizzling
            {swz_def},                                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            ISIStrategy::CLUSTERING,                                       // isi_strategy
            false,                                                         // weight_sparsity_enabled
    };
    const DPUWorkload wl_ref_elm = {
            device27,
            VPUNN::Operation::ELTWISE,
            {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(56, 56, 64, 1, DataType::INT8, Layout::ZXY, true)},   // output dimensions
            {1, 1},                                                          // kernels
            {1, 1},                                                          // strides
            {0, 0, 0, 0},                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                 // activation
            0.0F,                                                            // act_sparsity
            0.F,                                                             // weight_sparsity
            {swz_def, swz_def},                                              // input_swizzling
            {swz_def},                                                       // output_swizzling
            1,                                                               // output_write_tiles
            {0, 0, 0, 0},                                                    // offsets
            ISIStrategy::CLUSTERING,                                         // isi_strategy
            false,                                                           // weight_sparsity_enabled
    };

    {
        // sparsity (input) is allowed only for CONV and ELEMENTwise, prohibited for rest
        // output sparsity should be enabled for all// not influencing  infered runtime

        {  // conv sparse output
            DPUWorkload wl{wl_ref_cnv};
            std::string info;
            auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

            EXPECT_FALSE(Cycles::isErrorCode(cycles))
                    << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles) << "\n INFO: " << wl
                    << info << std::endl;

            EXPECT_TRUE((cycles != 0));

            DPUWorkload wl_nonsparseOut{wl_ref_cnv};
            wl_nonsparseOut.outputs[0].set_sparsity(false);

            wl = wl_nonsparseOut;
            auto cycles_ns = cost_models.getModel(device27).DPU(wl, info);  // will change

            EXPECT_FALSE(Cycles::isErrorCode(cycles_ns))
                    << "ERROR code received: " << cycles_ns << " : " << Cycles::toErrorText(cycles_ns)
                    << "\n INFO: " << wl << info << std::endl;

            EXPECT_TRUE((cycles_ns != 0));

            EXPECT_EQ(cycles, cycles_ns) << wl_ref_cnv << wl_nonsparseOut;
        }

        {  // avgpool/DW_Conv sparse output ()
            DPUWorkload wl{std::move(wl_ref_avpool)};
            std::string info;
            auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

            EXPECT_FALSE(Cycles::isErrorCode(cycles))
                    << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles) << "\n INFO: " << wl
                    << info << std::endl;

            EXPECT_TRUE((cycles != 0));
        }
        {  // elementwise Conv sparse output
            DPUWorkload wl{std::move(wl_ref_elm)};
            std::string info;
            auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

            EXPECT_FALSE(Cycles::isErrorCode(cycles))
                    << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles) << "\n INFO: " << wl
                    << info << std::endl;

            EXPECT_TRUE((cycles != 0));
        }
    }
}

// tests SOK/CLUSTERING + OWT 1/2 equivalence
// focus on elementwise
TEST_F(TestCostModelVPU2x, SOK_CLUSTERING_OWT_equivalence_test_103266) {
    const DPUWorkload wl_ref_cnv_dense = {
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUTensor(7, 7, 512, 1, DataType::INT8, Layout::ZXY, false)},  // input dimensions
            {VPUTensor(7, 7, 128, 1, DataType::INT8, Layout::ZXY, false)},  // output dimensions
            {3, 3},                                                         // kernels
            {1, 1},                                                         // strides
            {1, 1, 1, 1},                                                   // padding
            VPUNN::ExecutionMode::CUBOID_4x16,                              // execution mode
            VPUNN::ActivationFunction::NONE,                                // activation
            0.0F,                                                           // act_sparsity
            0.0F,                                                           // weight_sparsity
            {swz_def, swz_def},                                             // input_swizzling
            {swz_def},                                                      // output_swizzling
            1,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false,                                                          // weight_sparsity_enabled
    };

    const DPUWorkload wl_ref_elm = {
            device27,
            VPUNN::Operation::ELTWISE,
            {VPUTensor(56, 56, 64, 1, DataType::INT8)},  // input dimensions
            {VPUTensor(56, 56, 64, 1, DataType::INT8)},  // output dimensions
            {1, 1},                                      // kernels
            {1, 1},                                      // strides
            {0, 0, 0, 0},                                // padding
            VPUNN::ExecutionMode::CUBOID_16x16,          // execution mode
            VPUNN::ActivationFunction::NONE,             // activation
            0.0F,                                        // act_sparsity
            0.F,                                         // weight_sparsity
            {swz_def, swz_def},                          // input_swizzling
            {swz_def},                                   // output_swizzling
            1,                                           // output_write_tiles
            {0, 0, 0, 0},                                // offsets
            ISIStrategy::CLUSTERING,                     // isi_strategy
            false,                                       // weight_sparsity_enabled
    };
    const DPUWorkload wl_ref_elm_orig = {
            // has Swizz combination 5,5,0
            device27,
            VPUNN::Operation::ELTWISE,
            {VPUTensor(14, 7, 1024, 1, DataType::INT8)},  // input dimensions
            {VPUTensor(14, 7, 1024, 1, DataType::INT8)},  // output dimensions
            {1, 1},                                       // kernels
            {1, 1},                                       // strides
            {0, 0, 0, 0},                                 // padding
            VPUNN::ExecutionMode::CUBOID_8x16,            // execution mode
            VPUNN::ActivationFunction::NONE,              // activation
            0.0F,                                         // act_sparsity
            0.F,                                          // weight_sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},         // input_swizzling
            {Swizzling::KEY_0},                           // output_swizzling
            2,                                            // output_write_tiles
            {0, 0, 0, 0},                                 // offsets
            ISIStrategy::SPLIT_OVER_K,                    // isi_strategy
            false,                                        // weight_sparsity_enabled
    };
    // EXPECT_FALSE(true);//activate this to see logs
    {
        // sparsity (input) is allowed only for CONV and ELEMENTwise, prohibited for rest
        // output sparsity should be enabled for all// not influencing  infered runtime

        {  // conv dense output
            DPUWorkload wl_clu{wl_ref_cnv_dense};
            wl_clu.isi_strategy = ISIStrategy::CLUSTERING;
            DPUWorkload wl_sok{std::move(wl_ref_cnv_dense)};
            wl_sok.isi_strategy = ISIStrategy::SPLIT_OVER_K;
            wl_sok.output_write_tiles = 2;

            CyclesInterfaceType c_clu_1{0};
            CyclesInterfaceType c_clu_2{0};
            CyclesInterfaceType c_sok_2{0};

            wl_clu.output_write_tiles = 1;
            {
                auto wl{wl_clu};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_clu_1 = cycles;
            }
            wl_clu.output_write_tiles = 2;
            {
                auto wl{wl_clu};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_clu_2 = cycles;
            }

            wl_clu.output_write_tiles = 1;
            {
                auto wl{std::move(wl_sok)};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_sok_2 = cycles;
            }

            std::cout << "\n--------- RESULTS conv : "
                      << "\n cost clustering 1: " << c_clu_1 << "\n cost clustering 2: " << c_clu_2
                      << "\n cost sok_2: " << c_sok_2 << " \n ";

            EXPECT_GT(c_clu_2, c_clu_1) << "broadcasting must be greater " << wl_clu << std::endl;
            EXPECT_GT(c_sok_2, c_clu_1) << "broadcasting must be greater " << wl_clu << std::endl;
            int owt2_delta = abs((int)c_sok_2 - (int)c_clu_2);
            int owt1_2_delta = abs((int)c_clu_2 - (int)c_clu_1);
            EXPECT_LE(owt2_delta, owt1_2_delta) << "broadcasting must be grouped together " << wl_clu << std::endl;

        }  // conv

        {  // elementwise bigger
            DPUWorkload wl_clu{wl_ref_elm};
            wl_clu.isi_strategy = ISIStrategy::CLUSTERING;
            DPUWorkload wl_sok{std::move(wl_ref_elm)};
            wl_sok.isi_strategy = ISIStrategy::SPLIT_OVER_K;  // nopt allolwed, will become CLU +OWT1
            wl_sok.output_write_tiles = 2;

            CyclesInterfaceType c_clu_1{0};
            CyclesInterfaceType c_clu_2{0};
            CyclesInterfaceType c_sok_2{0};

            wl_clu.output_write_tiles = 1;
            {
                auto wl{wl_clu};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_clu_1 = cycles;
            }
            wl_clu.output_write_tiles = 2;
            {
                auto wl{wl_clu};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_clu_2 = cycles;
            }

            wl_clu.output_write_tiles = 1;
            {
                auto wl{std::move(wl_sok)};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_sok_2 = cycles;
            }

            std::cout << "\n--------- RESULTS elementwise : "
                      << "\n cost clustering 1: " << c_clu_1 << "\n cost clustering 2: " << c_clu_2
                      << "\n cost sok_2: " << c_sok_2 << " \n ";

            EXPECT_GE(c_clu_2, c_clu_1) << "broadcasting must be greater " << wl_clu << std::endl;
            EXPECT_GE(c_sok_2, c_clu_1) << "broadcasting must be greater " << wl_clu << std::endl;
            int owt2_delta = abs((int)c_sok_2 - (int)c_clu_2);
            int owt1_2_delta = abs((int)c_clu_2 - (int)c_clu_1);
            EXPECT_LE(owt2_delta, owt1_2_delta) << "broadcasting must be grouped together " << wl_clu << std::endl;
        }  // elemntviswe

        {  // elementwise orig
            DPUWorkload wl_clu{wl_ref_elm_orig};
            wl_clu.isi_strategy = ISIStrategy::CLUSTERING;
            DPUWorkload wl_sok{std::move(wl_ref_elm_orig)};
            wl_sok.isi_strategy = ISIStrategy::SPLIT_OVER_K;  // not allowed will become CLU+owt1
            wl_sok.output_write_tiles = 2;

            CyclesInterfaceType c_clu_1{0};
            CyclesInterfaceType c_clu_2{0};
            CyclesInterfaceType c_sok_2{0};

            wl_clu.output_write_tiles = 1;
            {
                auto wl{wl_clu};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_clu_1 = cycles;
            }
            wl_clu.output_write_tiles = 2;
            {
                auto wl{wl_clu};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_clu_2 = cycles;
            }

            wl_clu.output_write_tiles = 1;
            {
                auto wl{std::move(wl_sok)};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_FALSE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles != 0));
                c_sok_2 = cycles;
            }

            std::cout << "\n--------- RESULTS elementwise orig : "
                      << "\n cost clustering 1: " << c_clu_1 << "\n cost clustering 2: " << c_clu_2
                      << "\n cost sok_2: " << c_sok_2 << " \n ";

            EXPECT_GE(c_clu_2, c_clu_1) << "broadcasting must be greater " << wl_clu << std::endl;
            EXPECT_GE(c_sok_2, c_clu_1) << "broadcasting must be greater " << wl_clu << std::endl;
            int owt2_delta = abs((int)c_sok_2 - (int)c_clu_2);
            int owt1_2_delta = abs((int)c_clu_2 - (int)c_clu_1);
            EXPECT_LE(owt2_delta, owt1_2_delta) << "broadcasting must be grouped together " << wl_clu << std::endl;
        }  // elemntviswe orig
    }
}

TEST_F(TestCostModelVPU2x, Dual_sparsity_NN_Output_Cycle_valid_values_Test) {
    const HaloWorkload zeroHalo;
    const SEPModeInfo sepInfo{
            true,             // sep activators using Storage elements table with pointers
            {18, 18, 1, 1},   // SEP pointer table, 32 bits pointers assumed
            {512, 6, 70, 1},  // actual tensor shape for activators
            false             // no_sparse_map if true the sparse map is ignored/non existent
    };

    const VPUNN::DPUWorkload wl_ref_dualsparsity{
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(18, 18, 64, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},  // input dimensions
            {VPUNN::VPUTensor(16, 18, 48, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},  // output dimensions
            {3, 3},                                                                        // kernels
            {1, 1},                                                                        // strides
            {1, 1, 0, 0},                                                                  // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                            // execution mode
            VPUNN::ActivationFunction::NONE,                                               // activation
            0.7F,                                                                          // act_sparsity
            0.3F,                                                                          // weight_sparsity
            {swz_def, swz_def},                                                            // input_swizzling
            {swz_def},                                                                     // output_swizzling
            1,                                                                             // output_write_tiles
            {0, 0, 0, 0},                                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                                // isi_strategy
            true,                                                                          // weight_sparsity_enabled
            zeroHalo                                                                       // halo
    };

    // wl when active SEP
    VPUNN::DPUWorkload wl_dualsparsity_and_SEP{wl_ref_dualsparsity};
    wl_dualsparsity_and_SEP.sep_activators = sepInfo;

    // wl when output sparsity is active
    VPUNN::DPUWorkload wl_dualsparsity_and_output_sparsity{wl_ref_dualsparsity};
    wl_dualsparsity_and_output_sparsity.outputs[0].set_sparsity(true);

    // vector of dual sparsity workloads but also have active SEP or active output sparsity or none of them
    std::vector<VPUNN::DPUWorkload> dual_sparsity_workloads = {std::move(wl_ref_dualsparsity),
                                                               std::move(wl_dualsparsity_and_output_sparsity),
                                                               std::move(wl_dualsparsity_and_SEP)};

    EXPECT_TRUE(cost_models.getModel(device27).nn_initialized());
    std::string info = "";

    auto verify_cycle = [&info, this](std::vector<DPUWorkload>& dual_sparsity_workloads) {
        for (const DPUWorkload& wl_ref : dual_sparsity_workloads) {
            // PRECONDITION: verify if both input and weight sparsity are active
            ASSERT_TRUE(wl_ref.inputs[0].get_sparsity());
            ASSERT_TRUE(wl_ref.weight_sparsity_enabled);

            DPUWorkload wl_ref_weight_and_input_spars_on{wl_ref};

            // the initial wl, but just input sparsity is on
            DPUWorkload wl_ref_input_spars_on{wl_ref_weight_and_input_spars_on};
            wl_ref_input_spars_on.weight_sparsity_enabled = false;
            wl_ref_input_spars_on.weight_sparsity = 0.0F;

            // the initial wl, but just weight sparsity is on
            DPUWorkload wl_ref_weight_spars_on{wl_ref_weight_and_input_spars_on};
            wl_ref_weight_spars_on.inputs[0].set_sparsity(false);
            wl_ref_weight_spars_on.act_sparsity = 0.0F;

            DPUWorkload wl_ref_no_sparsity{wl_ref_weight_and_input_spars_on};
            wl_ref_no_sparsity.weight_sparsity_enabled = false;
            wl_ref_no_sparsity.weight_sparsity = 0.0F;
            wl_ref_no_sparsity.inputs[0].set_sparsity(false);
            wl_ref_no_sparsity.act_sparsity = 0.0F;

            // compute the runtime
            const CyclesInterfaceType cyc_dual_sparsity{
                    cost_models.getModel(device27).DPU(std::move(wl_ref_weight_and_input_spars_on), info)};
            const CyclesInterfaceType cyc_weight_sparsity{cost_models.getModel(device27).DPU(std::move(wl_ref_weight_spars_on), info)};
            const CyclesInterfaceType cyc_input_sparsity{cost_models.getModel(device27).DPU(std::move(wl_ref_input_spars_on), info)};
            const CyclesInterfaceType cyc_NO_sparsity{cost_models.getModel(device27).DPU(std::move(wl_ref_no_sparsity), info)};

            // PRECONDITIONS:
            //  cycles value for workloads when only input sparsity is active or only weight sparsity is active should
            //  not be an error if it is an error, assert stop the test
            ASSERT_FALSE(is_error_code(cyc_weight_sparsity))
                    << Cycles::toErrorText(cyc_weight_sparsity) << " Precondition not met, error found\n";
            ASSERT_FALSE(is_error_code(cyc_input_sparsity))
                    << Cycles::toErrorText(cyc_input_sparsity) << " Precondition not met, error found\n";

            ASSERT_FALSE(is_error_code(cyc_NO_sparsity))
                    << Cycles::toErrorText(cyc_NO_sparsity) << " Precondition not met, error found\n";

            // verify that cycle time for initial wl with both sparsities active is not an error code
            ASSERT_FALSE(is_error_code(cyc_dual_sparsity))
                    << Cycles::toErrorText(cyc_dual_sparsity) << " Precondition not met, error found\n";

            CyclesInterfaceType min_cyc = std::min(cyc_weight_sparsity, cyc_input_sparsity);
            EXPECT_EQ(cyc_dual_sparsity, min_cyc)
                    << cyc_dual_sparsity << ":" << Cycles::toErrorText(cyc_dual_sparsity) << "\n"
                    << cyc_input_sparsity << ":" << Cycles::toErrorText(cyc_input_sparsity) << "\n"
                    << cyc_weight_sparsity << ":" << Cycles::toErrorText(cyc_weight_sparsity) << "\n"
                    << cyc_NO_sparsity << ":" << Cycles::toErrorText(cyc_NO_sparsity) << "\n";

            EXPECT_LT(cyc_input_sparsity, cyc_NO_sparsity)
                    << cyc_dual_sparsity << ":" << Cycles::toErrorText(cyc_dual_sparsity) << "\n"
                    << cyc_input_sparsity << ":" << Cycles::toErrorText(cyc_input_sparsity) << "\n"
                    << cyc_weight_sparsity << ":" << Cycles::toErrorText(cyc_weight_sparsity) << "\n"
                    << cyc_NO_sparsity << ":" << Cycles::toErrorText(cyc_NO_sparsity) << "\n";
            EXPECT_LT(cyc_weight_sparsity, cyc_NO_sparsity)
                    << cyc_dual_sparsity << ":" << Cycles::toErrorText(cyc_dual_sparsity) << "\n"
                    << cyc_input_sparsity << ":" << Cycles::toErrorText(cyc_input_sparsity) << "\n"
                    << cyc_weight_sparsity << ":" << Cycles::toErrorText(cyc_weight_sparsity) << "\n"
                    << cyc_NO_sparsity << ":" << Cycles::toErrorText(cyc_NO_sparsity) << "\n";
        }
    };

    verify_cycle(dual_sparsity_workloads);
}

TEST_F(TestCostModelVPU2x, Dual_sparsity_NN_Output_Cycle_invalid_values_Test) {
    // TEST CASE1 explanation
    //  when we compute cycles for this workload, the cycles value is not an error code
    //  in that test when we deactivate input sparsity, cycles value is still not an error (weight sparsity active is
    //  enough larger to reduce cycles) when we deactivate weight sparsity cycles value is an error "INPUT TOO BIG"
    const VPUNN::DPUWorkload wl_ref_in_spars_err{
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 108, 256, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},  // input dimensions
            {VPUNN::VPUTensor(21, 36, 32, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},    // output dimensions
            {3, 3},                                                                          // kernels
            {3, 3},                                                                          // strides
            {0, 0, 0, 0},                                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                              // execution mode
            VPUNN::ActivationFunction::NONE,                                                 // activation
            0.21F,                                                                           // act_sparsity
            0.99F,                                                                           // weight_sparsity
            {swz_def, swz_def},                                                              // input_swizzling
            {swz_def},                                                                       // output_swizzling
            1,                                                                               // output_write_tiles
            {0, 0, 0, 0},                                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                                  // isi_strategy
            true                                                                             // weight_sparsity_enabled

    };

    // TEST CASE2 explanation
    // when we compute cycles for this workload, the cycles value is an error code "INPUT TOO BIG"
    // in that test when we deactivate input sparsity, cycles value is not an error (weight sparsity active is enough
    // larger to reduce cycles)
    //  when we deactivate weight sparsity cycles value is an error "INPUT TOO BIG" input sparsity is not large enough
    const VPUNN::DPUWorkload wl_ref_in_and_dual_spars_err{
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 120, 256, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},  // input dimensions
            {VPUNN::VPUTensor(21, 40, 32, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},    // output dimensions
            {3, 3},                                                                          // kernels
            {3, 3},                                                                          // strides
            {0, 0, 0, 0},                                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                              // execution mode
            VPUNN::ActivationFunction::NONE,                                                 // activation
            0.28F,                                                                           // act_sparsity
            0.99F,                                                                           // weight_sparsity
            {swz_def, swz_def},                                                              // input_swizzling
            {swz_def},                                                                       // output_swizzling
            1,                                                                               // output_write_tiles
            {0, 0, 0, 0},                                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                                  // isi_strategy
            true                                                                             // weight_sparsity_enabled

    };

    // TEST CASE3 explanation
    // when we compute cycles for this workload, the cycles value is an error code "INPUT TOO BIG"
    // in that test when we deactivate input sparsity, cycles value is an error "INPUT TOO BIG" weight sparsity value is
    // not large enough to reduce cycles
    // when we deactivate weight sparsity cycles value is also an error "INPUT TOO BIG" input sparsity value is not
    // large enough
    const VPUNN::DPUWorkload wl_ref_in_weight_dual_spars_err{
            device27,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(64, 128, 512, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},  // input dimensions
            {VPUNN::VPUTensor(21, 42, 512, 1, VPUNN::DataType::UINT8, Layout::ZXY, true)},   // output dimensions
            {3, 3},                                                                          // kernels
            {3, 3},                                                                          // strides
            {0, 0, 0, 0},                                                                    // padding
            VPUNN::ExecutionMode::CUBOID_16x16,                                              // execution mode
            VPUNN::ActivationFunction::NONE,                                                 // activation
            0.7F,                                                                            // act_sparsity
            0.32F,                                                                           // weight_sparsity
            {swz_def, swz_def},                                                              // input_swizzling
            {swz_def},                                                                       // output_swizzling
            1,                                                                               // output_write_tiles
            {0, 0, 0, 0},                                                                    // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                                  // isi_strategy
            true                                                                             // weight_sparsity_enabled

    };

    struct TestInput {
        VPUNN::DPUWorkload wl_ref;
    };

    // we test 3 wl with dualsparsity (input + weight), just input sparsity and just weight sparsity
    //  those values show if we expect an error or not
    struct TestExpectation {
        bool is_error_dualsparsity;     // both input and weight sparsity on
        bool is_error_weight_sparsity;  // weight sparsity on, input sparsity off
        bool is_error_input_sparsity;   // input sparsity on, weight sparsity off
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        std::string t_info = "";
    };

    using TestsVector = std::vector<TestCase>;

    EXPECT_TRUE(cost_models.getModel(device27).nn_initialized());

    // inside lambda function, we take an wl and create 2 other workloads, each with only one  active sparsity (either
    // weight or input), and then we test if the value of cycles is an error
    // for example: a wl that has both active sparsities may not generate an error, but when we deactivate weight
    // sparsity now this wl has only input sparsity active => may generate an error
    auto verify_cycle = [this](const TestsVector& tests) {
        for (const auto& t : tests) {
            std::cout << " Test CASE: " << t.t_info << "\n";

            // PRECONDITION: verify if both input and weight sparsity are active
            ASSERT_TRUE(t.t_in.wl_ref.inputs[0].get_sparsity());
            ASSERT_TRUE(t.t_in.wl_ref.weight_sparsity_enabled);

            DPUWorkload wl_ref_weight_and_input_spars_on{t.t_in.wl_ref};

            // the initial wl, but just input sparsity is on
            DPUWorkload wl_ref_input_spars_on{wl_ref_weight_and_input_spars_on};
            wl_ref_input_spars_on.weight_sparsity_enabled = false;
            wl_ref_input_spars_on.weight_sparsity = 0.0F;

            // the initial wl, but just weight sparsity is on
            DPUWorkload wl_ref_weight_spars_on{wl_ref_weight_and_input_spars_on};
            wl_ref_weight_spars_on.inputs[0].set_sparsity(false);
            wl_ref_weight_spars_on.act_sparsity = 0.0F;

            std::string info_dualsparsity = "";
            std::string info_input_sparsity = "";
            std::string info_weight_sparsity = "";

            // compute the runtime
            const CyclesInterfaceType cyc_dual_sparsity{
                    cost_models.getModel(device27).DPU(std::move(wl_ref_weight_and_input_spars_on), info_dualsparsity)};
            const CyclesInterfaceType cyc_weight_sparsity{
                    cost_models.getModel(device27).DPU(std::move(wl_ref_weight_spars_on), info_input_sparsity)};
            const CyclesInterfaceType cyc_input_sparsity{
                    cost_models.getModel(device27).DPU(std::move(wl_ref_input_spars_on), info_weight_sparsity)};

            // verify that cycle time is/is not an error code
            EXPECT_EQ(is_error_code(cyc_dual_sparsity), t.t_exp.is_error_dualsparsity)
                    << Cycles::toErrorText(cyc_dual_sparsity) << info_dualsparsity << "\n";
            EXPECT_EQ(is_error_code(cyc_weight_sparsity), t.t_exp.is_error_weight_sparsity)
                    << Cycles::toErrorText(cyc_weight_sparsity) << info_weight_sparsity << "\n";
            EXPECT_EQ(is_error_code(cyc_input_sparsity), t.t_exp.is_error_input_sparsity)
                    << Cycles::toErrorText(cyc_input_sparsity) << info_input_sparsity << "\n";
        }
    };

    //!! the explanations for each test are found above each corresponding wl
    // a test case contains:
    //   1. an wl : this should have both sparsities active (input + weight)
    //   2. inside lambda function @see verify_cycles we take this wl and we create 2 other workloads, each with only
    //   one active sparsity
    //   ---> here we have 3 booleans (what I expect): (true means the value of cycles is an error)
    //            1. is the cycles value for wl with dualsparsity on an error code?
    //            2. is the cycles value for wl with weight sparsity on an error code?
    //            3. is the cycles value for wl with input sparsity on an error code?
    const TestsVector tests = {
            // clang-format off
      //    ||      wl_ref       || cyc err for wl dualspars on || cyc err for wl weight spars on || cyc err for wl in spars on || test info ||
            {{std::move(wl_ref_in_spars_err)}, {false, false, true}, "Cycles value is an ERROR for the wl that only has input sparsity active "},
            {{std::move(wl_ref_in_and_dual_spars_err)},{true, false, true}, "Cycles value is an ERROR for the wl that has input sparsity active and for the wl that has dualsparsity " "active"},
            {{std::move(wl_ref_in_weight_dual_spars_err)}, {true, true, true}, "Cycles value is an ERROR for the wl that has dualsparsity, the wl that has input and the wl that has " "weight sparsity"},
    };
    // clang-format on
    verify_cycle(tests);
}

// This tests that K4096 is invalid.
// connected with : (VPULayerCM_InvestigationTest, Layer_MAXP_EISXW_na_MINGQI_NPU27 )
TEST_F(TestCostModelVPU2x, MAXPOOL_test_EISXW_na_Mingqi_NPU27) {
    Logger::activate2ndlog();
    {
        const HaloWorkload zeroHalo;
        const SEPModeInfo sepInfo{};
        const DPUWorkload wl_ref_ = {
                // K will be too big , max 64 allowed fro MAXPOOOL
                device27,
                Operation::MAXPOOL,
                {VPUTensor(40, 4, 4096, 1, DataType::UINT8, Layout::ZXY)},  // input dimensions
                {VPUTensor(40, 4, 4096, 1, DataType::UINT8, Layout::ZXY)},  // output dimensions
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
                ISIStrategy::SPLIT_OVER_H,                                  // isi_strategy
                false,                                                      // weight_sparsity_enabled
                zeroHalo,                                                   // halo
                sepInfo,                                                    // SEP
        };

        {
            // sparsity (input) is allowed only for CONV and ELEMENTwise, prohibited for rest
            // output sparsity should be enabled for all// not influencing  inferred runtime

            {  // conv sparse output
                DPUWorkload wl{std::move(wl_ref_)};
                std::string info;
                auto cycles = cost_models.getModel(device27).DPU(wl, info);  // will change

                EXPECT_TRUE(Cycles::isErrorCode(cycles))
                        << "ERROR code received: " << cycles << " : " << Cycles::toErrorText(cycles)
                        << "\n INFO: " << wl << info << std::endl;

                EXPECT_TRUE((cycles == Cycles::ERROR_INVALID_INPUT_CONFIGURATION));
            }
        }
    }
}

TEST_F(TestCostModelVPU2x, Weigths_types_NPU27_test) {
    constexpr int kw{3};
    constexpr int kh{3};
    constexpr int in_ch = 64;
    constexpr int o_ch = 64;

    DPU_OperationValidator dut;
    class Builder {
    public:
        static DPUWorkload makeWL(Operation op, unsigned int in_ch, unsigned int o_ch, DataType Tin, DataType Tout,
                                  unsigned int kw, unsigned int kh, DataType wts_t, unsigned int padd) {
            return DPUWorkload{
                    VPUDevice::VPU_2_7,
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

    auto verify = [this, &dut, input1_volume](const DPUWorkload& wl) {
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
        auto cycles = cost_models.getModel(device27).DPU(wl, msg);
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

TEST_F(TestCostModelVPU2x, MTL_Weigths_types_CONV_NPU27) {
    const HaloWorkload halo{};
    const SEPModeInfo sep_activators{};

    std::optional<DataType> wts_t{};

    constexpr int num16KB{16384};

    constexpr int kw{4};
    constexpr int kh{3};
    constexpr int in_ch = ((16 * 2) * 1024 / 32) / (4);
    constexpr int o_ch = 32 * 5;

    DPU_OperationValidator dut;  //

    const DPUWorkload base_wl8_8{
            device27,
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
        auto cycles = cost_models.getModel(device27).DPU(wl, msg);

        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;
        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    const DPUWorkload base_wl16_16{
            device27,
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
        auto cycles = cost_models.getModel(device27).DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    // int8 in - INT4 wts
    const DPUWorkload base_wl8_4{
            device27,
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
        auto cycles = cost_models.getModel(device27).DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    // FP16 in - INT4 wts
    const DPUWorkload base_wl16_4{
            device27,
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
        auto cycles = cost_models.getModel(device27).DPU(wl, msg);
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << Cycles::toErrorText(cycles) << msg;

        MemorySize mem;
        EXPECT_NO_THROW(mem = dut.compute_wl_memory(wl)) << wl << std::endl;

        std::cout << ttl << "Cycles: " << cycles << std::endl;

        EXPECT_EQ(mem.input_1, wl_wts_align_full_size) << ttl << "\nMemory size: " << mem << wl << std::endl;
    }

    // check runtime equivalence
    {
        EXPECT_EQ(cost_models.getModel(device27).DPU(base_wl8_4), cost_models.getModel(device27).DPU(std::move(base_wl8_8)));
        EXPECT_EQ(cost_models.getModel(device27).DPU(base_wl16_4), cost_models.getModel(device27).DPU(std::move(base_wl16_16)));

        DPUWorkload base_wl16_8{base_wl16_4};
        base_wl16_8.weight_type = DataType::INT8;

        EXPECT_EQ(cost_models.getModel(device27).DPU(std::move(base_wl16_4)), cost_models.getModel(device27).DPU(std::move(base_wl16_8)));

        DPUWorkload base_wl8_16{base_wl8_4};
        base_wl8_16.weight_type = DataType::FLOAT16;
        EXPECT_EQ(cost_models.getModel(device27).DPU(std::move(base_wl8_4)), cost_models.getModel(device27).DPU(std::move(base_wl8_16)));
    }

    // EXPECT_TRUE(false);
}

}  // namespace VPUNN_unit_tests