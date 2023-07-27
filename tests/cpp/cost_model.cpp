// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu_cost_model.h"

#include <gtest/gtest.h>
#include "common_helpers.h"
#include "vpu/compatibility/types01.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/sample_generator/random_task_generator.h"
#include "vpu/validation/interface_valid_values.h"

#include <algorithm>
#include <unordered_map>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {

class TestCostModel : public ::testing::Test {
public:
protected:
    VPUNN::DPUWorkload wl_glob_27{VPUNN::VPUDevice::VPU_2_7,
                                  VPUNN::Operation::CONVOLUTION,
                                  {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
                                  {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
                                  {3, 3},                                                     // kernels
                                  {1, 1},                                                     // strides
                                  {1, 1, 1, 1},                                               // padding
                                  VPUNN::ExecutionMode::CUBOID_16x16};
    VPUNN::DPUWorkload wl_glob_20;

    VPUNN::VPUCostModel model{VPUNN::VPUCostModel()};

    void SetUp() override {
        wl_glob_20 = wl_glob_27;
        wl_glob_20.device = VPUNN::VPUDevice::VPU_2_0;
        wl_glob_20.execution_order = VPUNN::ExecutionMode::MATRIX;
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

    using ModelDescriptor =
            VPUNNModelsFiles::ModelDescriptor;  ///< make this type directly visible inside of this class
    const VPUNNModelsFiles& the_NN_models{VPUNNModelsFiles::getModels()};  ///< the paths to available NN models

    bool is_error_code(unsigned int cycles) {
        if (cycles > std::numeric_limits<uint32_t>::max() - 1000)
            return true;
        return false;
    }

    struct DataOut {
        int errors_cnt{-1};
        double correlation{-10.0};
    };

    VPUNN::CyclesInterfaceType delta_cycles(const VPUNN::CyclesInterfaceType& v1,
                                            const VPUNN::CyclesInterfaceType& v2) {
        return (v1 >= v2) ? (v1 - v2) : (v2 - v1);  // aways positive
    }

    /// @brief max allowable delta between 2 cycles , so that we consider them still equal
    ///
    /// @param v1 a value
    /// @param v2 another value
    /// @param tolerance_level how permissive to be in delta.
    /// @returns max value that can be between v1 and v2 so that they are practically equal.
    VPUNN::CyclesInterfaceType max_tolerance_cycles(const VPUNN::CyclesInterfaceType& v1,
                                                    const VPUNN::CyclesInterfaceType& v2,
                                                    const int tolerance_level = 1) {
        const VPUNN::CyclesInterfaceType v{std::max(v1, v2)};

        VPUNN::CyclesInterfaceType tolerance{1U};  // rounding errors

        if (tolerance_level <= 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 4U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 4U;
            } else if (v >= 100000U) {  // 100k
                tolerance = 3U;
            } else if (v >= 1000U) {
                tolerance = 2U;
            }

        } else if (tolerance_level > 1) {
            if (v >= 10000000U) {  // 10 millions
                tolerance = 20U;
            } else if (v >= 1000000U) {  // 1 million
                tolerance = 10U;
            } else if (v >= 1000U) {
                tolerance = 2U;
            }
        }

        return tolerance;
    }

private:
};

TEST_F(TestCostModel, LoadModels_BasicAssertions) {
    {  // 2_0
        const std::string model_path = VPU_2_0_MODEL_PATH;
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }

    {  // 2_7
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }

    {  // 2_0 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH)};
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }

    {  // 2_7 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH)};
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());
    }
}

TEST_F(TestCostModel, LoadModels_NN_Valid_Interval) {
    float down_exp = 0.0F;
    float up_exp = 4000000000.0F;

    {  // empty models

        ASSERT_FALSE(model.nn_initialized());
        auto minmax = model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_0
        const std::string model_path = VPU_2_0_MODEL_PATH;
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_7
        const std::string model_path = VPU_2_7_MODEL_PATH;
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_0 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH)};
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }

    {  // 2_7 fast
        const std::string model_path{NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH)};
        EXPECT_NO_THROW(auto x = VPUNN::VPUCostModel(model_path));
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        ASSERT_TRUE(vpunn_model.nn_initialized());

        auto minmax = vpunn_model.get_NN_Valid_interval();
        ASSERT_FLOAT_EQ(down_exp, minmax.first);
        ASSERT_FLOAT_EQ(up_exp, minmax.second);
    }
}

// Demonstrate some basic assertions.
TEST_F(TestCostModel, BatchTestVPUNN_NN2_7) {
    // Dummy WL
    VPUNN::DPUWorkload wl0 = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
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

    auto runtime_model = VPUNN::Runtime(VPU_2_7_MODEL_PATH, 1);
    auto model_batched = VPUNN::Runtime(VPU_2_7_MODEL_PATH, batch_size);

    ASSERT_EQ(runtime_model.model_version_info().get_input_interface_version(), preprocessing.interface_version())
            << "Runtime expected input version and preprocessing version must be the same" << std::endl;
    ASSERT_EQ(model_batched.model_version_info().get_input_interface_version(), preprocessing.interface_version())
            << "Runtime expected input version and preprocessing version must be the same" << std::endl;

    const auto& input_shape = runtime_model.input_tensors()[0]->shape();
    const auto& output_shape = runtime_model.output_tensors()[0]->shape();
    const auto& input_shape_batched = model_batched.input_tensors()[0]->shape();

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
        auto& data = preprocessing.transform(workloads[idx]);
        // Create a new pointer otherwise it will get overwritten every time
        inference[idx] = new float[output_shape[1]];
        memcpy((void*)inference[idx], runtime_model.predict(&(data[0]), input_size), sizeof(float) * output_shape[1]);
        // Compute inference using the vector interface
        inference_vector[idx] = runtime_model.predict<float>(data);
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
    const auto& batched_data = preprocessing.transform(workloads);
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

/// Demonstrate outputs from batch are the same as from single runs. Random data
TEST_F(TestCostModel, BatchTestVPUNNCostModel_VPUNN_2_0_stochastic) {
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
TEST_F(TestCostModel, BatchTestVPUNNCostModel_VPUNN_2_7F_stochastic) {
    const std::string model_path = the_NN_models.fast_model_paths[1].first;
    const VPUNN::VPUDevice device_version = the_NN_models.fast_model_paths[1].second;

    for (auto batch_size : {1, 2, 4, 5, 7, 11, 13, 17, 30, 40, 100}) {
        for (auto n_workloads : {1, 2, 4, 5, 7, 11, 13, 17, 30, 40, 100}) {
            VPUNN::VPUCostModel batched_model = VPUNN::VPUCostModel(model_path, false, 0, batch_size);
            VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path, false, 0, 1);

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

TEST_F(TestCostModel, BatchTestVPUNNCostModel_VPUNN_2_7F_Particular1) {
    const std::string model_path = the_NN_models.fast_model_paths[1].first;
    const VPUNN::VPUDevice device_version = the_NN_models.fast_model_paths[1].second;
    int batch_size = 1;
    const int n_workloads = 1;

    VPUNN::VPUCostModel batched_model = VPUNN::VPUCostModel(model_path, false, 0, batch_size);
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path, false, 0, 1);

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
    workloads[0] = wl_ref;

    const auto w0{workloads[0]};

    {
        // Batched Inference for cycles.
        std::vector<VPUNN::CyclesInterfaceType> batched_cycles = batched_model.DPU(workloads);

        const auto w01{workloads[0]};
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
TEST_F(TestCostModel, BatchTest_SanitizedWorkloadsEquivalence) {
    // Dummy WL
    VPUNN::DPUWorkload wl0 = {
            VPUNN::VPUDevice::VPU_2_7,
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
            VPUNN::VPUDevice::VPU_2_7,
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
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},         // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled
    };

    // avepool is not supported, but should be transformed to an equivalent
    const VPUNN::DPUWorkload wl3 = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},         // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled

    };

    std::vector<VPUNN::DPUWorkload> workloads = {wl0, wl1, wl2, wl3};

    auto batch_size = (unsigned int)workloads.size();

    const std::string model_path = the_NN_models.fast_model_paths[1].first;  // fast nn2.7
    VPUNN::VPUCostModel batched_model = VPUNN::VPUCostModel(model_path, false, 0, batch_size);
    VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path, false, 0, 1);

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
TEST_F(TestCostModel, DISABLED_SmokeTestDPU) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
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
TEST_F(TestCostModel, SmokeTestDPUVPU_2_0Model) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {3, 3},                                                       // kernels
            {1, 1},                                                       // strides
            {1, 1, 1, 1},                                                 // padding
            VPUNN::ExecutionMode::VECTOR_FP16                             // execution mode
    };
    VPUNN::VPUCostModel model_2_0{VPUNN::VPUCostModel(VPU_2_0_MODEL_PATH)};

    float overhead = static_cast<float>(model_2_0.run_NN(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(overhead > 1);
}

TEST_F(TestCostModel, SmokeTestDPUVPU_2_0Model_Eltwise) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::VECTOR                                // execution mode
    };
    VPUNN::VPUCostModel model_2_0{VPUNN::VPUCostModel(VPU_2_0_MODEL_PATH)};
    float cycles = static_cast<float>(model_2_0.DPU(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(cycles > (16 / 4) * (16 / 4) * (64 / 16));
}

TEST_F(TestCostModel, SmokeTestDPUVPU27Model_Eltwise) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

    float cycles = static_cast<float>(model_2_7.DPU(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(cycles > (16 / 4) * (16 / 4) * (64 / 16));
}

// Demonstrate some basic assertions.
TEST_F(TestCostModel, SmokeTestDPUVPU27Model) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

    float overhead = static_cast<float>(model_2_7.run_NN(wl));

    // Expect hw overhead to be valid
    EXPECT_TRUE(overhead > 1);
}

TEST_F(TestCostModel, SmokeTestDMA) {
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

TEST_F(TestCostModel, SmokeTestCompressedDMA) {
    // Compressed DMA with 50% CR
    auto dma_cycles = model.DMA(VPUNN::VPUDevice::VPU_2_7, VPUNN::VPUTensor(25088, 1, 1, 1, VPUNN::DataType::UINT8),
                                VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8), VPUNN::MemoryLocation::DRAM,
                                VPUNN::MemoryLocation::CMX);

    // Expect equality.
    EXPECT_EQ(dma_cycles, 950 + static_cast<unsigned int>(ceil(25088 * 1300 / 27000.0f)));
}

TEST_F(TestCostModel, SmokeTestPermutedDMA) {
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

TEST_F(TestCostModel, Special_Tests_DPU_MAXPOOL_VPU_2_0_1_96_96_9_9_1_2_VALID_FLOAT16_MATRIX) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_0,
            VPUNN::Operation::MAXPOOL,
            {VPUNN::VPUTensor(9, 9, 96, 1, VPUNN::DataType::FLOAT16)},  // input dimensions WHCB
            {VPUNN::VPUTensor(9, 9, 96, 1, VPUNN::DataType::FLOAT16)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::VECTOR_FP16                           // execution mode
    };
    VPUNN::DPUWorkload wl_smaller = {
            VPUNN::VPUDevice::VPU_2_0,
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
        VPUNN::VPUCostModel model_slow{VPU_2_0_MODEL_PATH};
        TestCycles cycles_slow{model_slow.DPU(wl), model_slow.DPU(wl_smaller)};

        VPUNN::VPUCostModel model_fast{the_NN_models.fast_model_paths[0].first};
        TestCycles cycles_fast{model_fast.DPU(wl), model_fast.DPU(wl_smaller)};

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_slow.cycles_smaller))
                << VPUNN::Cycles::toErrorText(cycles_slow.cycles_smaller);
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_slow.cycles)) << VPUNN::Cycles::toErrorText(cycles_slow.cycles);

        EXPECT_LE(cycles_slow.cycles_smaller, cycles_slow.cycles)
                << " MOdel 2.0 slow \n"
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

TEST_F(TestCostModel, AVEPOOL_equivalence_test_27) {
    const VPUNN::DPUWorkload wl_avgpool_ref = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},         // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled

    };
    const auto equivalent_op{VPUNN::Operation::DW_CONVOLUTION};

    VPUNN::VPUCostModel crt_model{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};
    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = crt_model.DPU(wl_avgpool);
            auto cycles_equiv = crt_model.DPU(wl_equiv);

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
            auto cycles_avgpool = crt_model.DPU(wl_avgpool);
            auto cycles_equiv = crt_model.DPU(wl_equiv);

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }

    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        wl_avgpool.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
        wl_avgpool.output_write_tiles = 2;
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = crt_model.DPU(wl_avgpool);
            auto cycles_equiv = crt_model.DPU(wl_equiv);

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }
}

TEST_F(TestCostModel, AVEPOOL_equivalence_test_20) {
    const VPUNN::DPUWorkload wl_avgpool_ref = {
            VPUNN::VPUDevice::VPU_2_0,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},         // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                  // output_swizzling
            1,                                                          // output_write_tiles
            {0, 0, 0, 0},                                               // offsets
            VPUNN::ISIStrategy::CLUSTERING,                             // isi_strategy
            false,                                                      // weight_sparsity_enabled

    };
    const auto equivalent_op{VPUNN::Operation::DW_CONVOLUTION};

    VPUNN::VPUCostModel crt_model{VPUNN::VPUCostModel(VPU_2_0_MODEL_PATH)};
    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = crt_model.DPU(wl_avgpool);
            auto cycles_equiv = crt_model.DPU(wl_equiv);

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
            auto cycles_avgpool = crt_model.DPU(wl_avgpool);
            auto cycles_equiv = crt_model.DPU(wl_equiv);

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }

    {
        VPUNN::DPUWorkload wl_avgpool{wl_avgpool_ref};
        wl_avgpool.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;  // not supported on 2,.0
        VPUNN::DPUWorkload wl_equiv{wl_avgpool};
        wl_equiv.op = equivalent_op;

        {
            auto cycles_avgpool = crt_model.DPU(wl_avgpool);
            auto cycles_equiv = crt_model.DPU(wl_equiv);

            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_avgpool));
            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_avgpool, cycles_equiv);
            EXPECT_TRUE((cycles_avgpool != 0) && (cycles_equiv != 0));
        }
    }
}

TEST_F(TestCostModel, Datatype_Sanity_test_VPU27) {
    const VPUNN::DPUWorkload wl_ref_int = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled

    };
    const VPUNN::DataType normalized_data_int{VPUNN::DataType::UINT8};

    const VPUNN::DPUWorkload wl_ref_flt = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},            // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                     // output_swizzling
            1,                                                             // output_write_tiles
            {0, 0, 0, 0},                                                  // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                // isi_strategy
            false,                                                         // weight_sparsity_enabled

    };
    const VPUNN::DataType normalized_data_flt{VPUNN::DataType::FLOAT16};

    {
        VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

        {
            VPUNN::DPUWorkload wl{wl_ref_int};
            VPUNN::DPUWorkload wl_equiv{wl};
            wl_equiv.inputs[0].change_datatype_superficial(normalized_data_int);
            wl_equiv.outputs[0].change_datatype_superficial(normalized_data_int);

            auto cycles_raw = model_2_7.DPU(wl);  // will change
            auto cycles_equiv = model_2_7.DPU(wl_equiv);

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_raw));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_raw, cycles_equiv);
            EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
        }

        {
            VPUNN::DPUWorkload wl{wl_ref_flt};
            VPUNN::DPUWorkload wl_equiv{wl};
            wl_equiv.inputs[0].change_datatype_superficial(normalized_data_flt);
            wl_equiv.outputs[0].change_datatype_superficial(normalized_data_flt);

            auto cycles_raw = model_2_7.DPU(wl);  // will change
            auto cycles_equiv = model_2_7.DPU(wl_equiv);

            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_raw));
            EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_equiv));

            EXPECT_EQ(cycles_raw, cycles_equiv);
            EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
        }
    }
}

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

        auto cycles_20 = vpunn_model_buf.DPU(wl_glob_20);
        auto cycles_27 = vpunn_model_buf.DPU(wl_glob_27);

        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_20));
        EXPECT_FALSE(VPUNN::Cycles::isErrorCode(cycles_27));

        EXPECT_EQ(cycles_20, 28224);  // theoretical
        EXPECT_EQ(cycles_27, 3528);   // theoretical
    }
}

TEST_F(TestCostModel, ComaparativeRuns) {
    const std::string model_path = VPU_2_0_MODEL_PATH;

    auto modelRun = [](const std::string& model_path, VPUNN::DPUWorkload& wld) {
        VPUNN::VPUCostModel vpunn_model = VPUNN::VPUCostModel(model_path);
        std::cout << model_path << " : initialized: " << vpunn_model.nn_initialized() << std::endl;

        std::cout << "run_NN(wl)   : " << vpunn_model.run_NN(wld) << std::endl;
        std::cout << "DPU(wl)   : " << vpunn_model.DPU(wld) << std::endl;
        std::cout << "hw_utilization(wl)   : " << vpunn_model.hw_utilization(wld) << std::endl;
    };

    std::cout << "----------------------------------------------------------\n";
    modelRun(VPU_2_0_MODEL_PATH, wl_glob_20);
    // modelRun(VPU_2_0_MODEL_PATH, wl_glob_20);
    std::cout << "----------------------------------------------------------\n";
    modelRun(VPU_2_7_MODEL_PATH, wl_glob_27);
    // modelRun(VPU_2_7_MODEL_PATH, wl_glob_27);
    std::cout << "----------------------------------------------------------\n";
    modelRun(NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH), wl_glob_20);
    // modelRun(NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH), wl_glob_20);
    std::cout << "----------------------------------------------------------\n";
    modelRun(NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH), wl_glob_27);
    // modelRun(NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH), wl_glob_27);
    std::cout << "----------------------------------------------------------\n";
}

TEST_F(TestCostModel, OutputWriteTiles_multiple) {
    const VPUNN::DPUWorkload wl_ref_1x1_f = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},             // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                      // output_swizzling
            1,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false,                                                          // weight_sparsity_enabled

    };
    const VPUNN::DPUWorkload wl_ref_5x5_f = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},             // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                      // output_swizzling
            0,                                                              // output_write_tiles
            {0, 0, 0, 0},                                                   // offsets
            VPUNN::ISIStrategy::CLUSTERING,                                 // isi_strategy
            false,                                                          // weight_sparsity_enabled

    };

    VPUNN::VPUCostModel m{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

    VPUNN::DPU_OperationValidator ops;  ///< sanitizer mechanisms
    const VPUNN::IDeviceValidValues& cfg{ops.get_config(VPUNN::VPUDevice::VPU_2_7)};
    auto owt_list = cfg.output_write_tile_options;
    std::sort(owt_list.begin(), owt_list.end());

    auto run_test_1owt = [&m](const VPUNN::DPUWorkload& wl, const std::string& h) {
        auto cycles = m.DPU(wl);

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
            VPUNN::DPUWorkload wl{wl_ref_1x1_f};
            EXPECT_TRUE(expectationToForce) << wl;

            wl.isi_strategy = VPUNN::ISIStrategy::CLUSTERING;
            run_test_all_owt(wl, "Clustering: ");

            wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_H;
            run_test_all_owt(wl, "SOH: ");

            wl.isi_strategy = VPUNN::ISIStrategy::SPLIT_OVER_K;
            // run_test_all_owt(wl, "SOK: ");// not possible owt =1
        }
        {
            VPUNN::DPUWorkload wl{wl_ref_5x5_f};
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

class TestCyclesInterfaceType : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Some compilers do not support constexpr where references to them are kind of assumed to be needed (GCC/LLVM)
/// some have no problems (MSVC)
TEST_F(TestCyclesInterfaceType, BasicOps) {
    EXPECT_EQ(V(VPUNN::Cycles::NO_ERROR), 0);

    auto mm = std::max(V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG), V(VPUNN::Cycles::ERROR_INVALID_INPUT_CONFIGURATION));
    EXPECT_EQ(mm, V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG));

    EXPECT_NE(V(VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION), V(VPUNN::Cycles::ERROR_INPUT_TOO_BIG));

    // const VPUNN::CyclesInterfaceType& ref = VPUNN::Cycles::ERROR_INPUT_TOO_BIG;

    //    EXPECT_EQ(VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, VPUNN::Cycles::ERROR_INPUT_TOO_BIG);

    //  const VPUNN::CyclesInterfaceType* ptr = &VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION;

    // EXPECT_EQ(*ptr, VPUNN::Cycles::ERROR_INPUT_TOO_BIG);
    // EXPECT_EQ(ptr, nullptr);
    // EXPECT_EQ(&VPUNN::Cycles::ERROR_INPUT_TOO_BIG, nullptr);

    //    EXPECT_EQ(ptr, &VPUNN::Cycles::ERROR_INPUT_TOO_BIG);
    //   EXPECT_EQ(ptr, &VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION);
}
using namespace VPUNN;

/// compare CLustering versus SOH attributes and FUll Workloads vs halved workloads with SOH
class TestSplitMethodsComparisons : public ::testing::Test {
public:
protected:
    VPUNN::VPUCostModel model_2_7{VPU_2_7_MODEL_PATH};
    std::string info{};

    float strategy_scale = 2.0f;  // adjust SOK/SOH valued by this

    float tolerance_even = 0.1f;  // used for SOH
    // float tolerance_odd = 0.3f;
    float tolerance_SOK = 0.2f;

    void SetUp() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::activate2ndlog();
    }
    void TearDown() override {
        VPUNN::Logger::clear2ndlog();
        VPUNN::Logger::deactivate2ndlog();
    }

    void check_for_noError(CyclesInterfaceType cost_cyc, const DPUWorkload& wl, const std::string& t_header = "xxx") {
        EXPECT_FALSE(Cycles::isErrorCode(cost_cyc))
                << t_header << " > Unexpected ERROR code: " << cost_cyc << " : " << Cycles::toErrorText(cost_cyc)
                << "\n " << wl << Logger::get2ndlog();
        Logger::clear2ndlog();
    }

    void check_Cluster_vs_SOH(const DPUWorkload& wl_base, const std::string& t_header) {
        DPUWorkload wl{wl_base};
        std::cout << "\n****** TEST : " << t_header << "\n";
        wl.isi_strategy = ISIStrategy::CLUSTERING;
        auto cycles_clu = model_2_7.DPU(wl, info);
        check_for_noError(cycles_clu, wl, "CLUSTERING" + t_header + info);

        wl.isi_strategy = ISIStrategy::SPLIT_OVER_H;
        auto cycles_soh = model_2_7.DPU(wl, info);
        check_for_noError(cycles_soh, wl, "SOH" + t_header + info);

        cycles_soh = static_cast<decltype(cycles_soh)>(cycles_soh * strategy_scale);

        const std::int32_t delta = std::abs((std::int32_t)cycles_clu - (std::int32_t)cycles_soh);
        const std::int32_t maxDelta = (std::int32_t)(cycles_clu * tolerance_even);

        const auto deltapercent = ((float)delta / cycles_clu) * 100;

        // EXPECT_EQ(cycles_clu, cycles_soh) << t_header << " Cost not similar enough.\n"
        //                                   << "cost clustering: " << cycles_clu << "\n"
        //                                   << "cost soh       : " << cycles_soh << "\n"
        //                                   << wl_base;
        EXPECT_LE(delta, maxDelta) << t_header << " Cost not similar enough.\n"
                                   << "cost clustering: " << cycles_clu << "\n"
                                   << "cost soh       : " << cycles_soh << "\n"
                                   << "delta%         : " << deltapercent << "\n"
                                   << wl_base;
        std::cout << "\n--------- END TEST : " << t_header << " cost clustering: " << cycles_clu
                  << ", cost soh: " << cycles_soh << ", delta%: " << (int)(deltapercent) << " % \n";
    }

    void check_Cluster_vs_SOK(const DPUWorkload& wl_base, const std::string& t_header) {
        DPUWorkload wl{wl_base};
        std::cout << "\n****** TEST : " << t_header << "\n";
        wl.isi_strategy = ISIStrategy::CLUSTERING;
        auto cycles_clu = model_2_7.DPU(wl, info);
        check_for_noError(cycles_clu, wl, "CLUSTERING" + t_header + info);

        wl.isi_strategy = ISIStrategy::SPLIT_OVER_K;
        wl.output_write_tiles = 2;
        auto cycles_soh = model_2_7.DPU(wl, info);
        check_for_noError(cycles_soh, wl, "SOK" + t_header + info);

        cycles_soh = static_cast<decltype(cycles_soh)>(cycles_soh * strategy_scale);

        const std::int32_t delta = std::abs((std::int32_t)cycles_clu - (std::int32_t)cycles_soh);
        const std::int32_t maxDelta = static_cast<std::int32_t>(cycles_clu * tolerance_SOK);

        const auto deltapercent = ((float)delta / cycles_clu) * 100;

        // EXPECT_EQ(cycles_clu, cycles_soh) << t_header << " Cost not similar enough.\n"
        //                                   << "cost clustering: " << cycles_clu << "\n"
        //                                   << "cost soh       : " << cycles_soh << "\n"
        //                                   << wl_base;
        EXPECT_LE(delta, maxDelta) << t_header << " Cost not similar enough.\n"
                                   << "cost clustering: " << cycles_clu << "\n"
                                   << "cost sok       : " << cycles_soh << "\n"
                                   << "delta%         : " << deltapercent << "\n"
                                   << wl_base;
        std::cout << "\n--------- END TEST : " << t_header << " cost clustering: " << cycles_clu
                  << ", cost sok: " << cycles_soh << ", delta%: " << (int)(deltapercent) << " % \n";
    }
};

TEST_F(TestSplitMethodsComparisons, Convolution_3x3) {
    const DPUWorkload tst_refH16{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH20{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH38{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    {
        std::string tst = " 3x3H16 ";
        DPUWorkload wl{tst_refH16};
        check_Cluster_vs_SOH(wl, tst);
    }

    {
        std::string tst = " 3x3H19 ";
        DPUWorkload wl{tst_refH19};
        check_Cluster_vs_SOH(wl, tst);
    }
    check_Cluster_vs_SOH(tst_refH38, " 3x3H38 ");
    check_Cluster_vs_SOH(tst_refH20, " 3x3H20 ");
}

TEST_F(TestSplitMethodsComparisons, Convolution_5x5) {
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 5},                                               // kernels
            {1, 1},                                               // strides
            {2, 2, 2, 2},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH20{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 5},                                               // kernels
            {1, 1},                                               // strides
            {2, 2, 2, 2},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH40{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {5, 5},                                               // kernels
            {1, 1},                                               // strides
            {2, 2, 2, 2},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    {
        std::string tst = " 5x5H19 ";
        check_Cluster_vs_SOH(tst_refH19, tst);
    }

    {
        std::string tst = " 5x5H20 ";
        check_Cluster_vs_SOH(tst_refH20, tst);
    }

    {
        std::string tst = " 5x5H40 ";
        check_Cluster_vs_SOH(tst_refH40, tst);
    }
}

TEST_F(TestSplitMethodsComparisons, Convolution_1x1) {
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                               // kernels
            {1, 1},                                               // strides
            {0, 0, 0, 0},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH20{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 20, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                               // kernels
            {1, 1},                                               // strides
            {0, 0, 0, 0},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH40{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 40, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                               // kernels
            {1, 1},                                               // strides
            {0, 0, 0, 0},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    {
        std::string tst = " 1x1H19 ";
        check_Cluster_vs_SOH(tst_refH19, tst);
    }

    {
        std::string tst = " 1x1H20 ";
        check_Cluster_vs_SOH(tst_refH20, tst);
    }

    {
        std::string tst = " 1x1H40 ";
        check_Cluster_vs_SOH(tst_refH40, tst);
    }
}

TEST_F(TestSplitMethodsComparisons, Convolution_3x3_SOK) {
    const DPUWorkload tst_refH16{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 16, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH19{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 19, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH38{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 128, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };
    const DPUWorkload tst_refH38C64{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                              // kernels
            {1, 1},                                              // strides
            {1, 1, 1, 1},                                        // padding
            ExecutionMode::CUBOID_16x16,                         // execution mode
    };

    const DPUWorkload tst_refH38C256{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 38, 256, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 38, 256, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                               // kernels
            {1, 1},                                               // strides
            {1, 1, 1, 1},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    check_Cluster_vs_SOK(tst_refH16, " 3x3H16 ");
    check_Cluster_vs_SOK(tst_refH19, " 3x3H19 ");
    check_Cluster_vs_SOK(tst_refH38, " 3x3H38C128 ");

    check_Cluster_vs_SOK(tst_refH38C64, " 3x3H38C64 ");
    check_Cluster_vs_SOK(tst_refH38C256, " 3x3H38C256 ");
}

TEST_F(TestSplitMethodsComparisons, Convolution_11x11) {
    const DPUWorkload tst_refH49{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 49, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 49, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {11, 11},                                            // kernels
            {1, 1},                                              // strides
            {5, 5, 5, 5},                                        // padding
            ExecutionMode::CUBOID_16x16,                         // execution mode
    };
    const DPUWorkload tst_refH50{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 50, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 50, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {11, 11},                                            // kernels
            {1, 1},                                              // strides
            {5, 5, 5, 5},                                        // padding
            ExecutionMode::CUBOID_16x16,                         // execution mode
    };
    const DPUWorkload tst_refH100{
            VPUDevice::VPU_2_7,
            Operation::CONVOLUTION,
            {VPUTensor(60, 100, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUTensor(60, 100, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {11, 11},                                             // kernels
            {1, 1},                                               // strides
            {5, 5, 5, 5},                                         // padding
            ExecutionMode::CUBOID_16x16,                          // execution mode
    };

    check_Cluster_vs_SOH(tst_refH49, " 11x11 H 49");
    check_Cluster_vs_SOH(tst_refH50, " 11x11 H 50");
    check_Cluster_vs_SOH(tst_refH100, " 11x11 H 100");
}

TEST_F(TestCostModel, TestDPUVPU27ModelIC_4_16_32) {
    VPUNN::DPUWorkload wl0_prototype = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 4, 1, VPUNN::DataType::UINT8)},   // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::DPUWorkload wl1_prototype = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };
    VPUNN::DPUWorkload wl2_prototype = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 32, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1, 1, 1},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

    VPUNN::DPUWorkload wl0 = wl0_prototype;
    VPUNN::DPUWorkload wl1 = wl1_prototype;
    VPUNN::DPUWorkload wl2 = wl2_prototype;

    ASSERT_TRUE(model_2_7.nn_initialized())
            << "Model not loaded, might be due to file location: " << VPU_2_7_MODEL_PATH << std::endl;

    // wl0 < wl1 && wl0 < wl2

    {
        CyclesInterfaceType wl0_cycles = model_2_7.DPU(wl0);
        EXPECT_FALSE(Cycles::isErrorCode(wl0_cycles));
        CyclesInterfaceType wl1_cycles = model_2_7.DPU(wl1);
        EXPECT_FALSE(Cycles::isErrorCode(wl1_cycles));
        CyclesInterfaceType wl2_cycles = model_2_7.DPU(wl2);
        EXPECT_FALSE(Cycles::isErrorCode(wl2_cycles));

        EXPECT_LT(wl0_cycles, wl1_cycles);
        EXPECT_LT(wl0_cycles, wl2_cycles);

        std::cout << "On Execution Mode Cuboid 16x16\nIC = 4:" << wl0_cycles << "\nIC = 16:" << wl1_cycles
                  << "\nIC = 32:" << wl2_cycles;
    }

    wl0.execution_order = wl1.execution_order = wl2.execution_order = VPUNN::ExecutionMode::CUBOID_8x16;

    {
        CyclesInterfaceType wl0_cycles = model_2_7.DPU(wl0);
        EXPECT_FALSE(Cycles::isErrorCode(wl0_cycles));
        CyclesInterfaceType wl1_cycles = model_2_7.DPU(wl1);
        EXPECT_FALSE(Cycles::isErrorCode(wl1_cycles));
        CyclesInterfaceType wl2_cycles = model_2_7.DPU(wl2);
        EXPECT_FALSE(Cycles::isErrorCode(wl2_cycles));

        EXPECT_LT(wl0_cycles, wl1_cycles);
        EXPECT_LT(wl0_cycles, wl2_cycles);

        std::cout << "On Execution Mode Cuboid 8x16\nIC = 4:" << wl0_cycles << "\nIC = 16:" << wl1_cycles
                  << "\nIC = 32:" << wl2_cycles;
    }
    wl0.execution_order = wl1.execution_order = wl2.execution_order = VPUNN::ExecutionMode::CUBOID_4x16;

    {
        CyclesInterfaceType wl0_cycles = model_2_7.DPU(wl0);
        EXPECT_FALSE(Cycles::isErrorCode(wl0_cycles));
        CyclesInterfaceType wl1_cycles = model_2_7.DPU(wl1);
        EXPECT_FALSE(Cycles::isErrorCode(wl1_cycles));
        CyclesInterfaceType wl2_cycles = model_2_7.DPU(wl2);
        EXPECT_FALSE(Cycles::isErrorCode(wl2_cycles));

        EXPECT_LT(wl0_cycles, wl1_cycles);
        EXPECT_LT(wl0_cycles, wl2_cycles);

        std::cout << "On Execution Mode Cuboid 4x16\nIC = 4:" << wl0_cycles << "\nIC = 16:" << wl1_cycles
                  << "\nIC = 32:" << wl2_cycles;
    }
}

TEST_F(TestCostModel, Compressed_CONV_Sanity_test_VPU27) {
    const VPUNN::DPUWorkload wl_ref = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled

    };
    const VPUNN::DPUWorkload wl_ref_equiv = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled

    };

    {
        VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

        {
            VPUNN::DPUWorkload wl{wl_ref};
            VPUNN::DPUWorkload wl_equiv{wl_ref_equiv};

            std::string info_raw, info_equiv;

            auto cycles_raw = model_2_7.DPU(wl, info_raw);  // will change
            auto cycles_equiv = model_2_7.DPU(wl_equiv, info_equiv);

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
}

TEST_F(TestCostModel, Compressed_CONV_Sanity_test_VPU27_IC1_special) {
    const VPUNN::DPUWorkload wl_ref = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled

    };
    const VPUNN::DPUWorkload wl_ref_equiv = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            false,                                                     // weight_sparsity_enabled

    };

    {
        VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

        {
            VPUNN::DPUWorkload wl{wl_ref};
            VPUNN::DPUWorkload wl_equiv{wl_ref_equiv};

            std::string info_raw, info_equiv;

            auto cycles_raw = model_2_7.DPU(wl, info_raw);  // will change
            auto cycles_equiv = model_2_7.DPU(wl_equiv, info_equiv);

            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_raw))
                    << "ERROR code received: " << cycles_raw << " : " << Cycles::toErrorText(cycles_raw)
                    << "\n INFO: " << wl << info_raw << std::endl;
            EXPECT_TRUE(VPUNN::Cycles::isErrorCode(cycles_equiv))
                    << "ERROR code received: " << cycles_equiv << " : " << Cycles::toErrorText(cycles_equiv)
                    << "\n INFO: " << wl_equiv << info_equiv << std::endl;
            ;

            EXPECT_EQ(cycles_raw, cycles_equiv) << wl << wl_equiv << std::endl;
            EXPECT_TRUE((cycles_raw != 0) && (cycles_equiv != 0));
        }
    }
}
TEST_F(TestCostModel, Compressed_CONV_Sanity_test_VPU27_sparse) {
    const VPUNN::DPUWorkload wl_ref = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            true,                                                      // weight_sparsity_enabled

    };
    const VPUNN::DPUWorkload wl_ref_equiv = {
            VPUNN::VPUDevice::VPU_2_7,
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
            {VPUNN::Swizzling::KEY_0, VPUNN::Swizzling::KEY_0},        // input_swizzling
            {VPUNN::Swizzling::KEY_0},                                 // output_swizzling
            1,                                                         // output_write_tiles
            {0, 0, 0, 0},                                              // offsets
            VPUNN::ISIStrategy::CLUSTERING,                            // isi_strategy
            true,                                                      // weight_sparsity_enabled

    };

    {
        VPUNN::VPUCostModel model_2_7{VPUNN::VPUCostModel(VPU_2_7_MODEL_PATH)};

        {
            VPUNN::DPUWorkload wl{wl_ref};
            VPUNN::DPUWorkload wl_equiv{wl_ref_equiv};

            std::string info_raw, info_equiv;

            auto cycles_raw = model_2_7.DPU(wl, info_raw);  // will change
            auto cycles_equiv = model_2_7.DPU(wl_equiv, info_equiv);

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
}

}  // namespace VPUNN_unit_tests