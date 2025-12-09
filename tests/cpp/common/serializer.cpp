// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/serializer.h"

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <filesystem>
#include <chrono>
#include <random>

#include "vpu/vpu_tensor.h"
#include <vpu/validation/data_dpu_operation.h>
#include <vpu_cost_model.h>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUNNSerializerTest : public ::testing::Test {
public:
    using SerializerT = Serializer<FileFormat::CSV>;

protected:
    void SetUp() override {
        set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "TRUE");
        set_env_var("VPUNN_FILE_NAME_POSTFIX", "serialization_test");

        assert(get_env_vars({"ENABLE_VPUNN_DATA_SERIALIZATION"}).at("ENABLE_VPUNN_DATA_SERIALIZATION") == "TRUE");

        serializer = std::make_unique<SerializerT>();

        // 1. create empty test file
        std::ofstream file(empty_filename + get_extension(serializer->get_format()),
                           std::ofstream::out | std::ofstream::trunc);
        file.close();

        // 2. create test file with header
        std::ofstream file1(empty_filename_w_header + get_extension(serializer->get_format()),
                            std::ofstream::out | std::ofstream::trunc);
        file1 << "col1,col2,col3,col5" << std::endl;
        file1.close();
    }

    void TearDown() override {
        // Remove any files created during the test
        if (serializer->get_file_stream().is_open()) {
            serializer->reset();
        }
        std::filesystem::remove(serializer->get_file_name());
        std::filesystem::remove(empty_filename + get_extension(serializer->get_format()));
        std::filesystem::remove(empty_filename_w_header + get_extension(serializer->get_format()));

        set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
        set_env_var("VPUNN_FILE_NAME_POSTFIX", "");
    }

    std::unique_ptr<SerializerT> serializer;  // why ptr?

    std::string empty_filename = "serializer_test_file_empty";
    std::string empty_filename_w_header = "serializer_test_file_w_header";
};

TEST_F(VPUNNSerializerTest, Check_if_Enabled) {
    EXPECT_TRUE(serializer->is_serialization_enabled());

    set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
    serializer = std::make_unique<SerializerT>(true);  // check force enable
    EXPECT_TRUE(serializer->is_serialization_enabled());
}

TEST_F(VPUNNSerializerTest, Check_if_Disabled) {
    set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
    serializer = std::make_unique<SerializerT>(false);
    EXPECT_FALSE(serializer->is_serialization_enabled());
}

TEST_F(VPUNNSerializerTest, Init_Check_is_Initialized) {
    std::string filename = "test_file";

    EXPECT_FALSE(serializer->is_initialized());

    serializer->initialize(filename, FileMode::READ_WRITE, {});
    EXPECT_FALSE(serializer->is_initialized());

    serializer->initialize(filename, FileMode::READ_WRITE, {"int value", "float value"});
    EXPECT_TRUE(serializer->is_initialized());
}

TEST_F(VPUNNSerializerTest, Init_From_ExistingEmptyFile) {
    EXPECT_TRUE(std::filesystem::exists(empty_filename + get_extension(serializer->get_format())));

    serializer->initialize(empty_filename, FileMode::READ_WRITE, {"int value", "float value", "int value"});

    EXPECT_TRUE(serializer->is_initialized());

    EXPECT_EQ(serializer->get_file_name(), empty_filename + get_extension(serializer->get_format()));

    auto field_names = serializer->get_field_names();
    EXPECT_EQ(field_names.size(), 2);

    for (const auto& field : field_names) {
        EXPECT_TRUE(field == "int value" || field == "float value");
    }
}

TEST_F(VPUNNSerializerTest, Init_From_ExistingFileWHeader) {
    EXPECT_TRUE(std::filesystem::exists(empty_filename_w_header + get_extension(serializer->get_format())));

    serializer->initialize(empty_filename_w_header, FileMode::READ_WRITE, {});
    EXPECT_TRUE(serializer->is_initialized());

    auto field_names = serializer->get_field_names();

    for (const auto& field : field_names) {
        EXPECT_TRUE(field == "col1" || field == "col2" || field == "col3" || field == "col5");
    }
}

TEST_F(VPUNNSerializerTest, Serialize_DpuOperation) {
    serializer->initialize("test_dpu_op_serialize", FileMode::READ_WRITE, DPUOperation::_get_member_names());

    EXPECT_TRUE(serializer->is_initialized());

    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    serializer->serialize(DPUOperation(wl));
    serializer->end();

    auto field_names = serializer->get_field_names();
    std::vector<std::string> expected_field_names = DPUOperation::_get_member_names();

    // EXPECT_EQ(field_names, expected_field_names);
}

TEST_F(VPUNNSerializerTest, Serialize_MultipleCalls) {
    serializer->initialize("test_dpu_op_serialize", FileMode::READ_WRITE, {"col0", "col1"});

    EXPECT_TRUE(serializer->is_initialized());

    serializer->serialize(SerializableField{"col0", 0});
    serializer->serialize(SerializableField{"col1", 1});
    serializer->end();

    serializer->jump_to_beginning();

    SerializableField field0{"col0", -1};
    SerializableField field1{"col1", -1};

    serializer->deserialize(field0, field1);

    EXPECT_EQ(field0.value, 0);
    EXPECT_EQ(field1.value, 1);
}

TEST_F(VPUNNSerializerTest, Serialize_Deserialize_DpuOperation) {
    serializer->initialize("test_dpu_op_deserialize", FileMode::READ_WRITE, DPUOperation::_get_member_names());

    EXPECT_TRUE(serializer->is_initialized());

    const VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_4_0,
            VPUNN::Operation::ELTWISE,
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(16, 16, 64, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {1, 1},                                                     // kernels
            {1, 1},                                                     // strides
            {0, 0, 0, 0},                                               // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    auto orig_dpu_op = DPUOperation(wl);
    serializer->serialize(orig_dpu_op);
    serializer->end();

    serializer->jump_to_beginning();

    DPUOperation deserialized_dpu_op;
    serializer->deserialize(deserialized_dpu_op);

    EXPECT_EQ(orig_dpu_op.device, deserialized_dpu_op.device);
    EXPECT_EQ(orig_dpu_op.operation, deserialized_dpu_op.operation);
    // EXPECT_EQ(orig_dpu_op.input_0, deserialized_dpu_op.input_0);
    // EXPECT_EQ(orig_dpu_op.output_0, deserialized_dpu_op.output_0);
    // EXPECT_EQ(orig_dpu_op.kernel, deserialized_dpu_op.kernel);
    EXPECT_EQ(orig_dpu_op.execution_order, deserialized_dpu_op.execution_order);
}

TEST_F(VPUNNSerializerTest, Serialize_Deserialize_Multi_SerializableField) {
    serializer->initialize(empty_filename_w_header, FileMode::READ_WRITE, {});

    EXPECT_TRUE(serializer->is_initialized());

    SerializableField int_field{"col1", 123};
    SerializableField str_field{"col2", std::string("value")};
    SerializableField bool_field("col3", true);
    SerializableField bool_field_copy("col3", true);
    SerializableField str_field_2{"col5", std::string("valueX me1")};

    serializer->serialize(int_field, str_field, bool_field, str_field_2);
    serializer->end();

    int_field.value = 456;
    str_field.value = "value1";
    bool_field.value = false;
    str_field_2.value = "vaxlue1,mex";

    serializer->serialize(int_field, str_field, bool_field, str_field_2);
    serializer->end();

    str_field_2.value = "reset";

    serializer->jump_to_beginning();
    serializer->deserialize(str_field_2, int_field, str_field, bool_field, bool_field_copy);

    EXPECT_EQ(int_field.value, 123);
    EXPECT_EQ(str_field.value, "value");  // default
    EXPECT_EQ(bool_field.value, true);
    EXPECT_EQ(bool_field_copy.value, true);
    EXPECT_EQ(str_field_2.value, "valueX me1");  // default

    str_field_2.value = "reset2";

    serializer->deserialize(str_field_2, int_field, str_field, bool_field, bool_field_copy);

    EXPECT_EQ(int_field.value, 456);
    EXPECT_EQ(str_field.value, "value1");
    EXPECT_EQ(bool_field.value, false);
    EXPECT_EQ(bool_field_copy.value, false);
    EXPECT_EQ(str_field_2.value, "vaxlue1.mex");  // no comma?
}

TEST_F(VPUNNSerializerTest, CopyRow) {
    serializer->initialize("test_dpu_op_serialize", FileMode::READ_WRITE, {"col2", "col1", "col0", "new_col"});

    EXPECT_TRUE(serializer->is_initialized());

    serializer->serialize(SerializableField{"col0", 0});
    serializer->serialize(SerializableField{"col2", 1});
    serializer->serialize(SerializableField<std::string>{"col1", "hello"});
    serializer->end();

    serializer->jump_to_beginning();

    Series row{};
    serializer->read_row(row);

    EXPECT_EQ(row.at("col0"), "0");
    EXPECT_EQ(row.at("col1"), "hello");
    EXPECT_EQ(row.at("col2"), "1");

    serializer->serialize(row, SerializableField<std::string>{"new_col", "new_data"}, SerializableField{"col0", 123});
    serializer->end();

    serializer->jump_to_beginning();

    row = {};
    serializer->read_row(row);
    serializer->read_row(row);

    EXPECT_EQ(row.at("col0"), "123");  // new value
    EXPECT_EQ(row.at("col1"), "hello");
    EXPECT_EQ(row.at("col2"), "1");
    EXPECT_EQ(row.at("new_col"), "new_data");
}

TEST_F(VPUNNSerializerTest, MultiThreaded_Serialization) {
    const std::string filename = "test_multithreaded_serialize";
    serializer->initialize(filename, FileMode::READ_WRITE, {"thread_id", "value", "2nd_value"});

    ASSERT_TRUE(serializer->is_initialized());

    constexpr int num_threads = 8;
    constexpr int rows_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> ready_count{0};
    std::mutex start_mutex;
    bool start_flag{false};
    std::condition_variable start_cv;

    // Each thread will write its own rows
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Thread-local random number generator
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(1, 5);  // Random delay between 1 and 5 millise

            // Notify that this thread is ready
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                ready_count.fetch_add(1);
                if (ready_count.load() == num_threads) {
                    start_cv.notify_all();  // Notify all threads when all are ready
                }
            }
            
            {
                // Wait for the start signal
                std::unique_lock<std::mutex> lock(start_mutex);
                start_cv.wait(lock, [&]() {
                    return start_flag;
                });
            }

            for (int i = 0; i < rows_per_thread; ++i) {
                serializer->serialize(SerializableField{"thread_id", t}, SerializableField{"value", i});
                std::this_thread::sleep_for(std::chrono::milliseconds(dist(gen)));
                serializer->serialize(SerializableField{"2nd_value", i + 10});
                serializer->end();
            }
        });
    }

    // Wait until all threads are ready
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_cv.wait(lock, [&]() {
            return ready_count.load() == num_threads;
        });
    }

    // Signal all threads to start
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start_flag = true;
        start_cv.notify_all();
    }

    for (auto& th : threads) {
        th.join();
    }

    // Now, read back the file and check all expected rows are present
    serializer->jump_to_beginning();

    std::set<std::pair<int, int>> found;
    auto thread_id_buf = SerializableField{"thread_id", 0};
    auto value_buf = SerializableField{"value", 0};
    auto second_value_buf = SerializableField{"2nd_value", 0};
    while (serializer->deserialize(thread_id_buf, value_buf, second_value_buf)) {
        found.emplace(thread_id_buf.value, value_buf.value);
    }

    // Check that all expected (thread_id, value) pairs are present
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < rows_per_thread; ++i) {
            EXPECT_TRUE(found.count({t, i})) << "Missing row for thread " << t << " value " << i;
        }
    }

    EXPECT_EQ(found.size(), num_threads * rows_per_thread);
}

}  // namespace VPUNN_unit_tests
