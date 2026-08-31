// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "common/common_helpers.h"
#include "core/serializer.h"
#include "vpu/dma_descriptors.h"
#include "vpu/serialization/dma_cost_serialization_wrapper.h"
#include "vpu/validation/serializable_dma_descriptor.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class SerializableVPUDMADescriptorTest : public ::testing::Test {
protected:
    std::filesystem::path make_temp_csv_path(const std::string& stem) {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(ticks) + ".csv");
    }

    VPUDMADescriptor make_descriptor() {
        VPUDMADescriptor desc{};
        desc.device = VPUDevice::VPU_4_0;
        desc.src_location = MemoryLocation::DRAM;
        desc.dst_location = MemoryLocation::CMX;

        desc.src.dtype = DataType::INT8;
        desc.src.setDimension_OutermostFirst({{2, 8}, {4, 4}, {8, 1}});

        desc.dst.dtype = DataType::FLOAT16;
        desc.dst.setDimension_OutermostFirst({{3, 24}, {6, 4}});

        return desc;
    }

    std::string enum_cell(const VPUDevice value) {
        return enumName<VPUDevice>() + "." + VPUDevice_ToText.at(static_cast<int>(value));
    }

    std::string enum_cell(const DataType value) {
        return enumName<DataType>() + "." + DataType_ToText.at(static_cast<int>(value));
    }

    std::string enum_cell(const MemoryLocation value) {
        return enumName<MemoryLocation>() + "." + MemoryLocation_ToText.at(static_cast<int>(value));
    }
};

TEST_F(SerializableVPUDMADescriptorTest, MemberMap_RoundTrip_PreservesRepresentativeFields) {
    const auto temp_path = make_temp_csv_path("serializable_dma_descriptor_roundtrip");
    { std::ofstream{temp_path}; }  // pre-create so Serializer::initialize() uses this exact path
    ScopedFileCleanup cleanup{temp_path};

    const auto original_desc = make_descriptor();
    const SerializableVPUDMADescriptor original{original_desc};

    Serializer<FileFormat::CSV> serializer{true};
    serializer.initialize(temp_path.string(), FileMode::READ_WRITE, SerializableVPUDMADescriptor::_get_member_names());
    ASSERT_TRUE(serializer.is_initialized());

    serializer.serialize(original);
    serializer.end();

    serializer.jump_to_beginning();
    SerializableVPUDMADescriptor restored{};
    ASSERT_TRUE(serializer.deserialize(restored));

    const VPUDMADescriptor restored_desc = restored.dma_descriptor_data();

    EXPECT_EQ(restored_desc.device, original_desc.device);
    EXPECT_EQ(restored_desc.src.dtype, original_desc.src.dtype);
    EXPECT_EQ(restored_desc.src.num_dims, original_desc.src.num_dims);
    EXPECT_EQ(restored_desc.src.shape[0], original_desc.src.shape[0]);
    EXPECT_EQ(restored_desc.src.shape[1], original_desc.src.shape[1]);
    EXPECT_EQ(restored_desc.src.byte_strides[0], original_desc.src.byte_strides[0]);
    EXPECT_EQ(restored_desc.src.byte_strides[1], original_desc.src.byte_strides[1]);
    EXPECT_EQ(restored_desc.dst.dtype, original_desc.dst.dtype);
    EXPECT_EQ(restored_desc.dst.num_dims, original_desc.dst.num_dims);
    EXPECT_EQ(restored_desc.dst.shape[0], original_desc.dst.shape[0]);
    EXPECT_EQ(restored_desc.dst.byte_strides[0], original_desc.dst.byte_strides[0]);
    EXPECT_EQ(restored_desc.src_location, original_desc.src_location);
    EXPECT_EQ(restored_desc.dst_location, original_desc.dst_location);
}

TEST_F(SerializableVPUDMADescriptorTest, SerializerNames_IncludeDescriptorFieldsThenCostMetadata) {
    const std::size_t descriptor_field_count = static_cast<std::size_t>(7 + (4 * VPU_DMA_MAX_DIMS));
    const std::size_t serializer_field_count = descriptor_field_count + 3;  // vpunn_cycles, cost_source, error_info

    EXPECT_EQ(SerializableVPUDMADescriptor::_get_member_names().size(), descriptor_field_count);
    EXPECT_EQ(SerializableVPUDMADescriptor::get_names_for_serializer().size(), serializer_field_count);

    const std::vector<std::string> expected = {
            "device",       "src_dtype",    "src_num_dims", "src_shape_0",  "src_shape_1",  "src_shape_2",
            "src_shape_3",  "src_shape_4",  "src_shape_5",  "src_stride_0", "src_stride_1", "src_stride_2",
            "src_stride_3", "src_stride_4", "src_stride_5", "dst_dtype",    "dst_num_dims", "dst_shape_0",
            "dst_shape_1",  "dst_shape_2",  "dst_shape_3",  "dst_shape_4",  "dst_shape_5",  "dst_stride_0",
            "dst_stride_1", "dst_stride_2", "dst_stride_3", "dst_stride_4", "dst_stride_5", "src_location",
            "dst_location", "vpunn_cycles", "cost_source",  "error_info",
    };

    EXPECT_EQ(SerializableVPUDMADescriptor::get_names_for_serializer(), expected);
}

TEST_F(SerializableVPUDMADescriptorTest, DMAWrapper_WritesExpectedCSVRow) {
    const auto temp_path = make_temp_csv_path("serializable_dma_descriptor_wrapper");
    { std::ofstream{temp_path}; }  // pre-create so Serializer::initialize() uses this exact path
    ScopedFileCleanup cleanup{temp_path};

    Serializer<FileFormat::CSV> serializer{true};
    serializer.initialize(temp_path.string(), FileMode::READ_WRITE,
                          SerializableVPUDMADescriptor::get_names_for_serializer());
    ASSERT_TRUE(serializer.is_initialized());

    DMACostSerializationWrap<VPUDMADescriptor> wrapper{serializer};
    const VPUDMADescriptor desc{make_descriptor()};

    wrapper.serializeDMAWorkload(desc);
    wrapper.serializeCyclesAndCostInfo_closeLine(123, "dma_cost_test", "all good");

    serializer.jump_to_beginning();
    Series row{};
    ASSERT_TRUE(serializer.read_row(row));

    EXPECT_EQ(row.size(), SerializableVPUDMADescriptor::get_names_for_serializer().size());
    EXPECT_EQ(row.at("device"), enum_cell(VPUDevice::VPU_4_0));
    EXPECT_EQ(row.at("src_dtype"), enum_cell(DataType::INT8));
    EXPECT_EQ(row.at("dst_dtype"), enum_cell(DataType::FLOAT16));
    EXPECT_EQ(row.at("src_num_dims"), "3");
    EXPECT_EQ(row.at("dst_num_dims"), "2");
    EXPECT_EQ(row.at("src_shape_0"), "2");
    EXPECT_EQ(row.at("src_stride_1"), "4");
    EXPECT_EQ(row.at("dst_shape_0"), "3");
    EXPECT_EQ(row.at("dst_stride_0"), "24");
    EXPECT_EQ(row.at("src_location"), enum_cell(MemoryLocation::DRAM));
    EXPECT_EQ(row.at("dst_location"), enum_cell(MemoryLocation::CMX));
    EXPECT_EQ(row.at("vpunn_cycles"), "123");
    EXPECT_EQ(row.at("cost_source"), "dma_cost_test");
    EXPECT_EQ(row.at("error_info"), "all good");
}

TEST_F(SerializableVPUDMADescriptorTest, MemberMap_SerializesUnusedDimsAsZero) {
    const auto temp_path = make_temp_csv_path("serializable_dma_descriptor_zero_dims");
    { std::ofstream{temp_path}; }  // pre-create so Serializer::initialize() uses this exact path
    ScopedFileCleanup cleanup{temp_path};

    const SerializableVPUDMADescriptor serializable{make_descriptor()};

    Serializer<FileFormat::CSV> serializer{true};
    serializer.initialize(temp_path.string(), FileMode::READ_WRITE, SerializableVPUDMADescriptor::_get_member_names());
    ASSERT_TRUE(serializer.is_initialized());

    serializer.serialize(serializable);
    serializer.end();

    serializer.jump_to_beginning();
    Series row{};
    ASSERT_TRUE(serializer.read_row(row));

    for (int i = 3; i < VPU_DMA_MAX_DIMS; ++i) {
        EXPECT_EQ(row.at("src_shape_" + std::to_string(i)), "0");
        EXPECT_EQ(row.at("src_stride_" + std::to_string(i)), "0");
    }

    for (int i = 2; i < VPU_DMA_MAX_DIMS; ++i) {
        EXPECT_EQ(row.at("dst_shape_" + std::to_string(i)), "0");
        EXPECT_EQ(row.at("dst_stride_" + std::to_string(i)), "0");
    }
}

}  // namespace VPUNN_unit_tests