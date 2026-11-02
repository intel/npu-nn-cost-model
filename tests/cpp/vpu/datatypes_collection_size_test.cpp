// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/datatype_collection_size.h"
#include "vpu/dpu_dtypes_category_info.h"
#include "vpu/dpu_dtypes_dimension_info.h"

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common/common_helpers.h"

#include <optional>
#include <variant>

#include "vpu/dpu_types.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class DataTypes_Collection_Size_Test : public ::testing::Test {};
TEST_F(DataTypes_Collection_Size_Test, Dtype_to_bytes_Test) {
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT1), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT1), 1);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT2), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT2), 1);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT4), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT4), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::FLOAT4), 1);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT8), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT8), 1);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::BFLOAT16), 2);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::FLOAT16), 2);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT16), 2);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT16), 2);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::BF8), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::HF8), 1);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT32), 4);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT32), 4);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::FLOAT32), 4);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::INT64), 8);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::UINT64), 8);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bytes(DataType::FLOAT64), 8);
}

TEST_F(DataTypes_Collection_Size_Test, Dtype_to_bits_Test) {
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT1), 1);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT1), 1);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT2), 2);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT2), 2);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT4), 4);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT4), 4);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::FLOAT4), 4);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT8), 8);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT8), 8);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::BFLOAT16), 16);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::FLOAT16), 16);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT16), 16);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT16), 16);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::BF8), 8);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::HF8), 8);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT32), 32);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT32), 32);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::FLOAT32), 32);

    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::INT64), 64);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::UINT64), 64);
    EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(DataType::FLOAT64), 64);
}

TEST_F(DataTypes_Collection_Size_Test, Number_of_DTypes_per_byte) {
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT1), 8);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT1), 8);

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT2), 4);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT2), 4);

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT4), 2);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT4), 2);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::FLOAT4), 2);

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT8), 1);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT8), 1);

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::BF8), 1);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::HF8), 1);

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::BFLOAT16), 0);  //?
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::FLOAT16), 0);   //?

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT16), 0);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT16), 0);

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT32), 0);    //?
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT32), 0);   //?
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::FLOAT32), 0);  //?

    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::INT64), 0);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::UINT64), 0);
    EXPECT_EQ(DTypeDimensionInfo::types_per_byte(DataType::FLOAT64), 0);
}

// Test is_same_datatype_footprint function
TEST_F(DataTypes_Collection_Size_Test, IsSameDatatypeFootprint) {
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT16, DataType::FLOAT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::BFLOAT16, DataType::BFLOAT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT4, DataType::UINT4));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT4, DataType::INT4));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::FLOAT4));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT2, DataType::UINT2));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT2, DataType::INT2));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT1, DataType::UINT1));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT1, DataType::INT1));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT8, DataType::UINT8));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT8, DataType::INT8));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::BF8, DataType::BF8));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::HF8, DataType::HF8));

    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT16, DataType::BFLOAT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT4, DataType::INT4));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT2, DataType::INT2));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT1, DataType::INT1));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT8, DataType::INT8));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::BF8, DataType::HF8));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::BF8, DataType::INT8));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::INT4));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::UINT4));

    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT32, DataType::INT32));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT32, DataType::UINT32));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT64, DataType::INT64));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT64, DataType::UINT64));

    // Additional test cases with expect false
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::HF8, DataType::BFLOAT16));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT4, DataType::INT8));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT2, DataType::INT4));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT1, DataType::INT2));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT8, DataType::INT1));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::BF8, DataType::FLOAT16));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT32, DataType::FLOAT16));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT32, DataType::FLOAT16));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT32, DataType::INT8));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::INT8));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::FLOAT16));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::FLOAT32));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::INT2));
    EXPECT_FALSE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::FLOAT4, DataType::INT1));

    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT16, DataType::UINT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT16, DataType::INT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT16, DataType::INT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::INT16, DataType::FLOAT16));
    EXPECT_TRUE(DTypeDimensionInfo::is_same_datatype_footprint(DataType::UINT16, DataType::FLOAT16));
}

TEST_F(DataTypes_Collection_Size_Test, Compute_elements_size_in_bytes_Test) {
    {  // 1byte
        auto test1byte = [](const DataType dt) {
            EXPECT_EQ(DTypeDimensionInfo::types_per_byte(dt), 1);

            EXPECT_EQ(compute_size_in_bytes(-1, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(-10, dt), 0) << "type:" << DataType_ToText.at((int)dt);

            EXPECT_EQ(compute_size_in_bytes(0, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(1, dt), 1);
            EXPECT_EQ(compute_size_in_bytes(2, dt), 2);
            EXPECT_EQ(compute_size_in_bytes(3, dt), 3);
            EXPECT_EQ(compute_size_in_bytes(128, dt), 128);
            EXPECT_EQ(compute_size_in_bytes(256, dt), 256);
            EXPECT_EQ(compute_size_in_bytes(125436, dt), 125436);
        };

        test1byte(DataType::INT8);
        test1byte(DataType::UINT8);
        test1byte(DataType::BF8);
        test1byte(DataType::HF8);
    }

    {  // 2bytes
        auto test2byte = [](const DataType dt) {
            EXPECT_EQ(DTypeDimensionInfo::types_per_byte(dt), 0);
            EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(dt), 16);

            EXPECT_EQ(compute_size_in_bytes(-1, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(-10, dt), 0) << "type:" << DataType_ToText.at((int)dt);

            EXPECT_EQ(compute_size_in_bytes(0, dt), 0);
            EXPECT_EQ(compute_size_in_bytes(1, dt), 2);
            EXPECT_EQ(compute_size_in_bytes(2, dt), 4);
            EXPECT_EQ(compute_size_in_bytes(3, dt), 6);
            EXPECT_EQ(compute_size_in_bytes(63, dt), 126);
            EXPECT_EQ(compute_size_in_bytes(128, dt), 256);
            EXPECT_EQ(compute_size_in_bytes(256, dt), 512);
            EXPECT_EQ(compute_size_in_bytes(125436, dt), 125436 * 2);
        };

        test2byte(DataType::FLOAT16);
        test2byte(DataType::BFLOAT16);
        test2byte(DataType::INT16);
        test2byte(DataType::UINT16);
    }

    {  // 0.5 bytes, half a byte
        auto testhalfbyte = [](const DataType dt) {
            EXPECT_EQ(DTypeDimensionInfo::types_per_byte(dt), 2);
            EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(dt), 4);

            EXPECT_EQ(compute_size_in_bytes(-1, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(-10, dt), 0) << "type:" << DataType_ToText.at((int)dt);

            EXPECT_EQ(compute_size_in_bytes(0, dt), 0);
            EXPECT_EQ(compute_size_in_bytes(1, dt), 1);
            EXPECT_EQ(compute_size_in_bytes(2, dt), 1);
            EXPECT_EQ(compute_size_in_bytes(3, dt), 2);
            EXPECT_EQ(compute_size_in_bytes(4, dt), 2);

            EXPECT_EQ(compute_size_in_bytes(63, dt), 32);
            EXPECT_EQ(compute_size_in_bytes(128, dt), 64);
            EXPECT_EQ(compute_size_in_bytes(256, dt), 128);
            EXPECT_EQ(compute_size_in_bytes(125436, dt), 125436 / 2);
        };

        testhalfbyte(DataType::INT4);
        testhalfbyte(DataType::UINT4);
        testhalfbyte(DataType::FLOAT4);
    }

    {  // 0.25 bytes, quarter of a byte
        auto testquarterbyte = [](const DataType dt) {
            EXPECT_EQ(DTypeDimensionInfo::types_per_byte(dt), 4);
            EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(dt), 2);

            EXPECT_EQ(compute_size_in_bytes(-1, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(-10, dt), 0) << "type:" << DataType_ToText.at((int)dt);

            EXPECT_EQ(compute_size_in_bytes(0, dt), 0);
            EXPECT_EQ(compute_size_in_bytes(1, dt), 1);
            EXPECT_EQ(compute_size_in_bytes(2, dt), 1);
            EXPECT_EQ(compute_size_in_bytes(3, dt), 1);
            EXPECT_EQ(compute_size_in_bytes(4, dt), 1);

            EXPECT_EQ(compute_size_in_bytes(5, dt), 2);
            EXPECT_EQ(compute_size_in_bytes(6, dt), 2);
            EXPECT_EQ(compute_size_in_bytes(7, dt), 2);
            EXPECT_EQ(compute_size_in_bytes(8, dt), 2);

            EXPECT_EQ(compute_size_in_bytes(63, dt), 16);
            EXPECT_EQ(compute_size_in_bytes(128, dt), 32);
            EXPECT_EQ(compute_size_in_bytes(256, dt), 64);
            EXPECT_EQ(compute_size_in_bytes(125436, dt), 125436 / 4);
        };

        testquarterbyte(DataType::INT2);
        testquarterbyte(DataType::UINT2);

        EXPECT_EQ(compute_size_in_bytes(133, DataType::INT2), 34);
    }

    {  // 4bytes
        auto testxbyte = [](const DataType dt) {
            EXPECT_EQ(DTypeDimensionInfo::types_per_byte(dt), 0);
            EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(dt), 32);

            EXPECT_EQ(compute_size_in_bytes(-1, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(-10, dt), 0) << "type:" << DataType_ToText.at((int)dt);

            EXPECT_EQ(compute_size_in_bytes(0, dt), 0);
            EXPECT_EQ(compute_size_in_bytes(1, dt), 4);
            EXPECT_EQ(compute_size_in_bytes(2, dt), 8);
            EXPECT_EQ(compute_size_in_bytes(3, dt), 12);
            EXPECT_EQ(compute_size_in_bytes(63, dt), 63 * 4);
            EXPECT_EQ(compute_size_in_bytes(128, dt), 128 * 4);
            EXPECT_EQ(compute_size_in_bytes(256, dt), 256 * 4);
            EXPECT_EQ(compute_size_in_bytes(125436, dt), 125436 * 4);
        };

        testxbyte(DataType::INT32);
        testxbyte(DataType::UINT32);
        testxbyte(DataType::FLOAT32);
    }

    {  // 8bytes
        auto testxbyte = [](const DataType dt) {
            EXPECT_EQ(DTypeDimensionInfo::types_per_byte(dt), 0);
            EXPECT_EQ(DTypeDimensionInfo::dtype_to_bits(dt), 64);

            EXPECT_EQ(compute_size_in_bytes(-1, dt), 0) << "type:" << DataType_ToText.at((int)dt);
            EXPECT_EQ(compute_size_in_bytes(-10, dt), 0) << "type:" << DataType_ToText.at((int)dt);

            EXPECT_EQ(compute_size_in_bytes(0, dt), 0);
            EXPECT_EQ(compute_size_in_bytes(1, dt), 8);
            EXPECT_EQ(compute_size_in_bytes(2, dt), 16);
            EXPECT_EQ(compute_size_in_bytes(3, dt), 24);
            EXPECT_EQ(compute_size_in_bytes(63, dt), 63 * 8);
            EXPECT_EQ(compute_size_in_bytes(128, dt), 128 * 8);
            EXPECT_EQ(compute_size_in_bytes(256, dt), 256 * 8);
            EXPECT_EQ(compute_size_in_bytes(125436, dt), 125436 * 8);
        };

        testxbyte(DataType::INT64);
        testxbyte(DataType::UINT64);
        testxbyte(DataType::FLOAT64);
    }
}

TEST_F(DataTypes_Collection_Size_Test, Compute_elements_number_from_bytes_Test) {
    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::INT8), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::INT8), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::INT8), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(133, DataType::INT8), 133);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::INT2), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::INT2), 4);
    EXPECT_EQ(compute_elements_count_from_bytes(34, DataType::INT2), 34 * 4);

    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::INT4), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::INT4), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::INT4), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::INT4), 2);
    EXPECT_EQ(compute_elements_count_from_bytes(133, DataType::INT4), 133 * 2);

    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::FLOAT4), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::FLOAT4), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::FLOAT4), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::FLOAT4), 2);
    EXPECT_EQ(compute_elements_count_from_bytes(64, DataType::FLOAT4), 128);
    EXPECT_EQ(compute_elements_count_from_bytes(133, DataType::FLOAT4), 133 * 2);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::FLOAT16), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::FLOAT16), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(2, DataType::FLOAT16), 1);
    EXPECT_EQ(compute_elements_count_from_bytes(3, DataType::FLOAT16), 1);

    EXPECT_EQ(compute_elements_count_from_bytes(388, DataType::FLOAT16), 194);
    EXPECT_EQ(compute_elements_count_from_bytes(122, DataType::BFLOAT16), 61);

    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::BFLOAT16), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::BFLOAT16), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::BFLOAT16), 50);
    EXPECT_EQ(compute_elements_count_from_bytes(101, DataType::BFLOAT16), 50);
    EXPECT_EQ(compute_elements_count_from_bytes(102, DataType::BFLOAT16), 51);

    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::INT32), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::FLOAT32), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::FLOAT32), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::FLOAT32), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(2, DataType::FLOAT32), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(3, DataType::FLOAT32), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(4, DataType::FLOAT32), 1);

    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::FLOAT32), 25);
    EXPECT_EQ(compute_elements_count_from_bytes(101, DataType::FLOAT32), 25);
    EXPECT_EQ(compute_elements_count_from_bytes(102, DataType::FLOAT32), 25);
    EXPECT_EQ(compute_elements_count_from_bytes(103, DataType::FLOAT32), 25);
    EXPECT_EQ(compute_elements_count_from_bytes(104, DataType::FLOAT32), 26);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::FLOAT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::FLOAT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(7, DataType::FLOAT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(8, DataType::FLOAT64), 1);
    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::FLOAT64), 12);
    EXPECT_EQ(compute_elements_count_from_bytes(104, DataType::FLOAT64), 13);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::INT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::INT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(7, DataType::INT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(8, DataType::INT64), 1);
    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::INT64), 12);
    EXPECT_EQ(compute_elements_count_from_bytes(104, DataType::INT64), 13);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::UINT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(1, DataType::UINT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(7, DataType::UINT64), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(8, DataType::UINT64), 1);
    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::UINT64), 12);
    EXPECT_EQ(compute_elements_count_from_bytes(104, DataType::UINT64), 13);

    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::INT16), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::INT16), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::INT16), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::INT16), 50);
    EXPECT_EQ(compute_elements_count_from_bytes(133, DataType::INT16), 66);
}

TEST_F(DataTypes_Collection_Size_Test, Dtype_to_bits_and_Dtype_to_bytes_DefaultCase) {
    EXPECT_EQ(-1, DTypeDimensionInfo::dtype_to_bits(static_cast<DataType>(100)));
    EXPECT_EQ(-1, DTypeDimensionInfo::dtype_to_bytes(static_cast<DataType>(100)));
}

TEST_F(DataTypes_Collection_Size_Test, DatatypeFamiliesClassification) {
    struct FamilyCase {
        DataType dtype;
        bool any_float;
        bool any_int;
        bool fp16_family;
        bool fp8_family;
        bool fp32_family;
        bool fp64_family;
        bool i16_family;
        bool i8_family;
        bool i32_family;
        bool i64_family;
    };

    const std::array<FamilyCase, 21> cases{{
            //             dtype              any_float any_int fp16   fp8    fp32   fp64   i16    i8     i32    i64
            {DataType::UINT8, false, true, false, false, false, false, false, true, false, false},
            {DataType::INT8, false, true, false, false, false, false, false, true, false, false},
            {DataType::FLOAT16, true, false, true, false, false, false, false, false, false, false},
            {DataType::BFLOAT16, true, false, true, false, false, false, false, false, false, false},
            {DataType::BF8, true, false, false, true, false, false, false, false, false, false},
            {DataType::HF8, true, false, false, true, false, false, false, false, false, false},
            {DataType::UINT4, false, true, false, false, false, false, false, true, false, false},
            {DataType::INT4, false, true, false, false, false, false, false, true, false, false},
            {DataType::UINT2, false, true, false, false, false, false, false, true, false, false},
            {DataType::INT2, false, true, false, false, false, false, false, true, false, false},
            {DataType::UINT1, false, true, false, false, false, false, false, true, false, false},
            {DataType::INT1, false, true, false, false, false, false, false, true, false, false},
            {DataType::INT32, false, true, false, false, false, false, false, false, true, false},
            {DataType::FLOAT32, true, false, false, false, true, false, false, false, false, false},
            {DataType::UINT16, false, true, false, false, false, false, true, false, false, false},
            {DataType::INT16, false, true, false, false, false, false, true, false, false, false},
            {DataType::FLOAT4, true, false, false, false, false, false, false, false, false, false},
            {DataType::INT64, false, true, false, false, false, false, false, false, false, true},
            {DataType::UINT64, false, true, false, false, false, false, false, false, false, true},
            {DataType::FLOAT64, true, false, false, false, false, true, false, false, false, false},
            {DataType::UINT32, false, true, false, false, false, false, false, false, true, false},
    }};

    for (const auto& tc : cases) {
        EXPECT_EQ(DTypeCategoryInfo::is_any_float_dtype(tc.dtype), tc.any_float)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_any_int_dtype(tc.dtype), tc.any_int)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_fp16family_dtype(tc.dtype), tc.fp16_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_fp8family_dtype(tc.dtype), tc.fp8_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_fp32family_dtype(tc.dtype), tc.fp32_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_fp64family_dtype(tc.dtype), tc.fp64_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_i16family_dtype(tc.dtype), tc.i16_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_i8family_dtype(tc.dtype), tc.i8_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_i32family_dtype(tc.dtype), tc.i32_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
        EXPECT_EQ(DTypeCategoryInfo::is_i64family_dtype(tc.dtype), tc.i64_family)
                << DataType_ToText.at(static_cast<int>(tc.dtype));
    }
}

}  // namespace VPUNN_unit_tests