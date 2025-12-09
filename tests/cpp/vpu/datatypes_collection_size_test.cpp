// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpu/datatype_collection_size.h"
#include "vpu/dpu_types_info.h"

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
    EXPECT_EQ(dtype_to_bytes(DataType::INT1), 1);
    EXPECT_EQ(dtype_to_bytes(DataType::UINT1), 1);

    EXPECT_EQ(dtype_to_bytes(DataType::INT2), 1);
    EXPECT_EQ(dtype_to_bytes(DataType::UINT2), 1);

    EXPECT_EQ(dtype_to_bytes(DataType::INT4), 1);
    EXPECT_EQ(dtype_to_bytes(DataType::UINT4), 1);

    EXPECT_EQ(dtype_to_bytes(DataType::INT8), 1);
    EXPECT_EQ(dtype_to_bytes(DataType::UINT8), 1);

    EXPECT_EQ(dtype_to_bytes(DataType::BFLOAT16), 2);
    EXPECT_EQ(dtype_to_bytes(DataType::FLOAT16), 2);

    EXPECT_EQ(dtype_to_bytes(DataType::UINT16), 2);
    EXPECT_EQ(dtype_to_bytes(DataType::INT16), 2);

    EXPECT_EQ(dtype_to_bytes(DataType::BF8), 1);
    EXPECT_EQ(dtype_to_bytes(DataType::HF8), 1);

    EXPECT_EQ(dtype_to_bytes(DataType::INT32), 4);
    EXPECT_EQ(dtype_to_bytes(DataType::FLOAT32), 4);
}

TEST_F(DataTypes_Collection_Size_Test, Dtype_to_bits_Test) {
    EXPECT_EQ(dtype_to_bits(DataType::INT1), 1);
    EXPECT_EQ(dtype_to_bits(DataType::UINT1), 1);

    EXPECT_EQ(dtype_to_bits(DataType::INT2), 2);
    EXPECT_EQ(dtype_to_bits(DataType::UINT2), 2);

    EXPECT_EQ(dtype_to_bits(DataType::INT4), 4);
    EXPECT_EQ(dtype_to_bits(DataType::UINT4), 4);

    EXPECT_EQ(dtype_to_bits(DataType::INT8), 8);
    EXPECT_EQ(dtype_to_bits(DataType::UINT8), 8);

    EXPECT_EQ(dtype_to_bits(DataType::BFLOAT16), 16);
    EXPECT_EQ(dtype_to_bits(DataType::FLOAT16), 16);

    EXPECT_EQ(dtype_to_bits(DataType::UINT16), 16);
    EXPECT_EQ(dtype_to_bits(DataType::INT16), 16);

    EXPECT_EQ(dtype_to_bits(DataType::BF8), 8);
    EXPECT_EQ(dtype_to_bits(DataType::HF8), 8);

    EXPECT_EQ(dtype_to_bits(DataType::INT32), 32);
    EXPECT_EQ(dtype_to_bits(DataType::FLOAT32), 32);
}

TEST_F(DataTypes_Collection_Size_Test, Number_of_DTypes_per_byte) {
    EXPECT_EQ(types_per_byte(DataType::INT1), 8);
    EXPECT_EQ(types_per_byte(DataType::UINT1), 8);

    EXPECT_EQ(types_per_byte(DataType::INT2), 4);
    EXPECT_EQ(types_per_byte(DataType::UINT2), 4);

    EXPECT_EQ(types_per_byte(DataType::INT4), 2);
    EXPECT_EQ(types_per_byte(DataType::UINT4), 2);

    EXPECT_EQ(types_per_byte(DataType::INT8), 1);
    EXPECT_EQ(types_per_byte(DataType::UINT8), 1);

    EXPECT_EQ(types_per_byte(DataType::BF8), 1);
    EXPECT_EQ(types_per_byte(DataType::HF8), 1);

    EXPECT_EQ(types_per_byte(DataType::BFLOAT16), 0);  //?
    EXPECT_EQ(types_per_byte(DataType::FLOAT16), 0);   //?

    EXPECT_EQ(types_per_byte(DataType::UINT16), 0);
    EXPECT_EQ(types_per_byte(DataType::INT16), 0);

    EXPECT_EQ(types_per_byte(DataType::INT32), 0);    //?
    EXPECT_EQ(types_per_byte(DataType::FLOAT32), 0);  //?
}

// Test is_same_datatype_footprint function
TEST_F(DataTypes_Collection_Size_Test, IsSameDatatypeFootprint) {
    EXPECT_TRUE(is_same_datatype_footprint(DataType::FLOAT16, DataType::FLOAT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::BFLOAT16, DataType::BFLOAT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT4, DataType::UINT4));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::INT4, DataType::INT4));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT2, DataType::UINT2));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::INT2, DataType::INT2));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT1, DataType::UINT1));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::INT1, DataType::INT1));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT8, DataType::UINT8));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::INT8, DataType::INT8));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::BF8, DataType::BF8));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::HF8, DataType::HF8));

    EXPECT_TRUE(is_same_datatype_footprint(DataType::FLOAT16, DataType::BFLOAT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT4, DataType::INT4));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT2, DataType::INT2));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT1, DataType::INT1));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT8, DataType::INT8));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::BF8, DataType::HF8));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::BF8, DataType::INT8));

    EXPECT_TRUE(is_same_datatype_footprint(DataType::FLOAT32, DataType::INT32));

    // Additional test cases with expect false
    EXPECT_FALSE(is_same_datatype_footprint(DataType::HF8, DataType::BFLOAT16));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::UINT4, DataType::INT8));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::UINT2, DataType::INT4));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::UINT1, DataType::INT2));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::UINT8, DataType::INT1));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::BF8, DataType::FLOAT16));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::INT32, DataType::FLOAT16));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::FLOAT32, DataType::FLOAT16));
    EXPECT_FALSE(is_same_datatype_footprint(DataType::INT32, DataType::INT8));

    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT16, DataType::UINT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::INT16, DataType::INT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT16, DataType::INT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::INT16, DataType::FLOAT16));
    EXPECT_TRUE(is_same_datatype_footprint(DataType::UINT16, DataType::FLOAT16));
}

TEST_F(DataTypes_Collection_Size_Test, Compute_elements_size_in_bytes_Test) {
    {  // 1byte
        auto test1byte = [](const DataType dt) {
            EXPECT_EQ(types_per_byte(dt), 1);

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
            EXPECT_EQ(types_per_byte(dt), 0);
            EXPECT_EQ(dtype_to_bits(dt), 16);

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
            EXPECT_EQ(types_per_byte(dt), 2);
            EXPECT_EQ(dtype_to_bits(dt), 4);

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
    }

    {  // 0.25 bytes, quarter of a byte
        auto testquarterbyte = [](const DataType dt) {
            EXPECT_EQ(types_per_byte(dt), 4);
            EXPECT_EQ(dtype_to_bits(dt), 2);

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
            EXPECT_EQ(types_per_byte(dt), 0);
            EXPECT_EQ(dtype_to_bits(dt), 32);

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
        testxbyte(DataType::FLOAT32);
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

    EXPECT_EQ(compute_elements_count_from_bytes(-1, DataType::INT16), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(-10, DataType::INT16), 0);

    EXPECT_EQ(compute_elements_count_from_bytes(0, DataType::INT16), 0);
    EXPECT_EQ(compute_elements_count_from_bytes(100, DataType::INT16), 50);
    EXPECT_EQ(compute_elements_count_from_bytes(133, DataType::INT16), 66);
}

TEST_F(DataTypes_Collection_Size_Test, Dtype_to_bits_and_Dtype_to_bytes_DefaultCase) {


    EXPECT_EQ(-1, dtype_to_bits(static_cast<DataType>(20)));
    EXPECT_EQ(-1, dtype_to_bytes(static_cast<DataType>(20)));
}

}  // namespace VPUNN_unit_tests