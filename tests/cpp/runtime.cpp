// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "vpunn.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "common_helpers.h"

namespace VPUNN_unit_tests {

class TestRuntime : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

    auto read_a_file(const std::string& filename) const {
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

/// Test cases covering the creation of the Runtime object
TEST_F(TestRuntime, CreationBasicTest) {
    {  // good data
        const std::string vpunn_file = VPU_2_0_MODEL_PATH;
        auto runtime_model_f{VPUNN::Runtime(vpunn_file)};
        EXPECT_TRUE(runtime_model_f.initialized());

        const auto file_content{read_a_file(vpunn_file)};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        auto runtime_model_b{VPUNN::Runtime(file_content.data(), file_content.size(), false)};
        EXPECT_TRUE(runtime_model_b.initialized());

        auto runtime_model_bc{VPUNN::Runtime(file_content.data(), file_content.size(), true)};
        EXPECT_TRUE(runtime_model_bc.initialized());

        EXPECT_EQ(runtime_model_f.input_tensors().size(), runtime_model_b.input_tensors().size());
        EXPECT_EQ(runtime_model_f.output_tensors().size(), runtime_model_b.output_tensors().size());

        EXPECT_EQ(runtime_model_f.input_tensors().size(), runtime_model_bc.input_tensors().size());
        EXPECT_EQ(runtime_model_f.output_tensors().size(), runtime_model_bc.output_tensors().size());

        EXPECT_EQ(runtime_model_f.model_version_info().get_raw_name(),
                  runtime_model_bc.model_version_info().get_raw_name());
        EXPECT_EQ(runtime_model_f.model_version_info().get_raw_name(),
                  runtime_model_b.model_version_info().get_raw_name());
    }
    {  // good data
        const std::string vpunn_file = VPU_2_7_MODEL_PATH;
        auto runtime_model_f{VPUNN::Runtime(vpunn_file)};
        EXPECT_TRUE(runtime_model_f.initialized());

        const auto file_content{read_a_file(vpunn_file)};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        auto runtime_model_b{VPUNN::Runtime(file_content.data(), file_content.size(), false)};
        EXPECT_TRUE(runtime_model_b.initialized());

        auto runtime_model_bc{VPUNN::Runtime(file_content.data(), file_content.size(), true)};
        EXPECT_TRUE(runtime_model_bc.initialized());

        EXPECT_EQ(runtime_model_f.input_tensors().size(), runtime_model_b.input_tensors().size());
        EXPECT_EQ(runtime_model_f.output_tensors().size(), runtime_model_b.output_tensors().size());

        EXPECT_EQ(runtime_model_f.input_tensors().size(), runtime_model_bc.input_tensors().size());
        EXPECT_EQ(runtime_model_f.output_tensors().size(), runtime_model_bc.output_tensors().size());

        EXPECT_EQ(runtime_model_f.model_version_info().get_raw_name(),
                  runtime_model_bc.model_version_info().get_raw_name());
        EXPECT_EQ(runtime_model_f.model_version_info().get_raw_name(),
                  runtime_model_b.model_version_info().get_raw_name());
    }

    {  // garbage data/no file
        const std::string vpunn_file = "NoFileHere.vpunn";
        auto runtime_model_f{VPUNN::Runtime(vpunn_file)};
        EXPECT_FALSE(runtime_model_f.initialized());

        const decltype(read_a_file("")) file_content{'M', 'u', 's', 't', 'h', 'a', 'v', 'e', ' ', '0', '1'};
        ASSERT_GT(file_content.size(), 10) << "Must have some content";

        auto runtime_model_b{VPUNN::Runtime(file_content.data(), file_content.size(), false)};
        EXPECT_FALSE(runtime_model_b.initialized());

        auto runtime_model_bc{VPUNN::Runtime(file_content.data(), file_content.size(), true)};
        EXPECT_FALSE(runtime_model_bc.initialized());
    }
}

}  // namespace VPUNN_unit_tests
