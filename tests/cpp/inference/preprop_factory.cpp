// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "inference/preprop_factory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {

class RuntimeProcessingFactoryTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Test basic cases of normal versions
TEST_F(RuntimeProcessingFactoryTest, SimpleChecks) {
    const VPUNN::RuntimeProcessingFactory factory;
    {
        int v = -1;
        //EXPECT_EQ(factory.exists_preprocessing(v = 0), true) << "Must be present v: " << v;
        EXPECT_EQ(factory.exists_preprocessing(v = 1), true) << "Must be present v: " << v;
        EXPECT_EQ(factory.exists_preprocessing(v = (int)VPUNN::NNVersions::VERSION_10_ENUMS_SAME), true)
                << "Must be present v: " << v;
        EXPECT_EQ(factory.exists_preprocessing(v = (int)VPUNN::NNVersions::VERSION_11_NPU40), true)
                << "Must be present v: " << v;
        EXPECT_EQ(factory.exists_preprocessing(v = (int)VPUNN::NNVersions::VERSION_11_VPU27_BETA), true)
                << "Must be present v: " << v;

        EXPECT_EQ(factory.exists_preprocessing(v = 2), false) << "Must be NOT present v: " << v;

        EXPECT_EQ(factory.exists_preprocessing(v = -5), false) << "Must be NOT present v: " << v;
    }

    {
        int v = (int)VPUNN::NNVersions::VERSION_01_BASE;
        ASSERT_NO_THROW(factory.make_preprocessing(v););
        auto& pp = factory.make_preprocessing(v);
        EXPECT_EQ(pp.interface_version(), v);
    }
    {
        int v = (int)VPUNN::NNVersions::VERSION_10_ENUMS_SAME;
        ASSERT_NO_THROW(factory.make_preprocessing(v););
        auto& pp = factory.make_preprocessing(v);
        EXPECT_EQ(pp.interface_version(), v);
    }
    {
        int v = (int)VPUNN::NNVersions::VERSION_11_VPU27_BETA;
        ASSERT_NO_THROW(factory.make_preprocessing(v););
        auto& pp = factory.make_preprocessing(v);
        EXPECT_EQ(pp.interface_version(), v);
    }
    {
        int v = (int)VPUNN::NNVersions::VERSION_11_NPU40;
        ASSERT_NO_THROW(factory.make_preprocessing(v););
        auto& pp = factory.make_preprocessing(v);
        EXPECT_EQ(pp.interface_version(), v);
    }

    {
        int v = 2;
        ASSERT_ANY_THROW(factory.make_preprocessing(v););
        // auto& pp = factory.make_preprocessing(v);
        // EXPECT_EQ(pp.interface_version(), v);
    }
}

}  // namespace VPUNN_unit_tests
