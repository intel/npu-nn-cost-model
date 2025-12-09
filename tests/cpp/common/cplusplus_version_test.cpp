// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.


#include <optional>
#include <variant>

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestCplusplusVersion : public ::testing::Test {};

// here we verify if our C++ compiler version is 17 or newer
TEST_F(TestCplusplusVersion, CompilerVersion_Test) {
    // _MSC_VER is a predefined macro by the MSVC compiler that indicates the compiler version
    // this macro is used to identify the specific version of the MSVC compiler that is compiling the source code
    // the value of _MSC_VER is an integer number that changes with each new version or update of the Visual C++
    // compiler example of _MSC_VER values for C++ versions: C++ 17.0 - 1930  ;  C++ 17.1 - 1931  ; ...  C++ 17.9 - 1939
    // more info here:
    // https://learn.microsoft.com/en-us/cpp/overview/compiler-versions?view=msvc-170#service-releases-starting-with-visual-studio-2017
    // https://hackingcpp.com/cpp/std/macro_cplusplus.html

#if defined(_MSC_VER)
    ASSERT_GE(_MSC_VER, 1930);  // we verify that the version is 17.0 or newer
#elif defined(__GNUC__)         // gcc
    ASSERT_GE(__cplusplus, 201703L);
#elif defined(__clang__)        // clang
    ASSERT_GE(__cplusplus, 201703L);
#endif
}

TEST_F(TestCplusplusVersion, Test_C_plus_plus_17_features) {
    // optional
    std::optional<int> optionalValue;
    // testing if the optional does not have a value assigned
    EXPECT_FALSE(optionalValue.has_value());
    optionalValue = 2;
    EXPECT_TRUE(optionalValue.has_value());  // testing if the optional have a value assigned
    EXPECT_EQ(*optionalValue, 2);

    // variant
    std::variant<int, double> variant_val;
    // variant holds an int
    variant_val = 5;
    EXPECT_EQ(std::get<int>(variant_val), 5);
    // variant holds a double
    variant_val = 4.5;
    EXPECT_EQ(std::get<double>(variant_val), 4.5);

    // if with initializer
    int a = 5;
    if (int b = a * 2; b > 5) {
        EXPECT_TRUE(true);  // verify if b is greater 5
    }

    // lambda using constexpr
    auto lambda_getN = [](int n) constexpr {
        return n;
    };
    EXPECT_EQ(lambda_getN(5), 5);
}
}  // namespace VPUNN_unit_tests