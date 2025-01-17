// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/logger.h"
#include <gtest/gtest.h>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUNNLoggerTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
        Logger::clear2ndlog();
        Logger::deactivate2ndlog();
    }
};

// Demonstrate some basic assertions.
TEST_F(VPUNNLoggerTest, BasicAssertions) {
    Logger::initialize();

    std::string output;

#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(Logger::enabled(), true);
    EXPECT_EQ(toString(Logger::level()), "WARNING");
#else
    EXPECT_EQ(Logger::enabled(), false);
    EXPECT_EQ(toString(Logger::level()), "NONE");
#endif

    // TEST ERROR

    testing::internal::CaptureStdout();
    Logger::error() << "error";
    output = testing::internal::GetCapturedStdout();
#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(output, "[VPUNN ERROR]: error\n");
#else
    EXPECT_EQ(output, "");
#endif

    testing::internal::CaptureStdout();
    Logger::fatal() << "fatal";
    output = testing::internal::GetCapturedStdout();
#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(output, "[VPUNN FATAL]: fatal\n");
#else
    EXPECT_EQ(output, "");
#endif

    testing::internal::CaptureStdout();
    Logger::warning() << "warning";
    output = testing::internal::GetCapturedStdout();
#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(output, "[VPUNN WARNING]: warning\n");
#else
    EXPECT_EQ(output, "");
#endif

    testing::internal::CaptureStdout();
    Logger::info() << "info";
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "");

    testing::internal::CaptureStdout();
    Logger::debug() << "debug";
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "");

    testing::internal::CaptureStdout();
    Logger::trace() << "trace";
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "");
}

TEST_F(VPUNNLoggerTest, Alternative2ndOutput_present) {
    Logger::initialize();

    {
        std::string i0 = Logger::get2ndlog();
        EXPECT_EQ(i0, "");  // nothing  in the beginning

        Logger::error() << "error1";

        std::string i1 = Logger::get2ndlog();
        EXPECT_EQ(i1, "");  // not active
    }
    Logger::activate2ndlog();

    {
        std::string i0 = Logger::get2ndlog();
        EXPECT_EQ(i0, "");  // nothing  in the beginning

        Logger::error() << "error2";
        std::string i1 = Logger::get2ndlog();

        EXPECT_TRUE(i1 != "");
        EXPECT_EQ(i1, "[VPUNN ERROR]: error2\n");
    }
    {
        Logger::trace() << "error3";
        std::string i2 = Logger::get2ndlog();
        EXPECT_EQ(i2, "[VPUNN ERROR]: error2\n[VPUNN TRACE]: error3\n");
    }
    // deactivate
    Logger::deactivate2ndlog();

    {
        Logger::trace() << "error4";
        std::string i2 = Logger::get2ndlog();
        EXPECT_EQ(i2, "[VPUNN ERROR]: error2\n[VPUNN TRACE]: error3\n");  // notlogged, OFF
    }
}

TEST_F(VPUNNLoggerTest, Alternative2ndOutput_Reset) {
    Logger::initialize();

    Logger::activate2ndlog();

    {
        Logger::error() << "error2";
        std::string i1 = Logger::get2ndlog();

        EXPECT_TRUE(i1 != "");
        EXPECT_EQ(i1, "[VPUNN ERROR]: error2\n");
    }
    Logger::clear2ndlog();
    {
        std::string i2 = Logger::get2ndlog();
        EXPECT_EQ(i2, "");
    }

    {
        Logger::trace() << "error3";
        std::string i2 = Logger::get2ndlog();
        EXPECT_EQ(i2, "[VPUNN TRACE]: error3\n");
    }
}

}  // namespace VPUNN_unit_tests
