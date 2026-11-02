// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <gtest/gtest.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <streambuf>
#include <string>

#include "logger/log_level_scope.h"
#include "logger/logger.h"

namespace VPUNN_unit_tests {

class TestLogger : public ::testing::Test {
protected:
    /// Swaps std::cout's stream buffer for the object's lifetime to capture what's printed.
    /// Replacement for gtest's CaptureStdout() that redirects the file descriptor instead, which can invalidate
    /// the console handle cached by the logger and silently kill utest_log output for the rest of the run.
    struct CoutCapture {
        std::ostringstream captured;
        std::streambuf* previous;
        CoutCapture(): previous(std::cout.rdbuf(captured.rdbuf())) {}
        ~CoutCapture() { std::cout.rdbuf(previous); }
        std::string str() const { return captured.str(); }
    };
};

TEST_F(TestLogger, TraceLevelEmitsAllMessages) {
    utest_log::LogLevelScope guard(utest_log::Level::trace);
    std::string logged_output;

    {
        CoutCapture capture;
        utest_log::trace() << "trace message\n";
        utest_log::debug() << "debug message\n";
        utest_log::info() << "info message\n";
        utest_log::warn() << "warning message\n";
        utest_log::err() << "error message\n";
        utest_log::critical() << "critical message\n";
        logged_output = capture.str();
    }

    EXPECT_NE(logged_output.find("trace message"), std::string::npos);
    EXPECT_NE(logged_output.find("debug message"), std::string::npos);
    EXPECT_NE(logged_output.find("info message"), std::string::npos);
    EXPECT_NE(logged_output.find("warning message"), std::string::npos);
    EXPECT_NE(logged_output.find("error message"), std::string::npos);
    EXPECT_NE(logged_output.find("critical message"), std::string::npos);
}

TEST_F(TestLogger, WarnLevelSuppressesLowerPriorityMessages) {
    utest_log::LogLevelScope guard(utest_log::Level::warn);
    std::string logged_output;

    {
        CoutCapture capture;
        utest_log::trace() << "trace message\n";
        utest_log::debug() << "debug message\n";
        utest_log::info() << "info message\n";
        utest_log::warn() << "warning message\n";
        utest_log::err() << "error message\n";
        utest_log::critical() << "critical message\n";
        logged_output = capture.str();
    }

    EXPECT_EQ(logged_output.find("trace message"), std::string::npos);
    EXPECT_EQ(logged_output.find("debug message"), std::string::npos);
    EXPECT_EQ(logged_output.find("info message"), std::string::npos);
    EXPECT_NE(logged_output.find("warning message"), std::string::npos);
    EXPECT_NE(logged_output.find("error message"), std::string::npos);
    EXPECT_NE(logged_output.find("critical message"), std::string::npos);
}

TEST_F(TestLogger, OffLevelSuppressesAllMessages) {
    utest_log::LogLevelScope guard(utest_log::Level::off);
    std::string logged_output;

    {
        CoutCapture capture;
        utest_log::trace() << "trace message\n";
        utest_log::debug() << "debug message\n";
        utest_log::info() << "info message\n";
        utest_log::warn() << "warning message\n";
        utest_log::err() << "error message\n";
        utest_log::critical() << "critical message\n";
        logged_output = capture.str();
    }

    EXPECT_TRUE(logged_output.empty());
}

TEST_F(TestLogger, OnePrintedLineWithMultipleLogs) {
    utest_log::LogLevelScope guard(utest_log::Level::info);
    std::string logged_output;

    {
        CoutCapture capture;
        utest_log::info() << "info ";
        utest_log::info() << "message\n";
        logged_output = capture.str();
    }

    EXPECT_EQ(logged_output, "info message\n");
}

TEST_F(TestLogger, InfoLogsChainedValuesAndManipulator) {
    utest_log::LogLevelScope guard(utest_log::Level::info);
    std::string logged_output;

    {
        CoutCapture capture;
        utest_log::info() << "something " << "something " << 12 << ' ' << std::setw(4) << 34 << '\n';
        logged_output = capture.str();
    }

    EXPECT_EQ(logged_output, "something something 12   34\n");
}

}  // namespace VPUNN_unit_tests
