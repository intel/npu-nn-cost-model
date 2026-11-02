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

#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUNNLoggerTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
        Logger::clear2ndlog();
        Logger::deactivate2ndlog();
    }

    /// Swaps std::cout's stream buffer for the object's lifetime to capture what's printed.
    /// Replacement for gtest's CaptureStdout() that redirects the file descriptor instead, which invalidates
    /// the console handle cached by spdlog and silently kills utest_log output for the rest of the run.
    struct CoutCapture {
        std::ostringstream captured; 
        std::streambuf* previous;
        CoutCapture() : previous(std::cout.rdbuf(captured.rdbuf())) {}
        ~CoutCapture() { std::cout.rdbuf(previous); }
        std::string str() const { return captured.str(); }
    };
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

    {
        CoutCapture capture;
        Logger::error() << "error";
        output = capture.str();
    }
#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(output, "[VPUNN ERROR]: error\n");
#else
    EXPECT_EQ(output, "");
#endif

    {
        CoutCapture capture;
        Logger::fatal() << "fatal";
        output = capture.str();
    }
#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(output, "[VPUNN FATAL]: fatal\n");
#else
    EXPECT_EQ(output, "");
#endif

    {
        CoutCapture capture;
        Logger::warning() << "warning";
        output = capture.str();
    }
#ifdef VPUNN_ENABLE_LOGGING
    EXPECT_EQ(output, "[VPUNN WARNING]: warning\n");
#else
    EXPECT_EQ(output, "");
#endif

    {
        CoutCapture capture;
        Logger::info() << "info";
        output = capture.str();
    }
    EXPECT_EQ(output, "");

    {
        CoutCapture capture;
        Logger::debug() << "debug";
        output = capture.str();
    }
    EXPECT_EQ(output, "");

    {
        CoutCapture capture;
        Logger::trace() << "trace";
        output = capture.str();
    }
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

TEST_F(VPUNNLoggerTest, MultiThreaded_Test) {
    Logger::initialize();
    Logger::activate2ndlog();

    const auto num_threads = 8;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i]() {
            Logger::error() << "error from thread " << i;
            Logger::warning() << "warning from thread " << i;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Collect the 2nd log output
    std::string log_output = Logger::get2ndlog();

    // Check that each thread's error and warning messages are present
    for (int i = 0; i < num_threads; ++i) {
        std::string error_msg = "[VPUNN ERROR]: error from thread " + std::to_string(i) + "\n";
        std::string warning_msg = "[VPUNN WARNING]: warning from thread " + std::to_string(i) + "\n";
        EXPECT_NE(log_output.find(error_msg), std::string::npos) << "Missing: " << error_msg;
        EXPECT_NE(log_output.find(warning_msg), std::string::npos) << "Missing: " << warning_msg;
        // Optionally, check info is not present
        std::string info_msg = "[VPUNN INFO]: info from thread " + std::to_string(i) + "\n";
        EXPECT_EQ(log_output.find(info_msg), std::string::npos) << "Unexpected: " << info_msg;
    }
}

}  // namespace VPUNN_unit_tests
