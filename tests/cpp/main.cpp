// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include <cxxopts.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "logger/global_utest_logger.h"
#include "logger/logger.h"
#include "logger/logger_level.h"

namespace VPUNN_unit_tests {
namespace {

constexpr const char* logger_level_env_var{"LOGGER_LEVEL"};
constexpr utest_log::Level::level_enum default_logger_level{utest_log::Level::info};

// ctest runs the executable without forwarding flags, so the environment variable is the fallback there.
std::string get_env_value(const char* name) {
#if defined(_MSC_VER)  // avoid using getenv directly on Windows, which generates a warning (C4996)
    char* value{nullptr};
    size_t value_size{0};
    if (_dupenv_s(&value, &value_size, name) != 0 || value == nullptr) {
        return {};
    }

    std::string result{value};
    std::free(value);
    return result;
#else
    const auto value = std::getenv(name);
    return value != nullptr ? std::string{value} : std::string{};
#endif
}

utest_log::Level::level_enum resolve_logger_level(const cxxopts::ParseResult& result) {
    if (result.count("logger_level") != 0) {
        return utest_log::Level::from_str(result["logger_level"].as<std::string>());
    }

    const auto from_env = get_env_value(logger_level_env_var);
    if (!from_env.empty()) {
        return utest_log::Level::from_str(from_env);
    }

    return default_logger_level;
}

}  // namespace

// If not specified, the logger will use the default level 'info'. Everything with a severity level equal to
// or higher than logger's level will be logged.
//
// Changing globally logger's level can be done from cli in 2 ways:
// - use the flag --logger_level followed by a valid level (exclusive for using the executable)
// - use the environment variable: set LOGGER_LEVEL to a valid level (for using both the executable and ctest)
// The valid levels are: [trace, debug, info, warn, err, critical, off]. Invalid values will be treated as 'off'
//
// Changing logger's level for a specific test can be done by using LogLevelScope guard(utest_log::Level::debug);
//
// How to use the logger:
// Example usage:
// utest_log::info() << "This is an info message"
// utest_log::debug() << "This is a debug message"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    cxxopts::Options options("test_cost_model", "VPUNN test runner");
    options.add_options()("logger_level",
                          "Logger level [trace, debug, info, warn, err, critical, off]. Falls back to the "
                          "LOGGER_LEVEL environment variable, then to 'info'",
                          cxxopts::value<std::string>())("h,help", "Show help");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n' << options.help() << '\n';
        return 2;
    }

    if (result.count("help") != 0) {
        std::cout << '\n' << options.help() << '\n';
        return 0;
    }

    utest_log::GlobalUTestLogger::instance().set_level(resolve_logger_level(result));
    return RUN_ALL_TESTS();
}
}  // namespace VPUNN_unit_tests

// The CRT is looking for a global main function, not a namespaced one.
// In this way, a future set of tests could be easily integrated without heavily modifying the global main
// function by creating another namespaced main function that handles that specific set of tests.
// Choosing the right main could be done with a cli argument.
int main(int argc, char** argv) {
    return VPUNN_unit_tests::main(argc, argv);
}
