// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "logger/global_utest_logger.h"
#include "logger/logger_level.h"

#include <iostream>
#include <memory>
#include <string>

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/ostream_sink.h>

namespace VPUNN_unit_tests {
namespace utest_log {

/// Returns the singleton instance for the shared unit-test logger.
GlobalUTestLogger& GlobalUTestLogger::instance() {
    static GlobalUTestLogger logger{};
    return logger;
}

/// Sets the minimum severity emitted by the shared logger.
void GlobalUTestLogger::set_level(Level::level_enum level) {
    logger_.set_level(level);
}

/// Returns the current minimum severity threshold for the shared logger.
Level::level_enum GlobalUTestLogger::level() const {
    return logger_.level();
}

/// Checks whether the given severity would currently be emitted.
bool GlobalUTestLogger::should_log(Level::level_enum level) const {
    return logger_.should_log(level);
}

/// Emits a message through the shared logger at the requested severity.
void GlobalUTestLogger::log(Level::level_enum level, const std::string& message) {
    logger_.log(level, "{}", message);
}

/// Constructs the shared logger with a stdout sink and default info-level threshold.
GlobalUTestLogger::GlobalUTestLogger()
        : logger_("tests", std::make_shared<spdlog::sinks::ostream_sink_mt>(std::cout, true)) {
    logger_.set_level(Level::info);
    logger_.set_formatter(std::make_unique<spdlog::pattern_formatter>("%v", spdlog::pattern_time_type{}, ""));
}

}  // namespace utest_log
}  // namespace VPUNN_unit_tests