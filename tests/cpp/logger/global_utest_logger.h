// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_GLOBAL_UTEST_LOGGER_H
#define VPUNN_GLOBAL_UTEST_LOGGER_H

#include <string>

#include <spdlog/logger.h>

#include "logger/logger_level.h"

namespace VPUNN_unit_tests {
namespace utest_log {

/**
 * @brief Singleton logger used by the C++ unit tests.
 *
 * The logger is configured once and shared across the test process. It exposes
 * the same severity model as spdlog while keeping the rest of the test code
 * decoupled from the underlying library details.
 */
class GlobalUTestLogger {
public:
    /**
     * @brief Returns the process-wide test logger instance.
     * @return Singleton instance of the test logger.
     */
	static GlobalUTestLogger& instance();

    /**
     * @brief Sets the active minimum severity level for the logger.
     * @param level New minimum threshold for emitted messages.
     */
	void set_level(Level::level_enum level);

    /**
     * @brief Returns the current active severity threshold.
     * @return Current logger threshold.
     */
	Level::level_enum level() const;

    /**
     * @brief Checks whether a message at the given severity would be emitted.
     * @param level Severity being evaluated.
     * @return True when the current threshold allows this level.
     */
	bool should_log(Level::level_enum level) const;

    /**
     * @brief Emits a message at the requested severity.
     * @param level Severity attached to the message.
     * @param message Message content to log.
     */
	void log(Level::level_enum level, const std::string& message);

    /**
     * Copying is disabled because this class owns the process-wide logger
     * configuration and must remain a singleton.
     */
    GlobalUTestLogger(const GlobalUTestLogger&) = delete;

    /**
     * Copy assignment is disabled because it could create a second logger
     * object with independent configuration.
     */
    GlobalUTestLogger& operator=(const GlobalUTestLogger&) = delete;

    /**
     * Moving is disabled because the process-wide logger must retain one
     * stable owner for its entire lifetime.
     */
    GlobalUTestLogger(GlobalUTestLogger&&) = delete;

    /**
     * Move assignment is disabled for the same singleton ownership reason.
     */
    GlobalUTestLogger& operator=(GlobalUTestLogger&&) = delete;

private:
    /**
     * @brief Constructs the singleton logger and attaches the console sink.
     */
	GlobalUTestLogger();

    /// Shared spdlog logger used across the test suite.
	spdlog::logger logger_;
};

}  // namespace utest_log
}  // namespace VPUNN_unit_tests

#endif  // VPUNN_GLOBAL_UTEST_LOGGER_H
