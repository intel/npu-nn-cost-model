// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UTEST_LOG_LEVEL_SCOPE_H
#define VPUNN_UTEST_LOG_LEVEL_SCOPE_H

#include "logger/global_utest_logger.h"
#include "logger/logger_level.h"

namespace VPUNN_unit_tests {
namespace utest_log {

/**
 * @brief Temporarily changes the shared test logger level.
 *
 * The previous level is restored when the guard is destroyed. This is useful
 * when a single test needs a different level without changing the level for
 * the rest of the test suite.
 *
 * The logger level is shared by the entire test process. This guard is safe
 * for the current GoogleTest execution model, where test bodies run
 * sequentially. Do not use overlapping guards or change the logger level from
 * concurrent test bodies in the same process.
 */
class LogLevelScope {
public:
    /**
     * @brief Sets the logger level for the current scope.
     * @param level The logger level to set.
     */
    explicit LogLevelScope(Level::level_enum level): previous_(GlobalUTestLogger::instance().level()) {
        GlobalUTestLogger::instance().set_level(level);
    }

    /**
     * @brief Restores the logger level active before construction.
     */
    ~LogLevelScope() {
        GlobalUTestLogger::instance().set_level(previous_);
    }

    LogLevelScope(const LogLevelScope&) = delete;
    LogLevelScope& operator=(const LogLevelScope&) = delete;

private:
    /// Saving the previous value makes nested scopes restore levels in order.
    Level::level_enum previous_;
};

}  // namespace utest_log
}  // namespace VPUNN_unit_tests

#endif  // VPUNN_UTEST_LOG_LEVEL_SCOPE_H
