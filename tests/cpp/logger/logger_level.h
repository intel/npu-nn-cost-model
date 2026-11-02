// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UTEST_LOGGER_LEVEL_H
#define VPUNN_UTEST_LOGGER_LEVEL_H

#include <string>

#include <spdlog/common.h>

namespace VPUNN_unit_tests {
namespace utest_log {

/**
 * @brief Thin wrapper around spdlog's log severity enum.
 *
 * This keeps the test logger API readable without duplicating the actual
 * underlying spdlog enum values or parser logic.
 */
struct Level {
    /// Underlying logger severity type from spdlog.
    using level_enum = spdlog::level::level_enum;

    /// Most verbose supported severity.
    static constexpr level_enum trace = spdlog::level::trace;
    /// Debug-level detail output.
    static constexpr level_enum debug = spdlog::level::debug;
    /// Default informational output.
    static constexpr level_enum info = spdlog::level::info;
    /// Warning-level output.
    static constexpr level_enum warn = spdlog::level::warn;
    /// Error-level output.
    static constexpr level_enum err = spdlog::level::err;
    /// Critical failure output.
    static constexpr level_enum critical = spdlog::level::critical;
    /// Disables logging for the selected level.
    static constexpr level_enum off = spdlog::level::off;

    /**
     * @brief Parses a string into the matching spdlog severity.
     *
     * @param level String form of the desired severity, such as "info" or "warn".
     * @return Matching severity enumeration value.
     */
    static level_enum from_str(const std::string& level) {
        return spdlog::level::from_str(level);
    }
};

}  // namespace utest_log
}  // namespace VPUNN_unit_tests

#endif  // VPUNN_UTEST_LOGGER_LEVEL_H
