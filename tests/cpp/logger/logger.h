// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UTEST_LOGGER_H
#define VPUNN_UTEST_LOGGER_H

#include "logger/logger_stream.h"

namespace VPUNN_unit_tests {
namespace utest_log {

/**
 * @brief Creates a stream configured for trace-level output.
 *
 * @return A stream object that emits messages only when the configured logger
 *         level allows trace entries.
 */
Stream trace();

/**
 * @brief Creates a stream configured for debug-level output.
 *
 * @return A stream object that emits messages only when debug logging is enabled.
 */
Stream debug();

/**
 * @brief Creates a stream configured for info-level output.
 *
 * @return A stream object that emits messages when info-level logging is active.
 */
Stream info();

/**
 * @brief Creates a stream configured for warning-level output.
 *
 * @return A stream object that emits messages when warnings are enabled.
 */
Stream warn();

/**
 * @brief Creates a stream configured for error-level output.
 *
 * @return A stream object that emits messages when error logging is active.
 */
Stream err();

/**
 * @brief Creates a stream configured for critical-level output.
 *
 * @return A stream object that emits messages only for critical failures.
 */
Stream critical();

}  // namespace utest_log
}  // namespace VPUNN_unit_tests

#endif  // VPUNN_UTEST_LOGGER_H
