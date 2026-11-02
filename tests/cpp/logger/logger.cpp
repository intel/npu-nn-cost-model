// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.



#include "logger/logger.h"
#include "logger/logger_level.h"
#include "logger/logger_stream.h"

namespace VPUNN_unit_tests {
namespace utest_log {

/// Creates a stream that emits trace-level messages when enabled.
Stream trace() {
    return Stream(Level::trace);
}

/// Creates a stream that emits debug-level messages when enabled.
Stream debug() {
    return Stream(Level::debug);
}

/// Creates a stream that emits info-level messages when enabled.
Stream info() {
    return Stream(Level::info);
}

/// Creates a stream that emits warning-level messages when enabled.
Stream warn() {
    return Stream(Level::warn);
}

/// Creates a stream that emits error-level messages when enabled.
Stream err() {
    return Stream(Level::err);
}

/// Creates a stream that emits critical-level messages when enabled.
Stream critical() {
    return Stream(Level::critical);
}

}  // namespace utest_log
}  // namespace VPUNN_unit_tests
