// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "logger/logger_stream.h"
#include "logger/global_utest_logger.h"

#include <ios>
#include <ostream>

namespace VPUNN_unit_tests {
namespace utest_log {

/// Creates a buffered stream for the requested severity and checks whether it is enabled.
Stream::Stream(Level::level_enum level): level_(level) /*, enabled_(GlobalUTestLogger::instance().should_log(level))*/ {
}

/// Flushes the buffered message to the shared logger when the current threshold allows it.
Stream::~Stream() {
    GlobalUTestLogger::instance().log(level_, buffer_.str());
}

/// Applies a standard ostream manipulator only when logging is enabled.
Stream& Stream::operator<<(std::ostream& (*manipulator)(std::ostream&)) {
    manipulator(buffer_);
    return *this;
}

/// Applies a std::ios_base manipulator only when logging is enabled.
Stream& Stream::operator<<(std::ios_base& (*manipulator)(std::ios_base&)) {
    manipulator(buffer_);
    return *this;
}

}  // namespace utest_log
}  // namespace VPUNN_unit_tests