// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/logger.h"

namespace VPUNN {

LogLevel Logger::_logLevel = LogLevel::None;

const std::string toString(LogLevel level) {
    switch (level) {
    case LogLevel::None:
        return "NONE";
    case LogLevel::Fatal:
        return "FATAL";
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Warning:
        return "WARNING";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Debug:
        return "DEBUG";
    case LogLevel::Trace:
        return "TRACE";
    default:
        return "NONE";
    }
}

}  // namespace VPUNN