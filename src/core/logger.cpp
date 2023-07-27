// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/logger.h"
#include "vpu/cycles_interface_types.h"

namespace VPUNN {

LogLevel Logger::_logLevel = LogLevel::None;

std::ostringstream Logger::buffer{};
std::ostringstream* Logger::active_second_logger{nullptr};

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

// allocate also for Cycles
// CyclesInterfaceType constexpr Cycles::NO_ERROR{};

}  // namespace VPUNN