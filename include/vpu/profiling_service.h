// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_PROFILING_SERVICE_H
#define VPUNN_PROFILING_SERVICE_H

#include "dpu_types.h"

namespace VPUNN {

enum class ProfilingServiceBackend { SILICON, VPUEM, __size };
static const EnumMap ProfilingServiceBackend_ToText{
    link(ProfilingServiceBackend::SILICON, "silicon"),
    link(ProfilingServiceBackend::VPUEM, "vpuem"),
};
template <>
inline const EnumMap& mapToText<ProfilingServiceBackend>() {
    return ProfilingServiceBackend_ToText;
}

template <>
inline std::string enumName<ProfilingServiceBackend>() {
    return "ProfilingServiceBackend";
}

}  // namespace VPUNN

#endif
