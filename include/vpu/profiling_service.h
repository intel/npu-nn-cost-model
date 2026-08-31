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

#include "core/utils.h"  // for get_env_vars
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

/// @brief Whether ENABLE_VPUNN_PROFILING_SERVICE forced profiling on/off, or left it unset (runtime-controlled).
enum class EnvProfilingState { NotSet, ForcedOn, ForcedOff };

inline EnvProfilingState readProfilingEnvState() {
    const auto env_map = get_env_vars({"ENABLE_VPUNN_PROFILING_SERVICE"});
    const auto& value = env_map.at("ENABLE_VPUNN_PROFILING_SERVICE");
    if (value == "TRUE") {
        return EnvProfilingState::ForcedOn;
    }
    if (value == "FALSE") {
        return EnvProfilingState::ForcedOff;
    }
    return EnvProfilingState::NotSet;
}

/// @brief Gate for whether a cache-miss AUTO-hint query may use the online profiling service. Defaults from
/// ENABLE_VPUNN_PROFILING_SERVICE; set() is a no-op if the environment forced a value.
struct ProfilingAutoHintGate {
    const EnvProfilingState envState{readProfilingEnvState()};
    bool enabled{envState == EnvProfilingState::ForcedOn};

    void set(bool value) noexcept {
        if (envState == EnvProfilingState::NotSet) {
            enabled = value;
        }
    }

    bool isEnabled() const noexcept {
        return enabled;
    }
};

}  // namespace VPUNN

#endif
