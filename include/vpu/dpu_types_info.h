// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_TYPES_INFO_H
#define VPUNN_DPU_TYPES_INFO_H

#include <array>
#include <cassert>
#include <vector>

#include "dim_enum.h"
#include "dpu_types.h"

namespace VPUNN {

/**
 * @brief Get the tensor serial order given a layout
 *
 * @param layout a Tensor Layout
 * @return std::array<unsigned int, 4>, order of dimensions from innermost to outermost. values represent Dim::Act
 *
 * Invalid will be mapped to the default one : ZMAJOR/ZXY
 */
constexpr std::array<unsigned int, 4> layout_to_order(Layout layout) noexcept {
    switch (layout) {
    case Layout::CMAJOR:
        return {Dim::Act::X, Dim::Act::Y, Dim::Act::Z, Dim::Act::B};  // X,Y,Z,B  from innermost to outermost dimensions
    case Layout::ZMAJOR:
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y,B  from innermost to outermost dimensions

    case Layout::XYZ:
        return {Dim::Act::X, Dim::Act::Y, Dim::Act::Z, Dim::Act::B};  // X,Y,Z, B
    case Layout::XZY:
        return {Dim::Act::X, Dim::Act::Z, Dim::Act::Y, Dim::Act::B};  // X,Z,Y, B

    case Layout::YXZ:
        return {Dim::Act::Y, Dim::Act::X, Dim::Act::Z, Dim::Act::B};  // Y,X,Z, B
    case Layout::YZX:
        return {Dim::Act::Y, Dim::Act::Z, Dim::Act::X, Dim::Act::B};  // Y,Z,X, B

    case Layout::ZXY:
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y, B
    case Layout::ZYX:
        return {Dim::Act::Z, Dim::Act::Y, Dim::Act::X, Dim::Act::B};  // Z,Y,X, B

    case Layout::INVALID:                                             // fall-through
    default:                                                          // ZMajor like, this is the first in the enum list
        return {Dim::Act::Z, Dim::Act::X, Dim::Act::Y, Dim::Act::B};  // Z,X,Y,B  from innermost to outermost dimensions
    }
}

/**
 * @brief Return grid in X, Y, Z, B format (alignments?)
 * Note: potentially OBSOLETE, since it does not depend on Device
 *
 * @param mode a DPUWorkload ExecutionMode
 * @return std::vector<unsigned int>
 */
inline std::vector<unsigned int> mpe_mode_to_grid(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::VECTOR:
        return {16, 1, 16, 1};
    case ExecutionMode::VECTOR_FP16:
        return {4, 1, 16, 1};
    default:
        return {4, 4, 16, 1};
    }
}

/**
 * @brief Return the NTHW/NTK grid in X, Y, Z, B format
 * Note: potentially OBSOLETE, since it does not depend on Device
 *
 * @param mode a DPUWorkload ExecutionMode
 * @return std::vector<unsigned int>
 */
inline std::vector<unsigned int> mpe_mode_to_nthw_ntk_grid(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::CUBOID_4x16:
        return {8, 8, 256, 1};
    case ExecutionMode::CUBOID_8x16:
        return {16, 8, 128, 1};
    case ExecutionMode::CUBOID_16x16:
        return {16, 16, 64, 1};
    default:
        return {1, 1, 1, 1};
    }
}

/// @brief Returns true if the given execution mode belongs to the dCIM family.
///
/// dCIM execution modes require MPEEngine::DCIM and represent distinct hardware tile
/// configurations. Any new dCIM variant must be added here.
///
/// @param mode the ExecutionMode to test
/// @return true if mode is dCIM_32x128 or dCIM_64x128, false otherwise
inline constexpr bool is_dcim_execution_mode(const ExecutionMode mode) noexcept {
    return mode == ExecutionMode::dCIM_32x128 || mode == ExecutionMode::dCIM_64x128;
}

}  // namespace VPUNN

#endif  // VPUNN_DPU_TYPES_INFO_H
