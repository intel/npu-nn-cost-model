// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_CYCLE_INTERFACE_TYPES_H
#define VPUNN_CYCLE_INTERFACE_TYPES_H

#include <cstdint>
#include <limits>

namespace VPUNN {

/// @brief cycles values and error code, dual meaning.
///
/// This is the type to be used for returning cycle values as normal values (positive and not absurdly large), and to
/// return error codes in case the cycle values are nor available
///
/// abnormal return codes
/// ERROR_INPUT_TOO_BIG:                UINT_MAX     :0 - 1: ~infinite, does not fit into cmx memory
/// ERROR_INVALID_INPUT_CONFIGURATION:  UINT_MAX - 1 :0 - 2: invalid configuration for specified device
/// ERROR_INVALID_INPUT_DEVICE:         UINT_MAX - 2 :0 - 3: invalid/unknown device, cannot sanitize/check
/// ERROR_INVALID_INPUT_OPERATION:      UINT_MAX - 3 :0 - 4: unknown/unsupported operation
/// ERROR_INVALID_OUTPUT_RANGE:         UINT_MAX - 4 :0 - 5: NN provided an Out of Range output (negative or very large)
/// ERROR_TILE_OUTPUT:                  UINT_MAX - 5 :0 - 6: One of the tiles provided error cycle types
/// ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT    UINT_MAX - 6 :0 - 7: When intra tile split , a zero cycles was observed
/// ERROR_TILE_SPLIT_EXCEPTION          UINT_MAX - 7 :0 - 8: When intra tile split , an exception was happening  for a
/// split
/// ERROR_INVALID_LAYER_CONFIGURATION   UINT_MAX - 8 :0 - 9: An unsplit layer has invalid content
///
///     Zero value is not an error, and can represent NN cycles output. This might let the NN communicate something like
///     it cannot solve the request.
///     The zero value behavior might change in the future!
using CyclesInterfaceType = std::uint32_t;

/// @brief helper class for CyclesInterfaceType
class Cycles {
    static constexpr CyclesInterfaceType MaxV{std::numeric_limits<CyclesInterfaceType>::max()};

public:
    static constexpr CyclesInterfaceType NO_ERROR{0};  ///< special valid value
    static constexpr CyclesInterfaceType ERROR_INPUT_TOO_BIG{MaxV - 0};
    static constexpr CyclesInterfaceType ERROR_INVALID_INPUT_CONFIGURATION{MaxV - 1};
    static constexpr CyclesInterfaceType ERROR_INVALID_INPUT_DEVICE{MaxV - 2};
    static constexpr CyclesInterfaceType ERROR_INVALID_INPUT_OPERATION{MaxV - 3};
    static constexpr CyclesInterfaceType ERROR_INVALID_OUTPUT_RANGE{MaxV - 4};

    static constexpr CyclesInterfaceType ERROR_TILE_OUTPUT{MaxV - 5};
    static constexpr CyclesInterfaceType ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT{MaxV - 6};
    static constexpr CyclesInterfaceType ERROR_TILE_SPLIT_EXCEPTION{MaxV - 7};

    static constexpr CyclesInterfaceType ERROR_INVALID_LAYER_CONFIGURATION{MaxV - 8};

    static constexpr CyclesInterfaceType START_ERROR_RANGE{MaxV - 1000};  ///< 1000 position for errors

    /// @brief true if v has a value that can be an error code
    ///
    /// @param v the value to be interpreted.
    /// @returns true if the value is large enough to be mapped to an error code (error code might exist or not)
    static constexpr bool isErrorCode(const CyclesInterfaceType& v) {
        return (v > START_ERROR_RANGE);
    }

    /// @brief provides a text if the value is an error or zero
    ///
    /// @param v the value to be interpreted. normally 0-reasonable values means cycles, and values close to max limit
    /// are error codes
    /// @returns a plain text with error name or "UNKNOWN"
    static constexpr char const* toErrorText(const CyclesInterfaceType& v) {
        switch (v) {
        case NO_ERROR:
            return "NO_ERROR";
        case ERROR_INPUT_TOO_BIG:
            return "ERROR_INPUT_TOO_BIG";
        case ERROR_INVALID_INPUT_CONFIGURATION:
            return "ERROR_INVALID_INPUT_CONFIGURATION";
        case ERROR_INVALID_INPUT_DEVICE:
            return "ERROR_INVALID_INPUT_DEVICE";
        case ERROR_INVALID_INPUT_OPERATION:
            return "ERROR_INVALID_INPUT_OPERATION";
        case ERROR_INVALID_OUTPUT_RANGE:
            return "ERROR_INVALID_OUTPUT_RANGE";
        case ERROR_TILE_OUTPUT:
            return "ERROR_TILE_OUTPUT";
        case ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT:
            return "ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT";
        case ERROR_TILE_SPLIT_EXCEPTION:
            return "ERROR_TILE_SPLIT_EXCEPTION";
        case ERROR_INVALID_LAYER_CONFIGURATION:
            return "ERROR_INVALID_LAYER_CONFIGURATION";
        default:
            return "UNKNOWN";
        }
    }
};

}  // namespace VPUNN

#endif  //
