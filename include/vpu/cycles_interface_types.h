// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_CYCLE_INTERFACE_TYPES_H
#define VPUNN_CYCLE_INTERFACE_TYPES_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

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
/// ERROR_CUMULATED_CYCLES_TOO_LARGE    UINT_MAX - 9 :0 - 10: Adding the cycles result in accessing the error area
/// ERROR_INVALID_CONVERSION_TO_CYCLES  UINT_MAX - 10:0 - 11: Converting another type to CyclesInterfaceType cannot fit
/// or is an error
/// ERROR_SHAVE                         UINT_MAX - 11:0 - 12:Shave operation was not successful. Generic problem (maybe
/// unavailable)
///
/// ERROR_INFERENCE_NOT_POSSIBLE        UINT_MAX - 12:0 - 13:Inference is not possible to be run. Maybe no NN
/// available.
///
/// ERROR_SHAVE_PARAMS                  UINT_MAX - 13:0 - 14:Problems with SHAVE parameters. Maybe some variadic
/// parameters were not passed!
/// ERROR_SHAVE_LAYOUT                  UINT_MAX - 14:0 - 15:Problems with SHAVE layout.
/// ERROR_SHAVE_INVALID_INPUT           UINT_MAX - 15:0 - 16:Problems with SHAVE workload input.
/// ERROR_L2_INVALID_PARAMETERS         UINT_MAX - 16:0 - 17:Invalid parameters at L2 API 
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

    static constexpr CyclesInterfaceType ERROR_CUMULATED_CYCLES_TOO_LARGE{MaxV - 9};

    static constexpr CyclesInterfaceType ERROR_INVALID_CONVERSION_TO_CYCLES{MaxV - 10};

    static constexpr CyclesInterfaceType ERROR_SHAVE{MaxV - 11};

    static constexpr CyclesInterfaceType ERROR_INFERENCE_NOT_POSSIBLE{MaxV - 12};

    static constexpr CyclesInterfaceType ERROR_SHAVE_PARAMS{MaxV - 13};
    static constexpr CyclesInterfaceType ERROR_SHAVE_LAYOUT{MaxV - 14};
    static constexpr CyclesInterfaceType ERROR_SHAVE_INVALID_INPUT{MaxV - 15};

    static constexpr CyclesInterfaceType ERROR_L2_INVALID_PARAMETERS{MaxV - 16}; // used for invalid parameters for L2 API

    static constexpr CyclesInterfaceType ERROR_PROFILING_SERVICE{MaxV - 17};  // used for invalid output from profiling service

    static constexpr CyclesInterfaceType ERROR_CACHE_MISS{MaxV - 18};  // used for cache miss

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
        case ERROR_CUMULATED_CYCLES_TOO_LARGE:
            return "ERROR_CUMULATED_CYCLES_TOO_LARGE";
        case ERROR_INVALID_CONVERSION_TO_CYCLES:
            return "ERROR_INVALID_CONVERSION_TO_CYCLES";
        case ERROR_SHAVE:
            return "ERROR_SHAVE";
        case ERROR_INFERENCE_NOT_POSSIBLE:
            return "ERROR_INFERENCE_NOT_POSSIBLE";
        case ERROR_SHAVE_PARAMS:
            return "ERROR_SHAVE_PARAMS";
        case ERROR_SHAVE_LAYOUT:
            return "ERROR_SHAVE_LAYOUT";
        case ERROR_SHAVE_INVALID_INPUT:
            return "ERROR_SHAVE_INVALID_INPUT";
        case ERROR_L2_INVALID_PARAMETERS:
            return "ERROR_L2_INVALID_PARAMETERS";
        case ERROR_PROFILING_SERVICE:
            return "ERROR_PROFILING_SERVICE";
        case ERROR_CACHE_MISS:
            return "ERROR_CACHE_MISS";
        default:
            return "UNKNOWN";
        }
    }

    /** @brief safe sum of cycles considering also the error handling situations and overflow
     * If the sum of the valid numbers gets in the error area , it will result Cycles::EROOR_SUM_TOO_LARGE
     * If one of the terms is already an error, the error is kept as result (first term has priority of both are errors)
     *
     * @param lhs left term
     * @param rhs right term
     *
     * @returns the sum of lhs with rhs or the specific error in case that we have a sum
     * too large or with error in terms
     */
    static constexpr CyclesInterfaceType cost_adder(const CyclesInterfaceType lhs, const CyclesInterfaceType rhs) {
        // if one of them are an error than the code will need to propagate the first error encountered
        // usually the lhs_cost is either the very first term of adding/ the accumulator for the sum
        // second parameter is usually the second term of adding/ the cost you want to add to the accumulator
        if (isErrorCode(lhs)) {
            return lhs;
        } else if (isErrorCode(rhs)) {
            return rhs;
        }
        // in case neither one of the terms is an error than we should check if we are not overflowing
        // if the maximum value allowed for the CyclesInterfaceType minus one of the costs can't cover the new cost we
        // want to add in we will be in a state of overflow so we are going to receive an error
        if ((std::numeric_limits<CyclesInterfaceType>::max() - lhs) < rhs) {
            return Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE;
        }
        //  then if it is not over the upper error limit, then we should simply try to see if we got the new cost in a
        //  zone of errors
        const CyclesInterfaceType result_cost = lhs + rhs;
        if (isErrorCode(result_cost)) {
            return Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE;
        }
        // in case that all of this are passing than we can return the new number of cycles
        return result_cost;
    }

    /**
     * @brief This method has the role to convert any another numeric types into CyclesInterfaceType. This function
     * will assure that the number returned is a valid one in terms of cycles. In case that the provided number is out
     * of bounds than it will return the specific error. If our generic type is integer it will return the exact number,
     * but if we have a floating unit then we are going to ceil to make sure that we are covering the float in a cycle
     *
     * @param conversion_number is the provided number to be converted in cycles
     * @return will return the number converted in CyclesInterfaceType
     */
    template <class T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    inline static constexpr CyclesInterfaceType toCycleInterfaceType(T conversion_number) {
        // in case that we got a signed number in this operation and the number is negative than we can not convert
        // it properly into a valid number of cycles. This way we are making sure that we are excluding any case which
        // this number is negative
        constexpr bool is_signed{std::is_signed<T>::value};
        if (is_signed) {
            if (conversion_number < 0)
                return Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES;
        }

        // the number now is a positive one. If the number is any integral type, we want to check if our number will fit
        // is in the Error range of CycleInterfaceType or not. In case that our number is over the error range limit
        // then we are considering it an error. To ensure that our number can be verified, since we know it is a
        // positive number we are going to cast the numbers to the largest type
        constexpr bool is_integral{std::is_integral<T>::value};
        if (is_integral) {
            // since our number will be positive any type that is represented on lower or equal number of bits
            // CycleInterfaceType is represented, will be casted to CycleInterfaceType, otherwise it will be casted to
            // the type of T if the number of bits is greater than the CycleInterfaceType size
            using CompareType =
                    typename std::conditional<sizeof(T) <= sizeof(CyclesInterfaceType), CyclesInterfaceType, T>::type;
            /* coverity[result_independent_of_operands] */
            if (static_cast<CompareType>(conversion_number) > static_cast<CompareType>(Cycles::START_ERROR_RANGE))
                return Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES;

            // if our number is not above the error range we are going to simply return it
            return static_cast<CyclesInterfaceType>(conversion_number);
        } else {
            // if our number is a floating number, we are casting our upper limit to the floating type that we are
            // comparing it with.
            if (conversion_number > static_cast<T>(Cycles::START_ERROR_RANGE))
                return Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES;

            // if the type of our number is a floating unit then we are going to make sure to cover the floats in an
            // additional cycle by ceiling the result
            return static_cast<CyclesInterfaceType>(std::ceil(conversion_number));
        }
    }

    /// @brief This method is an overload because if we got a CycleInterfaceType in this conversion method we want to
    /// propagate the error that already was assigned and not to change it to ERROR_INVALID_CONVERSION_TO_CYCLES
    ///
    /// @param conversion_number number we are trying to convert
    /// @return the same number because it is already in CyclesInterfaceType
    inline static constexpr CyclesInterfaceType toCycleInterfaceType(CyclesInterfaceType conversion_number) {
        return conversion_number;
    }
};

}  // namespace VPUNN

#endif  //
