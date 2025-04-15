// Copyright @ 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_RANGES
#define VPUNN_RANGES

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vpu/utils.h"

namespace VPUNN {
// policy for defining a range based on divisibility by a number
class SmartRanges {
private:
    const int lowerBound;                     /// < [
    const int upperBound;                     /// < ]
    const int divisor;                        ///< values have to be divisible with this number
    const std::optional<int> second_divisor;  ///< if present values have to be divisible with this number too only if
                                              ///< they are larger or equal than its value.

public:
    using value_type = int;
    SmartRanges(int lowerBound_, int upperBound_, int divisor_ = 1, std::optional<int> second_div = {})
            : lowerBound(lowerBound_), upperBound(upperBound_), divisor(divisor_), second_divisor{second_div} {
        // maybe we need here to verify if range is valid, something like lowerBound<upperBound
        // take care when range is [-3, -2] if multiplier is positive eg: multiplier=2 when you want to extend
        // upperBound =>[-3, -4] !!THIS IS INVALID RANGE
    }
    SmartRanges(int lowerBound_, int upperBound_, int divisor_, int second_div)
            : lowerBound(lowerBound_), upperBound(upperBound_), divisor(divisor_), second_divisor{second_div} {
    }

    /// @brief: here we verify if a value respect all the range requirements
    /// @param value: the value we want to verify
    /// @param text: a string with information when value does not respect all the requirements
    /// @return true if value respect all the requirements, false if not
    bool is_in(int value, std::string& text) const {
        // const bool belongs{(value >= lowerBound) && (value <= upperBound)};
        // const bool divisible{belongs ? ((value % divisor) == 0 ? true : false) : true};  // test only if belongs

        // const bool second_check_enabled{(belongs && divisible) ? second_divisor.has_value()
        //                                                        : false};  // test only if belongs and divisible
        // const bool second_divisible_OK{
        //         (second_check_enabled && (value >= *second_divisor))         // cases when we are interested
        //                 ? ((value % (*second_divisor)) == 0 ? true : false)  // has to be divisible by second_divisor
        //                 : true  // OK because either not enabled or other conditions not met
        // };

        const bool part_of_range{is_in(value)};

        if (!part_of_range) {
            const bool belongs{(value >= lowerBound) && (value <= upperBound)};
            const bool divisible{belongs ? ((value % divisor) == 0 ? true : false) : true};  // test only if belongs

            const bool second_check_enabled{(belongs && divisible) ? second_divisor.has_value()
                                                                   : false};  // test only if belongs and divisible
            const bool second_divisible_OK{
                    (second_check_enabled && (value >= *second_divisor))  // cases when we are interested
                            ? ((value % (*second_divisor)) == 0 ? true
                                                                : false)  // has to be divisible by second_divisor
                            : true  // OK because either not enabled or other conditions not met
            };
            // error handling only if error, otherwise do not waist time
            text = "";

            if (!belongs) {
                text = " Value :" + std::to_string(value) + " is not in interval [" + std::to_string(lowerBound) +
                       ", " + std::to_string(upperBound) + "]";
            }

            if (!divisible) {
                text += " Value :" + std::to_string(value) + " is not divisible by " + std::to_string(divisor) + "!";
            }
            if (!second_divisible_OK) {
                text += " Value :" + std::to_string(value) + " is not second divisible by " +
                        std::to_string(*second_divisor) + "!";
            }
        }

        return part_of_range;
    }
    bool is_in(int value) const {
        const bool belongs{(value >= lowerBound) && (value <= upperBound)};
        const bool divisible{belongs ? ((value % divisor) == 0 ? true : false) : true};  // test only if belongs

        const bool second_check_enabled{(belongs && divisible) ? second_divisor.has_value()
                                                               : false};  // test only if belongs and divisible
        const bool second_divisible_OK{
                (second_check_enabled && (value >= *second_divisor))         // cases when we are interested
                        ? ((value % (*second_divisor)) == 0 ? true : false)  // has to be divisible by second_divisor
                        : true  // OK because either not enabled or other conditions not met
        };

        const bool part_of_range{belongs && divisible && second_divisible_OK};

        return part_of_range;
    }

    bool is_roundToNextLarger(int& value) const {
        std::vector<int> candidates{};
        candidates.reserve(2);

        if (betweenLimits(value)) {
            // first criteria  divisibility
            {
                const int div1_value{round_up(value, divisor)};
                if (betweenLimits(div1_value)) {
                    candidates.push_back(div1_value);  // candidate 1
                }
            }

            {
                if (second_divisor.has_value()) {
                    const int div2_value{round_up(value , *second_divisor)};
                    if (betweenLimits(div2_value)) {
                        candidates.push_back(div2_value);  // candidate 2
                    }
                }
            }
        }

        // chose smallest candidate that fits to the range
        std::sort(candidates.begin(), candidates.end());
        // iterate candidates and chose the first that fits to the range
        for (const auto& candidate : candidates) {
            if (is_in(candidate)) {
                value = candidate;  // FOUND!
                return true;
            }
        }

        return false;  // no candidate found
    };

    std::optional<int> roundToNextLarger(int value) const {
        if (is_roundToNextLarger(value)) {
            return value;
        }
        return std::nullopt;
    }

    bool betweenLimits(int value) const {
        return (value >= lowerBound) && (value <= upperBound);
    }

    /// @brief multiplies the upper bound of the range by the given value
    /// @param multiplier used to adjust the range upper bound
    ///
    /// @return a new SmartRanges with the upper bound of the range updated based on the multiplier
    SmartRanges multiply_upper(int multiplier) const {
        SmartRanges newRange{getLowerBound(), getUpperBound() * multiplier, divisor, second_divisor};
        return newRange;
    }

    /// @brief multiplies the lower bound of the range by the given value
    /// @param multiplier used to adjust the range lower bound
    ///
    /// @return a new SmartRanges with the lower bound of the range updated based on the multiplier
    SmartRanges multiply_lower(int multiplier) const {
        SmartRanges newRange{getLowerBound() * multiplier, getUpperBound(), divisor, second_divisor};
        return newRange;
    }

    /// @brief increase or decrease (by adding a negative value) the upper bound of the range by the given value
    /// @param added_term used to adjust the range upper bound
    ///
    /// @return a new SmartRanges with the upper bound of the range updated based on the added_term
    SmartRanges add_upper(int added_term) const {
        SmartRanges newRange{getLowerBound(), getUpperBound() + added_term, divisor, second_divisor};
        return newRange;
    }

    /// @brief increase or decrease (by adding a negative value) the lower bound of the range by the given value
    /// @param added_term used to adjust the range lower bound
    ///
    /// @return a new SmartRanges with the lower bound of the range updated based on the added_term
    SmartRanges add_lower(int added_term) const {
        SmartRanges newRange{getLowerBound() + added_term, getUpperBound(), divisor, second_divisor};
        return newRange;
    }

    int getUpperBound() const {
        return this->upperBound;
    }

    int getLowerBound() const {
        return this->lowerBound;
    }
};

}  // namespace VPUNN

#endif  //