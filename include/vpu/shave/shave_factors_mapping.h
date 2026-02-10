// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.


#ifndef SHAVE_FACTORS_MAPPING_H
#define SHAVE_FACTORS_MAPPING_H

#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

#include <unordered_map>
#include <variant>

namespace VPUNN {
// All possible types for speed-up factors
using ShaveSpeedUpFactorType = float;  // extend with other types if needed

/**
 * @brief Lookup table for SHAVE operation speed-up factors, wrapper of a unordered_map
 * 
 * This class maintains a mapping of operation names to their corresponding speed-up factors used in SHAVE.
 * Speed-up factors represent performance multipliers that can be applied to base operation costs.
 * 
 * It is important so generated classes that inherits `FactorsLookUpTable` should populate the `speed_up_factors`
 * 
 * The factors for now can be only floating-point values, but in the future can be extended to other types
 * or classes.
 * @details The flow is following: 
 * You have csv file for shave factors. You call the corresponding function in the CMake to generate at build time the desired "populated class". 
 * That "populated class" is actually a class that inherits the `FactorsLookUpTable`, and in the constructor it calls the add methods to populate 
 * the internal factors mapping variable of `FactorsLookUpTable` (check template in `src/shave/ShaveFactorsPopulation.h.in`). 
 * So when declared, variable of the generated class is also initialized. Then, you can use that generated class for templatized class 
 * of instance holder so it will be populated with shave operations with corresponding factors from generated class.
 * 
 * @note The class provides a default speed-up factor of 1.0f for operations not found
 *       in the lookup table, ensuring neutral performance impact.
 */
class FactorsLookUpTable {
private:
    ///@brief internal storage for factors
    std::unordered_map<std::string, ShaveSpeedUpFactorType> speed_up_factors;

public:
    ///@brief Adds a speed-up factor for a given operation name
    /// This method should match the pattern used in the generated PopulatedFactorsLUT classes defined in CMake
    void add(const std::string& name, ShaveSpeedUpFactorType speed_up) {
        speed_up_factors[name] = speed_up;
    }

    ///@brief Retrieves the speed-up factor for a given operation name
    ShaveSpeedUpFactorType getOperatorFactor(const std::string& name) const {
        auto it = speed_up_factors.find(name);
        if (it != speed_up_factors.end()) {
            return it->second;
        }

        // if not found return type-dependent default
        return default_value();
    }

    ///@brief Checks if the lookup table has been populated with any factors
    /// If generated class didn't add any factors, this will return false
    bool is_populated() const {
        return !speed_up_factors.empty();
    }

private:
    /// @brief Returns the default speed-up factor value when an operation is not found in the lookup table
    /// @return Default speed-up factor (1.0f)
    ShaveSpeedUpFactorType default_value() const {
        // If no values exist yet, prefer float as default
        return 1.0f;
    }
};

} // namespace VPUNN

#endif
