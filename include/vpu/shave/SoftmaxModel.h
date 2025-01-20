// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SOFTMAXMODEL_H
#define SOFTMAXMODEL_H

#include <iostream>
#include <list>
#include <random>
#include <vector>

#include "shave_equations.h"
#include "shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN{

/// @brief Enum class for the different types of softmax equations
enum class SoftmaxEquationType {
    Type1,
    Type2,
    Type4,
    Type8,
    Type16,
    Type32
};

static const EnumMap SoftmaxEquationType_ToText{
        link(SoftmaxEquationType::Type1, "Type1"),
        link(SoftmaxEquationType::Type2, "Type2"),
        link(SoftmaxEquationType::Type4, "Type4"),
        link(SoftmaxEquationType::Type8, "Type8"),
        link(SoftmaxEquationType::Type16, "Type16"),
        link(SoftmaxEquationType::Type32, "Type32"),
};

/// @brief Struct for the parameters of a softmax equation for different types of unselected dimensions
class SoftmaxEquationParams {
public:
    FirstDegreeEquation slopeEquation;
    FirstDegreeEquation interceptEquation;
};


class SoftmaxModel : public ShaveCyclesProvider<SoftmaxModel> {
private:
    const DataType dataType;                               ///< data type of the output
    std::array<SoftmaxEquationParams, 6> equations;  ///< array of equations for each type of softmax based on the unselected space
    FirstDegreeEquation baseEquation;                ///< First degree equation which models the unselected being = 1

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::SoftmaxModel& d);

public:
    SoftmaxModel(DataType dataType, float baseSlope, float baseIntercept, SoftmaxEquationParams e1, 
                 SoftmaxEquationParams e2, SoftmaxEquationParams e4, SoftmaxEquationParams e8, 
                 SoftmaxEquationParams e16, SoftmaxEquationParams e32, unsigned int DpuFreq,
                 unsigned int ShvFreq) :
        ShaveCyclesProvider(DpuFreq, ShvFreq), 
        dataType(dataType),
        equations({e1, e2, e4, e8, e16, e32}),
        baseEquation{baseSlope, baseIntercept} {
    }

protected:
    /**
     * @brief Checking the unselected volume to see which multiple of 2 it is in order to determine the type of the equation.
     * It shifts on the last 5 bits to check its multiple
     * 
     * @return The type of equation needed
    */
    SoftmaxEquationType getEquationType(const int& unselected_dimension_size) const {
        if (unselected_dimension_size % 32 == 0) {
			return SoftmaxEquationType::Type32;
		} else if (unselected_dimension_size % 16 == 0) {
            return SoftmaxEquationType::Type16;
        } else if (unselected_dimension_size % 8 == 0) {
            return SoftmaxEquationType::Type8;
        } else if (unselected_dimension_size % 4 == 0) {
            return SoftmaxEquationType::Type4;
		} else if (unselected_dimension_size % 2 == 0) {
			return SoftmaxEquationType::Type2;
		} else {
			return SoftmaxEquationType::Type1;
		}
    }
    /**
     * @brief Inside of a vectorial block it will normalize the values from the same type to a single value
     * 
     * @return the normalized value
    */
    int normalizeUnselectedValue(const int& unselected_dimension_size,const SoftmaxEquationType& type) const {
        
        int normalizedValue = (unselected_dimension_size / 32) * 32;
        
        switch (type) {
            case SoftmaxEquationType::Type1:
				return normalizedValue + 1;
            case SoftmaxEquationType::Type2:
                return normalizedValue + 2;
            case SoftmaxEquationType::Type4:
                return normalizedValue + 4;
            case SoftmaxEquationType::Type8:
                return normalizedValue + 8;
            case SoftmaxEquationType::Type16:
                return normalizedValue + 16;
            case SoftmaxEquationType::Type32:
			default:
            	return unselected_dimension_size;
        }
    }

public:
    /**
     * @brief Returns the time based on the selected and unselected time.
     * 
     * @return the time in us
    */
    float getMicroSeconds(const int& selected_dimension_size, const int& unselected_dimension_size) const {
        
        //There is a special case in which if the base
        if(unselected_dimension_size == 1){
            return baseEquation(selected_dimension_size);
        }
        //Get the equation type
        const auto& type = getEquationType(unselected_dimension_size);
        
        int index = static_cast<int>(type);
        //Retrieve the equation based on it.
        const auto& eqs = equations[index];

        //Calculate the intercept and slope based on the selected volume
        float slope = eqs.slopeEquation(selected_dimension_size);
        float intercept = eqs.interceptEquation(selected_dimension_size);

        // calculate the time with the normalized value of the unselected volume
        return slope * normalizeUnselectedValue(unselected_dimension_size, type) + intercept;
    }

};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::SoftmaxModel& d) {
    stream << "SoftmaxModel: \n"
           << " dtype: \t" << (int)d.dataType << " : " << DataType_ToText.at(static_cast<int>(d.dataType)) << " ;\n"  //
           << " base equation:" << d.baseEquation.slope_ << ", " << d.baseEquation.intercept_ << " ;\n" 
           << " equations: \n";
           int index = 0;
           for (const auto& eq : d.equations) {
           stream << "  type: \t" << SoftmaxEquationType_ToText.at(index) << " ;\n"
           	      << "  slopeEquation: \t" << eq.slopeEquation.slope_ << ", " << eq.slopeEquation.intercept_ << " ;\n"
		   	      << "  interceptEquation: \t" << eq.interceptEquation.slope_ << ", " << eq.interceptEquation.intercept_ << " ;\n";
            index++;
           }
           stream << out_terminator() << "SoftmaxModel";  // terminator
    return stream;
}



}

#endif /*SOFTMAXMODEL_H*/
