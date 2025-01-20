// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef MVNMODEL_H
#define MVNMODEL_H

#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <vector>

#include "shave_equations.h"
#include "shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

class MVN6OneAxisModel : public ShaveCyclesProvider<MVN6OneAxisModel> {
private:
    const DataType dtype_;                    ///< the data type of the output ()
    VariableSlopeFirstDegreeEquation unroll;  ///< input for this eq is size in elements!
                                              ///< that gives the time in us for sw op based on the size at output

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::MVN6OneAxisModel& d);

public:
    MVN6OneAxisModel(DataType dtype, float slope, float intercept, float alpha, float maximum_diff_slope,
                     unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<MVN6OneAxisModel>{DpuFreq, ShvFreq},
              dtype_(dtype),
              unroll{slope, intercept, alpha, maximum_diff_slope} {
    }

public:
    /**
     * gets us time, specific parameters
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& output_size_samples, const int& selected_dimension_size) const {
        float y_microseconds = unroll(output_size_samples, selected_dimension_size);  // all in samples
        return y_microseconds;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::MVN6OneAxisModel& d) {
    stream << "MVN6OneAxisModel: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " equations: [slope,intercept, alpha, max_slope_delta]  \t{" << d.unroll.slope_ << ","
           << d.unroll.intercept_ << "," << d.unroll.alpha_ << "," << d.unroll.maximum_diff_slope << "} ;\n"  //
           << d.converter                                                                                     //
           << out_terminator() << "MVN6OneAxisModel"  // terminator
            ;
    return stream;
}

/// MOdel for MVN 6 with 1,2,3,4 axis selected,
/// the axis are always innermost
class MVN6MultiAxisModel : public ShaveCyclesProvider<MVN6MultiAxisModel> {
private:
    const DataType dtype_;  ///< the data type of the output ()

    VariableSlope4MultiAxesFrstDegEq equation;  ///< input for this eq is size in elements!
                                                ///< that gives the time in us for sw op based on the size at output

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::MVN6MultiAxisModel& d);

public:
    using Dimensions = std::array<int, 4>;  ///< size of selected dimensions;  Unused axis have to be ONE! OR ZERO!?
    /**
     * @brief Construct a new Shave Model 1 to 1 object
     */
    MVN6MultiAxisModel(DataType dtype, float best_slope, float intercept, float alpha, float worst_case_slope,
                       float slope_delta_diff, unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<MVN6MultiAxisModel>{DpuFreq, ShvFreq},
              dtype_(dtype),
              equation{best_slope, intercept, alpha, worst_case_slope, slope_delta_diff} {
    }

public:
    /**
     * @brief Get the time in us for the the activation based on the output size in bytes
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& output_size_samples, const Dimensions& selected_dimensions_size) const {
        float y_microseconds = equation(output_size_samples, selected_dimensions_size);  // all in samples

        return y_microseconds;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::MVN6MultiAxisModel& d) {
    stream << "MVN6MultiAxisModel: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " equation:" << d.equation << " ;\n"                                                                //
           << d.converter                                                                                         //
           << out_terminator() << "MVN6MultiAxisModel"  // terminator
            ;
    return stream;
}

class MVNSimple2and3AxisModel : public ShaveCyclesProvider<MVNSimple2and3AxisModel> {
private:
    const DataType dtype_;  ///< the data type of the output ()

    // for the next ones: the slope has to be scaled up with outermost dim (4th) ( for S3 = unselected)
    const FirstDegreeEquation base_outermost;  ///< scaled with last(4th) dimension. For S3 is the UnselectedVolume, for
                                               ///< S2 is the outermost Dim

    const FirstDegreeEquation thirdmost_unselected_support;  ///< x is (thirdmostdim-1). Becomes Zero term for S3 (no
                                                             ///< thirdmost unselected)

    // for the next ones:  the slope has to be scaled up with unselected volume (or 4thd dim for S3 )
    const FirstDegreeEquation base_support;  //< x is SV/(unroll * vector)
    const FirstDegreeEquation mod_support;   //< x is mod(SV,(unroll * vector))  / vector
    const FirstDegreeEquation vector;        //< x is mod(SV, vector)

    const unsigned int vector_size;  //< The number of ops for a VPUDevice Ex. 32 operations for NPU40
    const unsigned int unroll_size;  //< The number to determine what is the size of a block

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::MVNSimple2and3AxisModel& d);

public:
    /**
     * @brief Construct a new Shave Model 1 to 1 object
     */
    MVNSimple2and3AxisModel(DataType dtype, float baseSlope, float baseIntercept, float thirdMostSupportSlope,
                            float baseSupportSlope, float mod8SupportSlope, float vectorSlope,unsigned int vector_size,
                             unsigned int unroll_size, unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<MVNSimple2and3AxisModel>{DpuFreq, ShvFreq},
              dtype_(dtype),
              base_outermost{baseSlope, baseIntercept},
              thirdmost_unselected_support{thirdMostSupportSlope, 0.0f},
              base_support{baseSupportSlope, 0.0f},
              mod_support{mod8SupportSlope, 0.0f},
              vector{vectorSlope, 0.0f},
              vector_size(vector_size),
              unroll_size(unroll_size) {
    }

public:
    /**
     * @brief Get the time in us for the activation based on formula. FOr Special 2dim or 3dim MVN.
     * At least first 2 dims are selected
     * Third dim might be selected or not, the formula can cancel the term of third dim by setting the slope =0
     * Fourth dim is always unselected.
     *
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& selected_volume, const int& un_selected_volume, const int& outermost_dimension,
                          const int& thirdmost_dimension) const {
        const int unselected{un_selected_volume};  // will be a factor for changing slope
        float y_microseconds{0.0f};
        {  //(BaseSlope*OutermostDim+BaseIntercept)
            const float term_base{base_outermost(outermost_dimension)};
            y_microseconds += term_base;
        }

        {  //((ThirdmostDim−1)*ThirdmostSupportSlope∗OutermostDim)
            const float term_third{thirdmost_unselected_support((thirdmost_dimension - 1)) * outermost_dimension};
            y_microseconds += term_third;  // must add zero if no thirdmost slope
        }

        {
            const int input_support{static_cast<int>(selected_volume / (unroll_size * vector_size))};
            const float term_support{base_support(input_support) * unselected};
            y_microseconds += term_support;
        }

        {
            const int input_mod{static_cast<int>((selected_volume % (unroll_size * vector_size)) / vector_size)};
            const float term_mod{mod_support(input_mod) * unselected};
            y_microseconds += term_mod;
        }

        {
            const int input_vector{
                    (selected_volume > static_cast<int>(vector_size - 1)) ? (selected_volume % static_cast<int>(vector_size))
                                          : ((selected_volume % static_cast<int>(vector_size)) - 1)  // 7 is missing (gap there?)
            };
            const float term_vector{vector(input_vector) * unselected};
            y_microseconds += term_vector;
        }

        return y_microseconds;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::MVNSimple2and3AxisModel& d) {
    stream << "MVNSimple2and3AxisModel: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " BaseSlope       : " << d.base_outermost.slope_ << " , Base Intercept: " << d.base_outermost.intercept_
           << " ;\n"                                                                         //
           << " ThirdmostSupportSlope: " << d.thirdmost_unselected_support.slope_ << " ;\n"  //
           << " BaseSupportSlope: " << d.base_support.slope_ << " ;\n"                       //
           << " Mod8SupportSlope: " << d.mod_support.slope_ << " ;\n"                        //
           << " VectorSlope     : " << d.vector.slope_ << " ;\n"                             //
           << " UnrollSize      : " << d.unroll_size << " ;\n"                               //
           << " VectorSize      : " << d.vector_size << " ;\n"                               //

           << d.converter                                                                    //
           << out_terminator() << "MVNSimple2and3AxisModel"                                  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif /* SHAVEMODEL1to1_H*/
