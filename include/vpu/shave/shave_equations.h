// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_EQUATIONS_H
#define SHAVE_EQUATIONS_H

#include <array>
#include <iostream>
#include <vector>

#include "vpu/vpuem_models_struct.h"
#include "vpu/types.h"
#include "vpu/datatype_collection_size.h"


namespace VPUNN {

constexpr int NOUNROLL{1};

/**
 * @brief Defines the structure of the first degree equation of the line for each shave operation.
 * The equation is slope_ * size + intercept_
 * @param slope_ is defined by time in us divided by size of output bytes
 * @param intercept_ is defined  by time in us
 */

class FirstDegreeEquation {
public:
    float slope_;      ///< Represents us/bytes
    float intercept_;  ///< Represents us
    /**
     * @brief Overload for operator() which calculates the time based on the first degree equation of the Shave
     * activation and the size in bytes of the output
     *
     * @param size of output in bytes
     * @return the time in us
     */
    float operator()(const int& size) const {
        return slope_ * size + intercept_;
    }
};

/**
 * @brief Defines the structure of the first degree equation of the line for each shave operation.
 * The ecuation is slope_ * size + intercept_
 * @param slope_ is defined by time in us divided by size of output bytes
 * @param intercept_ is defined  by time in us
 */

class VariableSlopeFirstDegreeEquation {
public:
    float slope_;  ///< Represents base slope us/bytes or samples , (happens eg when selected axes has all the tensor
                   ///< elements)
    float intercept_;          ///< Represents us
    float alpha_;              ///< Represents the coeff for creating a different slope
    float maximum_diff_slope;  ///< Represents the slope for worst case, (when selected axis is one)

    /**
     * @brief Overload for operator() which calculates the time based on the first degree equation of the Shave
     * activation and the size of the output
     *
     * @param size of output elements
     * @param selected_dimension it is the special dimension's size that influences the slope
     * @return the time in us
     */
    float operator()(const int& size, const int& selected_dimension) const {
        const float coeff{getCoeff(size, selected_dimension)};
        const float slope = getSlope(coeff);
        return slope * size + intercept_;
    }

    float getCoeff(const int& size, const int& selected_dimension) const {
        // -1 because cannot be less than one element per dimension
        return (1.0F - static_cast<float>(selected_dimension - 1) / static_cast<float>(size - 1)) *
               std::exp(-alpha_ * (selected_dimension - 1)) / static_cast<float>(selected_dimension);
    }
    float getSlope(const float& coeff) const {
        return (slope_ + coeff * maximum_diff_slope);
    }
};
// Future create a VariableSlopeFirstDegEq interface. so we can implement the different functions separately
class VariableSlopeMiddleCommonAxesFrstDegEq {
public:
    using Dimensions = std::array<int, 4>;  ///< size of selected dimensions;  Unused axis have to be ONE! OR ZERO!?
    float best_case_slope;  ///< Represents base slope us/bytes or samples , (happens eg when selected axes has all
                                ///< the tensor elements)
    float intercept;  ///< Represents us
    float worst_case_slope;  ///< Represents the slope for worst case, when we have volume in the outermost axe of the tensor.
    float intermediate_case_slope; ///< Represents the slope for intermediate case, when the volume is in the middle of the tensor.

     /**
     * @brief Overload for operator() which calculates the time based on the first degree equation of the Shave
     * activation and the size of the output
     *
     * @param size of output elements
     * @param selected_dimension it is the special dimension's size that influences the slope
     * @return the time in us
     */
    float operator()(const int& size, const Dimensions& selected_dim) const {
        const int outermost_volume{dim_val(selected_dim[3])};

        const std::array<float, 2> coeff{compute_worst_coeff(size, outermost_volume),  //
                                         compute_intermediate_coeff(selected_dim)};

        const float slope = get_slope(coeff);
        return slope * size + intercept;
    }



    float compute_worst_coeff(const int& outermostDim, const int& totalVolume) const {
        const float denominator{(float)(outermostDim)};
        const float numerator{(float)(totalVolume)};  // var A, Var B is total volume

        return volume_ratio(numerator, denominator);
    }

    float compute_intermediate_coeff(const Dimensions& selected_dim) const {
        const float numerator{(float)(selected_dim[1] * selected_dim[2])};
        const float denominator{(float)(selected_dim[0] * selected_dim[3]) * numerator};  // var A, Var B is total volume

        return volume_ratio(numerator, denominator);
    }

    float get_slope(const std::array<float, 2>& c) const {
        float slope{best_case_slope + c[0] * worst_case_slope + c[1] * intermediate_case_slope};

        return slope;
    }
    /// filter not to let dimension smaller than 1. (like zero), destroy the selected volume
    int dim_val(const int& dimension_value) const {
        return (dimension_value > 1) ? dimension_value : 1;
    }

    float volume_ratio(const float& numerator, const float& denominator) const {
        // -1 because cannot be less than one element per dimension
        // zero/zero is zero!
        const float coef{(numerator > 1) ? (numerator - 1) / (denominator - 1) : (0.0F)};
        return coef;
    }
};

class VariableSlope4MultiAxesFrstDegEq {
public:
    using Dimensions = std::array<int, 4>;  ///< size of selected dimensions;  Unused axis have to be ONE! OR ZERO!?

    float best_case_slope;  ///< Represents base slope us/bytes or samples , (happens eg when selected axes has all the
                            ///< tensor elements)
    float intercept;        ///< Represents us
    float alpha;            ///< Represents the exponential coefficient for creating a different slope

    float worst_case_slope;  ///< Represents the slope for worst case, (when selected axis is one),
    float slope_delta_diff;  ///< constant added proportional with coefficients 1,2,3

    float get_delta_max() const {
        return worst_case_slope - best_case_slope;
    }

    /**
     * @brief Overload for operator() which calculates the time based on the first degree equation of the Shave
     * activation and the size of the output
     *
     * @param size of output elements
     * @param selected_dimension it is the special dimension's size that influences the slope
     * @return the time in us
     */
    float operator()(const int& size, const Dimensions& selected_dim) const {
        const int volume_selected{dim_val(selected_dim[0]) *  //
                                  dim_val(selected_dim[1]) *  //
                                  dim_val(selected_dim[2]) *  //
                                  dim_val(selected_dim[3])};

        const std::array<float, 4> coeff{compute_coef_0(size, volume_selected),  //
                                         compute_coef_1(selected_dim),           //
                                         compute_coef_2(selected_dim),           //
                                         compute_coef_3(selected_dim)};

        const float slope = get_slope(coeff);
        return slope * size + intercept;
    }


    float compute_coef_0(const int& v_total, const int& v_selected) const {
        // the factor will be one in case of v_selected =1, avoids 0/0 situation , makes it =0;
        const float factor{(1.0F - volume_ratio(static_cast<float>(v_selected), static_cast<float>(v_total)))};
        const float coef{(v_selected > 1)
                                 ? (factor * std::exp(-alpha * (v_selected - 1)) / static_cast<float>(v_selected))
                                 : (1.0F)};

        return coef;
    }

    float compute_coef_1(const Dimensions& sel_dim) const {
        const float numerator{(float)(sel_dim[1] * sel_dim[2] * sel_dim[3])};
        const float denominator{sel_dim[0] * numerator};  // var A, Var B is total volume

        return volume_ratio(numerator, denominator);
    }
    float compute_coef_2(const Dimensions& sel_dim) const {
        const float numerator{(float)(sel_dim[2] * sel_dim[3])};
        const float denominator{sel_dim[0] * sel_dim[1] * numerator};  // var A, Var B is total volume

        return volume_ratio(numerator, denominator);
        ;
    }
    float compute_coef_3(const Dimensions& sel_dim) const {
        const float numerator{(float)(sel_dim[3])};
        const float denominator{sel_dim[0] * sel_dim[1] * sel_dim[2] * numerator};  // var A, Var B is total volume

        return volume_ratio(numerator, denominator);
    }

    /// there are more variants to combine coefficients
    float get_slope(const std::array<float, 4>& c) const {
        float slope{best_case_slope + c[0] * get_delta_max()};

        slope += c[1] * slope_delta_diff;
        slope += c[2] * c[1] * slope_delta_diff;         // variant 2
        slope += c[3] * c[2] * c[1] * slope_delta_diff;  // variant 2

        return slope;
    }
    /// filter not to let dimension smaller than 1. (like zero), destroy the selected volume
    int dim_val(const int& dimension_value) const {
        return (dimension_value > 1) ? dimension_value : 1;
    }

    float volume_ratio(const float& numerator, const float& denominator) const {
        // -1 because cannot be less than one element per dimension
        // zero/zero is zero!
        const float coef{(numerator > 1) ? (numerator - 1) / (denominator - 1) : (0.0F)};
        return coef;
    }
};

class VPUEMPiecewiseEq {
public:

    const CostFunction3SlopesDescriptor costFunction3SlopesData_;

    // Constructor
    VPUEMPiecewiseEq(const CostFunction3SlopesDescriptor& costFunction3SlopesData)
            : costFunction3SlopesData_(costFunction3SlopesData) {
    }

    int compute_shave_cycles(DataType dtype, const int output_size_bytes, const float cost_curve_ratio) const {
        // commuting from bytes to operations
        // if i have a number of bytes and we know the data type then we know how many ops we made
        
        int blk_size = compute_elements_count_from_bytes(output_size_bytes, dtype);

        int vector_size = compute_elements_count_from_bytes(16, dtype);

        int unroll_size = vector_size * costFunction3SlopesData_.unroll_;
        int unroll_number = int(blk_size / unroll_size);
        int unroll_location = unroll_number * unroll_size;
        int vector_number = int((blk_size - unroll_location) / vector_size);
        int vector_location = vector_number * vector_size;
        int scalar_number = blk_size - unroll_location - vector_location;

        const auto cycles = (unroll_location / costFunction3SlopesData_.slope_[0] +
                            vector_location / costFunction3SlopesData_.slope_[1] +
                            scalar_number / costFunction3SlopesData_.slope_[2]) / cost_curve_ratio;

        return (int)cycles;
    }

};

class VPUEMSoftmaxEq {
public:
    const CostFunctionSoftmaxDescriptor costFunctionSoftmaxData_;

    VPUEMSoftmaxEq(const CostFunctionSoftmaxDescriptor costFunctionSoftmaxData)
            : costFunctionSoftmaxData_(costFunctionSoftmaxData) {
    }
    
    int compute_softmax_shave_cycles(DataType dtype, const int h_output_size_bytes, const int hw_output_size_bytes, const int c_output_size_bytes,
                                           float cost_ratio = 1.0) {
        
        int hw_elements_size = 1;
        int h_elements_size = 1;
        /* coverity[divide_by_zero] */
        int c_elements_size = c_output_size_bytes / dtype_to_bytes(dtype);

        if (hw_output_size_bytes != 1) {
            hw_elements_size = hw_output_size_bytes;
            h_elements_size = h_output_size_bytes;
            c_elements_size = c_output_size_bytes;
        }

        int vector_size = 16 / dtype_to_bytes(dtype);
        int unroll_size = vector_size * costFunctionSoftmaxData_.functionParams_[0].unroll_;
        if (costFunctionSoftmaxData_.spatial_) {
            int temp = h_elements_size;
            int width = hw_elements_size / h_elements_size;
            h_elements_size = c_elements_size / vector_size;
            c_elements_size = temp * width * vector_size;
            hw_elements_size = h_elements_size * 1;
        }

        int unroll_num = c_elements_size / unroll_size;
        int vector_num = (c_elements_size - unroll_num * unroll_size) / vector_size;
        int scalar_num = c_elements_size - unroll_num * unroll_size - vector_num * vector_size;

        if (unroll_num){
            int vector_cycles, scalar_cycles;
          
            int unroll_cycles = (int) (costFunctionSoftmaxData_.functionParams_[0].unroll_slope_ * unroll_num / cost_ratio);
            if (vector_num == 0) {
                vector_cycles = 0;
            } else {
                vector_cycles = (int) costFunctionSoftmaxData_.functionParams_[0].vector_slope0_ +
                                    costFunctionSoftmaxData_.functionParams_[0].vector_slope_ * (vector_num - 1);
                vector_cycles /= (int) cost_ratio;
            }

            if (scalar_num == 0) {
                scalar_cycles = 0;
            } else if (scalar_num == 1) {
                scalar_cycles = (int) (costFunctionSoftmaxData_.functionParams_[0].scalar_slope0_ / cost_ratio);
            } else {
                scalar_cycles = costFunctionSoftmaxData_.functionParams_[0].scalar_slope0_ +
                                    costFunctionSoftmaxData_.functionParams_[0].scalar_slope_ * (scalar_num - 1);
                scalar_cycles /= (int) cost_ratio;
            }

            int loop_overhead = costFunctionSoftmaxData_.functionParams_[0].unroll_overhead_ * (hw_elements_size - 1);

            return costFunctionSoftmaxData_.functionParams_[0].unroll_offset_ +
                   (unroll_cycles + vector_cycles + scalar_cycles) * hw_elements_size + loop_overhead;
        } else if (vector_num){
            int vector_cycles = (int) (costFunctionSoftmaxData_.functionParams_[0].vector_slope_ * vector_num / cost_ratio);
            int scalar_cycles;

            if (scalar_num == 0) {
                scalar_cycles = 0;
            } else {
                scalar_cycles = costFunctionSoftmaxData_.functionParams_[0].scalar_slope0_ +
                                costFunctionSoftmaxData_.functionParams_[0].scalar_slope_ * (scalar_num - 1);
                scalar_cycles /= (int) cost_ratio;

            }

            int loop_overhead = costFunctionSoftmaxData_.functionParams_[0].vector_overhead_ * (hw_elements_size - 1);
            return costFunctionSoftmaxData_.functionParams_[0].vector_offset_ +
                   (vector_cycles + scalar_cycles) * hw_elements_size + loop_overhead;
        } else if (scalar_num) {
            int scalar_cycles = (int)(costFunctionSoftmaxData_.functionParams_[0].scalar_slope_ * scalar_num / cost_ratio);
            int loop_overhead = costFunctionSoftmaxData_.functionParams_[0].scalar_overhead_*(hw_elements_size - 1);

            return costFunctionSoftmaxData_.functionParams_[0].scalar_offset_ + scalar_cycles * hw_elements_size +
                   loop_overhead;
        }

        return -1;
    
    }
};

class VPUEMSpatialEq {
public:
    const CostFunctionSpatialDescriptor costFunctionSpatialData_;

    // Constructor
    VPUEMSpatialEq(const CostFunctionSpatialDescriptor& costFunctionSpatialData)
            : costFunctionSpatialData_(costFunctionSpatialData) {
    }

    int compute_spatial_shave_cycles(DataType dtype, const int output_size_bytes) {
        /* coverity[divide_by_zero] */
        int c_elements_size = output_size_bytes / dtype_to_bytes(dtype);
        float slope = costFunctionSpatialData_.slope[0];
        int cycles = (int) (1 / slope) * (c_elements_size);
        return cycles;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::VariableSlope4MultiAxesFrstDegEq& d) {
    stream << "\t{ "
           << "base_slope: " << d.best_case_slope << ","         //
           << "intercept: " << d.intercept << ","                //
           << "worst_case_slope: " << d.worst_case_slope << ","  //
           << "slope_delta_diff: " << d.slope_delta_diff << ","  //
           << "alpha: " << d.alpha << ","                        //
           << "}"                                                //
            ;
    return stream;
}

}  // namespace VPUNN

#endif /* SHAVEMODEL1to1_H*/