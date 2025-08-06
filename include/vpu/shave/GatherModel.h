// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the Software Package)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the third-party-programs.txt or other similarly-named text file included with the
// Software Package for additional details.

#ifndef GATHERMODEL_H
#define GATHERMODEL_H

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
class GatherModel : public ShaveCyclesProvider<GatherModel>{
private:
    const DataType dtype_;
    VariableSlopeMiddleCommonAxesFrstDegEq
            gather_eq_;  ///< input for this eq is size in elements!
                         ///< that gives the time in us for sw op based on the size at output

    const float vector_offset; //< The offset for the vector size
    const unsigned int vector_size;  //< The number of ops for a VPUDevice Ex. 32 operations for NPU40
    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::GatherModel& d);

public:
    using Dimensions = std::array<int, 4>;  ///< size of selected dimensions;  Unused axis have to be ONE! OR ZERO!?

    GatherModel(DataType dtype, float baseSlope, float baseIntercept, float worstSlope, float interSlope, float vectorOffset,
           unsigned int vector_size, unsigned int DpuFreq, unsigned int ShvFreq) 
        : ShaveCyclesProvider<GatherModel> {DpuFreq, ShvFreq},
          dtype_(dtype), 
          gather_eq_{baseSlope, baseIntercept, worstSlope, interSlope},
		  vector_offset(vectorOffset), 
          vector_size(vector_size) {
    }
 
    float getMicroSeconds(const int& output_size_samples, const Dimensions& dimensions_size) const {
        float y_microseconds = gather_eq_(output_size_samples, dimensions_size);

        y_microseconds += compute_vector_offset(output_size_samples, dimensions_size[0]);
        
        return y_microseconds;
    }

protected:
    float compute_vector_offset(const int& total_volume, const int& innermost_dim) const {
        int vector_step =
                ((innermost_dim / vector_size) < 1) ? ((innermost_dim - 1) % vector_size) : (innermost_dim % vector_size);
        vector_step = vector_step >= 0 ? vector_step : 0;  //< Ensure it is a positive number

		return vector_offset * vector_step * (total_volume/innermost_dim);
	}
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::GatherModel& d) {
    stream << "Gather: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " equations: [base_slope, intercept, worst_slope, intermediate_slope]  \t{" 
           << d.gather_eq_.best_case_slope << "," << d.gather_eq_.intercept << ","
           << d.gather_eq_.worst_case_slope << "," << d.gather_eq_.intermediate_case_slope << "} ;\n"  //
           << d.converter                                                                                     //
           << out_terminator() << "Gather"  // terminator
            ;
    return stream;
}

}  // namespace VPUNN
#endif /* SHAVEMODEL1to1_H*/
