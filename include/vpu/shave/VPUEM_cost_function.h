// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPUEM_COSTFUNCTION_H
#define VPUNN_VPUEM_COSTFUNCTION_H

#include <iostream>
#include <vector>
#include <vpu/types.h>
#include <vpu/shave/shave_equations.h>

namespace VPUNN {
class VPUEM_CostFunction {
public:
    const CostFunction3SlopesDescriptor costFunction3SlopesData_;
    const VPUEMPiecewiseEq piecewiseEq_;

    // Constructor
    VPUEM_CostFunction(const CostFunction3SlopesDescriptor& costFunction3SlopesData): costFunction3SlopesData_(costFunction3SlopesData), piecewiseEq_(costFunction3SlopesData) {
    }

    // function to compute the no of cycles
    int getCycles(DataType dtype, const int output_size_bytes, const float cost_curve_ratio) const {
        
        return piecewiseEq_.compute_shave_cycles(dtype, output_size_bytes, cost_curve_ratio);
    }
};

class VPUEMSoftmax_CostFunction {
public:
    const CostFunctionSoftmaxDescriptor costFunctionSoftmaxData_;

    // Constructror
    VPUEMSoftmax_CostFunction(const CostFunctionSoftmaxDescriptor& costFunctionSoftmaxData)
            : costFunctionSoftmaxData_(costFunctionSoftmaxData) {
    }

    int getCycles(DataType dtype, const int h_output_size_bytes, const int hw_output_size_bytes, const int c_output_size_bytes) const {
        return VPUEMSoftmaxEq(costFunctionSoftmaxData_)
                .compute_softmax_shave_cycles(dtype, h_output_size_bytes, hw_output_size_bytes, c_output_size_bytes);
    }
};

class VPUEMSpatial_CostFunction {
public:
    const CostFunctionSpatialDescriptor costFunctionSpatialData_;

    // Constructror
    VPUEMSpatial_CostFunction(const CostFunctionSpatialDescriptor& costFunctionSpatialData)
            : costFunctionSpatialData_(costFunctionSpatialData) {
    }

    int getCycles(DataType dtype, int output_size_bytes) const {
        return VPUEMSpatialEq(costFunctionSpatialData_).compute_spatial_shave_cycles(dtype, output_size_bytes);
    }
};
} // namespace VPUNN
#endif // VPUNN_VPUEM_COSTFUNCTION_H
