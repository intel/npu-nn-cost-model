#ifndef VPUEM_COSTFUNCTION_H
#define VPUEM_COSTFUNCTION_H

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
#endif