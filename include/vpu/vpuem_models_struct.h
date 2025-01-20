#ifndef VPUNN_VPUEM_MODELS_STRUCT_H
#define VPUNN_VPUEM_MODELS_STRUCT_H

#include <iostream>

namespace VPUNN {

struct CostFunction3SlopesDescriptor {
    int unroll_;
    float offset_;
    std::array<float, 3> slope_;
    int loop_offset_ = 0;

    CostFunction3SlopesDescriptor(int unroll, float offset, std::array<float, 3> slope, int loop_offset = 0)
            : unroll_(unroll), offset_(offset), slope_(slope), loop_offset_(loop_offset) {
    }
};

struct VPUEMSoftmaxParamsDesciptor {
    int unroll_;
    int unroll_offset_;
    int unroll_slope_;
    int unroll_overhead_;
    int vector_offset_;
    int vector_slope0_;
    int vector_slope_;
    int vector_overhead_;
    int scalar_offset_;
    int scalar_slope0_;
    int scalar_slope_;
    int scalar_overhead_;

    VPUEMSoftmaxParamsDesciptor(int unroll, int unroll_offset, int unroll_slope, int unroll_overhead, int vector_offset,
                                int vector_slope0, int vector_slope, int vector_overhead, int scalar_offset,
                                int scalar_slope0, int scalar_slope, int scalar_overhead)
            : unroll_(unroll),
              unroll_offset_(unroll_offset),
              unroll_slope_(unroll_slope),
              unroll_overhead_(unroll_overhead),
              vector_offset_(vector_offset),
              vector_slope0_(vector_slope0),
              vector_slope_(vector_slope),
              vector_overhead_(vector_overhead),
              scalar_offset_(scalar_offset),
              scalar_slope0_(scalar_slope0),
              scalar_slope_(scalar_slope),
              scalar_overhead_(scalar_overhead) {
    }
};
struct CostFunctionSoftmaxDescriptor {
    bool spatial_ = false;
    int simd_;
    std::vector<VPUEMSoftmaxParamsDesciptor> functionParams_;

    CostFunctionSoftmaxDescriptor(bool spatial, int simd, std::vector<VPUEMSoftmaxParamsDesciptor> functionParams)
            : spatial_(spatial), simd_(simd), functionParams_(std::move(functionParams)) {
    }
};

struct CostFunctionSpatialDescriptor {
    int C_divider;
    int C_offset;
    int C_factor;
    std::string Order;
    std::vector<float> slope;
};

}  // namespace VPUNN

#endif  // VPUNN_VPUEM_MODELS_STRUCT_H