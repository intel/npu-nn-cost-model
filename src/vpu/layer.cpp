#include "vpu/layer.h"
#include "vpu/device_layer_properties/device_layer_properties_holder.h"

// This .cpp file provides the implementation for DPULayer constructors declared in the header
// It is necessary because inside these constructors we use LayerPropertiesHolder for device-specific configuration, and we
// want to avoid circular dependencies by including the header here instead of in the header file

namespace VPUNN {

DPULayer::DPULayer(VPUDevice device, Operation op, std::array<VPUTensor, 1> inputs, std::array<VPUTensor, 1> outputs,
                   std::array<unsigned int, 2> kernels, std::array<unsigned int, 2> strides,
                   std::array<unsigned int, 4> padding) {
    this->device = device;
    this->op = op;
    this->inputs = inputs;
    this->outputs = outputs;
    this->kernels = kernels;
    this->strides = strides;
    this->padding = padding;
    this->execution_order =
            LayerPropertiesHolder::get_properties(device).getValidDefaultExecutionMode(inputs[0]);

    // rest of fields are the workload default (CLUSTERING, output write tiles  =1)
}

DPULayer::DPULayer(const DPUWorkload& wl): DPUWorkload(wl) {
    this->execution_order =
            LayerPropertiesHolder::get_properties(device).getValidDefaultExecutionMode(inputs[0]);

    // rest of fields are the workload default (CLUSTERING, output write tiles  =1)
}
}  // namespace VPUNN
