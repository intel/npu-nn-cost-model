// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/validation/data_dpu_operation.h"

#include "core/primitive_hash.h"
#include "vpu/validation/interface_operations_behavior.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu/workload_hash.h"

namespace VPUNN {

size_t DPUOperation::hash() const {
    auto hash_tensor_info = [](uint32_t h, const TensorInfo& tensor) {
        h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(tensor.width));
        h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(tensor.height));
        h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(tensor.channels));
        h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(tensor.batch));
        h = PrimitiveHash::hash_enum(h, tensor.datatype);
        h = PrimitiveHash::hash_enum(h, tensor.layout);
        h = PrimitiveHash::hash_bool(h, tensor.sparsity_enabled);
        h = PrimitiveHash::hash_float(h, tensor.sparsity);
        h = PrimitiveHash::hash_enum(h, tensor.swizzling);
        return h;
    };

    uint32_t h = fnv_offset_basis;

    h = PrimitiveHash::hash_enum(h, device);
    h = PrimitiveHash::hash_enum(h, operation);

    h = hash_tensor_info(h, input_0);
    h = hash_tensor_info(h, input_1);
    h = hash_tensor_info(h, output_0);

    h = PrimitiveHash::hash_enum(h, execution_order);
    h = PrimitiveHash::hash_enum(h, activation_function);

    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.width));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.height));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.pad_top));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.pad_bottom));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.pad_left));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.pad_right));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.stride_width));
    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(kernel.stride_height));

    h = PrimitiveHash::hash_uint32(h, static_cast<uint32_t>(output_write_tiles));
    h = PrimitiveHash::hash_enum(h, isi_strategy);

    h = WorkloadHash::hash_halo(h, halo);
    h = WorkloadHash::hash_sep(h, sep_activators);

    h = PrimitiveHash::hash_bool(h, weightless_operation);
    h = PrimitiveHash::hash_bool(h, in_place_output_memory);
    h = PrimitiveHash::hash_bool(h, superdense);
    h = PrimitiveHash::hash_bool(h, input_autopad);
    h = PrimitiveHash::hash_bool(h, output_autopad);

    h = PrimitiveHash::hash_enum(h, mpe_engine);
    h = PrimitiveHash::hash_bool(h, reduce_minmax_op);

    return h;
}

DPUOperation::DPUOperation(const DPUWorkload& w, const IDeviceValidValues& config)
        : DPUOperation{w}  // delegate to the main constructor
{
    auto& operation_behaviour = config.get_specific_behaviour(this->operation);  // may throw
    operation_behaviour.deduce_input_1_shape_and_layout(input_0, output_0, config, kernel, input_1);
}

}  // namespace VPUNN