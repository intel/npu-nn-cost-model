// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_WORKLOAD_H
#define VPUNN_DPU_WORKLOAD_H

#include <array>
#include <iostream>
#include <map>
#include <optional>
#include <string>

#include "dpu_defaults.h"
#include "dpu_halo.h"
#include "dpu_types.h"
#include "sep_mode.h"
#include "vpu_tensor.h"
#include "profiling_service.h"

namespace VPUNN {

/// @brief The base structure that encodes a DPU workloads
/// Normally the tensors in/out that describe the operation are expected to be the compute tensors
struct DPUWorkload {
    VPUDevice device;  ///< device family, VPU2_0, 2_7, ...
    Operation op;      ///< operation, like convolution, etc

    /// input0 tensors, the data/activation tensor details. This is the Compute Tensor
    /// the weights tensor is deduced.
    std::array<VPUTensor, 1> inputs;

    /// output tensor. This is the Compute Tensor
    std::array<VPUTensor, 1> outputs;

    std::array<unsigned int, 2> kernels;  ///< kernel sizes WH
    std::array<unsigned int, 2> strides;  ///< kernel strides WH
    std::array<unsigned int, 4> padding;  ///< kernel padding Top, Bottom, Left, Right.

    // Padding and positive halo do not work together

    ExecutionMode execution_order;  ///< execution mode

    ActivationFunction activation_function = ActivationFunction::NONE;  ///< operation activation function

    float act_sparsity = 0;     ///< activation (input 0) sparsity level, interval [0..1]
    float weight_sparsity = 0;  ///< weight (input 1) sparsity level, interval [0..1]

    std::array<Swizzling, 2> input_swizzling = {
            default_init_swizzling(),
            default_init_swizzling()};  ///< input tensors swizzling, // use __size as Not available for input 1
    std::array<Swizzling, 1> output_swizzling = {default_init_swizzling()};  ///< output tensors swizzling

    /// @brief broadcast policy, Split Over K situation , In the SOK tiling strategy, weights are split across
    /// the tiles over the K dimension. The DPU in each tile compute a K-slice of the output tensors and
    /// then broadcast the result in each CMX tile, implicitly concatenating the results and having then
    /// all activations completely replicated.
    ///
    /// OWT = The full Output is written to tiles specified. (does not have a direction!) Not limited to SOK situations
    ///  (0, 1 is self, 2,3,4,5,6 is to how many in total).
    /// Individual output halo still have meaning independent of it(owt).
    unsigned int output_write_tiles{1};

    std::array<unsigned int, 4> offsets = {
            0, 0, 0, 0};  ///< offsets relative to the parent DPULayer, L2 API. Used only in intratile splitting

    ///@brief inter slice interconnect strategy , from 2.7 onwards
    /// reflects if the workload is a part of a split larger workload/layer.
    /// when the NN will use halo this will be ignored
    ISIStrategy isi_strategy{ISIStrategy::CLUSTERING};

    bool weight_sparsity_enabled{false};  ///< is sparsity enabled for input_1(weights). This cannot be deduced,is
                                          ///< independent(can be true for sparsity rate =0)

    HaloWorkload halo{};  ///< halo aspects

    /// mechanism to allow (by computing properly the memory) workloads that use SEP. StorageElements Table
    /// for pointers=> memory tensor is different than compute tensor.
    SEPModeInfo sep_activators{};

    /// add type of weights, if different than input_0 type (EISXW-103211)
    std::optional<DataType> weight_type{};

    /// textual information about the belonging of this workload. NOt mandatory, used only for logging
    std::string layer_info{""};

    /// operation does not have weights, in_place_input1, element-wise only
    std::optional<bool> weightless_operation{};

    /// output tensor can be the same as the input tensor, for element-wise ops only
    std::optional<bool> in_place_output_memory{};

    /// Superdense memory. ODU specific, superdense_output
    std::optional<bool> superdense_memory{};

    /// Input autopad - IDU specific
    std::optional<bool> input_autopad{};

    /// Output autopad - ODU specific
    std::optional<bool> output_autopad{};

    /// hint about what cost provider to use
    CostSourceHint cost_source_hint{CostSourceHint::AUTO};

    /// hint about what profiling service backend to use
    /// Hints should not be included in the hash computation or the cout operator, 
    /// as they do not define the workload itself and are highly likely to change without affecting the actual workload.
    ProfilingServiceBackend profiling_service_backend_hint{ProfilingServiceBackend::__size};

    /// MPE engine to be used, SCL by default -- It's a required field for both l2 and l1 APIs.
    /// Controls also what execution order can be used / is valid
    /// In case of dCIM, the execution order is set automatically during sanitization, therefore no optimization is
    /// allowed at layer level.
    MPEEngine mpe_engine{MPEEngine::SCL};

    /// this flag indicates if current operation does also the reduce min/max along with the main operation
    /// to do: output can be a extra 1 element (practically not impacting the memory footprint), or (future) can be a
    /// WxHX1 tensor if the reduce is channel based.
    bool reduce_minmax_op{false};

    // note: the above field is the last one that belongs to DPUWorkload::hash() computation, if you add more than
    // please update the hash, and remember it will invalidate previous hash caches

    std::string get_layer_info() const {
        return layer_info;
    }

    void set_layer_info(const std::string& layer_info_name) {
        layer_info = layer_info_name;
    }

    void set_inplace_input1(bool in_place) {
        weightless_operation = in_place;
    }

    void set_inplace_output(bool in_place) {
        in_place_output_memory = in_place;
    }

    void set_inplace_output_memory(bool in_place) {
        if (is_elementwise_like_operation()) {
            in_place_output_memory = in_place;  // true of false accepted
        } else if (false == in_place) {
            in_place_output_memory = false;
        } else {
            // no change, true cannot be set, remains as before
        }
    }

    bool is_inplace_output_memory() const {
        if (is_elementwise_like_operation()) {  // only for elementwise
            // optional not set. using older/initial mechanism, if set use the value
            return in_place_output_memory.value_or(is_preconditions_for_inplace_output());
        }
        return false;  // not elemntwise
    }
    /// alias of is_inplace_output_memory
    bool is_inplace_output() const {
        return is_inplace_output_memory();
    }

    bool is_weightless_operation() const {
        if (is_elementwise_like_operation()) {  // only for elementwise
            // optional not set. using older/initial mechanism, if set use the value
            return weightless_operation.value_or(is_special_No_weights_situation());
        }
        return false;  // not elemntwise
    }

    /// alias for is_weightless_operation
    bool is_inplace_input1() const {
        return is_weightless_operation();
    }

    /// detect if operation is elementwise fammily
    bool is_elementwise_like_operation() const {
        return ((op == Operation::ELTWISE) ||  //
                (op == Operation::ELTWISE_MUL));
    }

    /// superdense setter. becomes with value
    void set_superdense(bool superdense) {
        superdense_memory = superdense;
    }
    /// superdense getter
    bool is_superdense() const {
        // optional not set. using older/initial mechanism
        return superdense_memory.value_or(false);
    }

    /// in autopad getter
    bool is_input_autopad() const {
        return input_autopad.value_or(false);
    }

    /// out autopad getter
    bool is_output_autopad() const {
        return output_autopad.value_or(false);
    }

    /// in autopad setter
    void set_input_autopad(bool autopad) {
        input_autopad = autopad;
    }

    /// out autopad setter
    void set_output_autopad(bool autopad) {
        output_autopad = autopad;
    }

    /// reduce minmax operation setter
    void set_reduce_minmax_op(bool reduce_minmax) {
        reduce_minmax_op = reduce_minmax;
    }

    /// reduce minmax operation getter
    bool is_reduce_minmax_op() const {
        return reduce_minmax_op;
    }

    void set_all_swizzlings(Swizzling toSet) {
        input_swizzling[0] = toSet;
        input_swizzling[1] = toSet;
        output_swizzling[0] = toSet;
    }
    /// gets the type of the weight tensor, considering also input type in case not set
    DataType get_weight_type() const noexcept {
        // if not set, we assume the same type as input_0
        return weight_type.value_or(inputs[0].get_dtype());
    }

protected:
    /// @brief checks if the memory for input and output have the preconditions to be 1-1 in order to support in place
    /// does not look at operation specific fields, like kernels, etc
    ///
    /// @param w is the workload for which the memory to be computed
    /// @returns true if the preconditions are met, this does not imply that is possible
    bool is_preconditions_for_inplace_output() const {
        const VPUTensor& in{inputs[0]};
        const VPUTensor& out{outputs[0]};
        if ((in.get_layout() == out.get_layout())                             // same layout
            && (is_same_datatype_footprint(in.get_dtype(), out.get_dtype()))  // same type size
        ) {
            return true;
        } else {
            return false;
        }
    }

    /// @brief finds out if (assuming elementwise situation) the input_1 is not existing, no weighs
    /// This is in case we have a NCEPermute or Quantize/DeQuantize operation
    ///
    /// @param w is the workload for which the memory to be computed
    /// @returns true if looks like  input_1 is not to be considered
    bool is_special_No_weights_situation() const {
        const VPUTensor& in{inputs[0]};
        const VPUTensor& out{outputs[0]};

        // this is a temporary speculative(contextual) implementation. The final solution will have a explicit field in
        // the workload specifying that the weights are not present

        if ((in.get_layout() != out.get_layout())  // layout change
            || (!is_same_datatype_footprint(in.get_dtype(),
                                            out.get_dtype()))  // from a type size to another, not only F16 to [u]int8
        ) {
            return true;
        } else {
            return false;
        }
    }

    // operations/methods
public:
    /// equality test operator
    bool operator==(const DPUWorkload& b) const;

    /// less than operator for std::map compatibility
    bool operator<(const DPUWorkload& b) const;

    /// compute hash for cache key usage, directly from DPUWorkload fields
    /// Uses the same fnv1a_hash function as NNDescriptor, but without preprocessing
    uint32_t hash() const noexcept;

    DPUWorkload(const DPUWorkload&) = default;
    DPUWorkload& operator=(const DPUWorkload&) = default;
    DPUWorkload() = default;
    /// default destructor explicit stated here for gcov problems.
    ~DPUWorkload() = default;
};

}  // namespace VPUNN

namespace VPUNN {

/// Stream output operator for DPUWorkload
std::ostream& operator<<(std::ostream& stream, const VPUNN::DPUWorkload& d);

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
