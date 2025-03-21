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
#include <sstream>  //
#include <string>

#include "dpu_defaults.h"
#include "dpu_halo.h"
#include "dpu_types.h"
#include "sep_mode.h"
#include "vpu_tensor.h"

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

    std::array<unsigned int, 4> offsets = {0, 0, 0, 0};  ///< offsets relative to the parent DPULayer, L2 API

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

    /// add type of weights, if different than input_0 type
    std::optional<DataType> weight_type{};

    /// textual information about the belonging of this workload. NOt mandatory, used only for logging
    std::string layer_info{""};

    std::optional<bool> weightless_operation{};  ///< operation does not have weights

    /// output tensor can be the same as the input tensor, for Elementwise ops only
    std::optional<bool> in_place_output_memory{};

    /// Superdense memory. ODU specific?
    std::optional<bool> superdense_memory{};

    std::string get_layer_info() const {
        return layer_info;
    }
    void set_layer_info(const std::string& layer_info_name) {
        layer_info = layer_info_name;
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
            if (in_place_output_memory.has_value()) {
                return in_place_output_memory.value();
            } else {  // optional not set. using older/initial mechanism
                return is_preconditions_for_inplace_output();
            }
        }
        return false;  // not elemntwise
    }

    bool is_weightless_operation() const {
        if (is_elementwise_like_operation()) {  // only for elementwise
            if (weightless_operation.has_value()) {
                return weightless_operation.value();
            } else {  // optional not set. using older/initial mechanism
                return is_special_No_weights_situation();
            }
        }
        return false;  // not elemntwise
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
        if (superdense_memory.has_value()) {
            return superdense_memory.value();
        } else {  // optional not set. using older/initial mechanism
            return false;
        }
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
    bool operator==(const DPUWorkload& b) const {
        bool r{true};
        r = r && (device == b.device);
        r = r && (op == b.op);
        r = r && (inputs == b.inputs);
        r = r && (outputs == b.outputs);

        r = r && (kernels == b.kernels);
        r = r && (strides == b.strides);
        r = r && (padding == b.padding);

        r = r && (execution_order == b.execution_order);
        r = r && (activation_function == b.activation_function);

        const float EPSILON{0.00001f};
        auto is_equal = [&EPSILON](float a, float b) {
            return (std::fabs(a - b) < EPSILON);  // very simple since vals around zero
        };
        r = r && (is_equal(act_sparsity, b.act_sparsity));
        r = r && (is_equal(weight_sparsity, b.weight_sparsity));

        r = r && (input_swizzling == b.input_swizzling);
        r = r && (output_swizzling == b.output_swizzling);

        r = r && (output_write_tiles == b.output_write_tiles);
        r = r && (isi_strategy == b.isi_strategy);
        r = r && (weight_sparsity_enabled == b.weight_sparsity_enabled);

        // new halo
        r = r && (halo == b.halo);
        r = r && (sep_activators == b.sep_activators);  // sep

        r = r && (weight_type == b.weight_type);  // weight type

        r = r && (layer_info == b.layer_info);  // layer_info

        r = r && (weightless_operation == b.weightless_operation);      // weightless_operation
        r = r && (in_place_output_memory == b.in_place_output_memory);  // in_place_output_memory
        r = r && (superdense_memory == b.superdense_memory);            // superdense_memory
        return r;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DPUWorkload& d) {
    stream << "Workload: \n"                                                                                        //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << " Operation: \t" << (int)d.op << " : " << Operation_ToText.at(static_cast<int>(d.op))
           << " ;\n"  //

           // inputs and oytputs tensors
           << " input: \t{\n"
           << d.inputs[0] << " } ;\n"  //
           << " output: \t{\n"
           << d.outputs[0] << " } ;\n"  //

           << " kernels: [W,H]  \t{" << d.kernels[Dim::Grid::W] << "," << d.kernels[Dim::Grid::H] << "} ;\n"  //
           << " strides: [W,H]  \t{" << d.strides[Dim::Grid::W] << "," << d.strides[Dim::Grid::H] << "} ;\n"  //
           << " padding: [TBLR] \t{" << d.padding[Dim::TOP] << "," << d.padding[Dim::BOTTOM] << ","           //
           << d.padding[Dim::LEFT] << "," << d.padding[Dim::RIGHT] << "} ;\n"                                 //

           << " execution_order: \t" << (int)d.execution_order << " : "
           << ExecutionMode_ToText.at(static_cast<int>(d.execution_order)) << " ;\n"  //
           << " activation_function: \t" << (int)d.activation_function << " : "
           << ActivationFunction_ToText.at(static_cast<int>(d.activation_function)) << " ;\n"  //

           << " act_sparsity: \t" << d.act_sparsity << " ;\n"        //
           << " weight_sparsity: \t" << d.weight_sparsity << " ;\n"  //

           << " input_swizzling: \t{" << (int)d.input_swizzling[0] << "," << (int)d.input_swizzling[1] << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.input_swizzling[0])) << ","
           << Swizzling_ToText.at(static_cast<int>(d.input_swizzling[1])) << "} ;\n"  //

           << " output_swizzling: \t{" << (int)d.output_swizzling[0] << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.output_swizzling[0])) << "} ;\n"  //

           << " output_write_tiles: \t" << d.output_write_tiles << " ;\n"    //
           << " offsets: \t{" << d.offsets[0] << "," << d.offsets[1] << ","  //
           << d.offsets[2] << "," << d.offsets[3] << "} ;\n"                 //
           << " isi_strategy: \t" << (int)d.isi_strategy << " : "
           << ISIStrategy_ToText.at(static_cast<int>(d.isi_strategy)) << " ;\n"  //
           << " weight_sparsity_enabled: \t" << (int)d.weight_sparsity_enabled << " : "
           << (d.weight_sparsity_enabled ? "true" : "false") << " ;\n"  //
           << d.halo            //<< " ;\n"                                          //
           << d.sep_activators  //<< " ;\n"                                          //
           << " weight_type: \t"
           << (d.weight_type.has_value() ? DataType_ToText.at(static_cast<int>(d.weight_type.value())) : "Same")
           << " ;\n"  //
           //<< "layer_info:" << d.layer_info << " ;\n" //  keep out since affects layer hash (until layer is more
           // decoupled)
           << " weightless_operation: \t"
           << (d.weightless_operation.has_value() ? std::to_string(d.weightless_operation.value()) : "NA") << " ;\n"  //
           << " in_place_output_memory: \t"
           << (d.in_place_output_memory.has_value() ? std::to_string(d.in_place_output_memory.value()) : "NA")
           << " superdense_memory: \t"
           << (d.superdense_memory.has_value() ? std::to_string(d.superdense_memory.value()) : "NA")
           << " ;\n"                           //
           << out_terminator() << "Workload "  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
