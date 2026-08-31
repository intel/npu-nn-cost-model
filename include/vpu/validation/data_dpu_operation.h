// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_VALIDATOR_DATA_DPU_OPERATION_H
#define VPUNN_VPU_VALIDATOR_DATA_DPU_OPERATION_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "vpu/dpu_defaults.h"
#include "vpu/dpu_dtypes_dimension_info.h"
#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/profiling_service.h"
#include "vpu/workload_semantics_info.h"

#include "vpu/validation/serializable_tensor.h"

namespace VPUNN {

class IDeviceValidValues;  // cannot include the header here, circular dependency

template <class T>
using Values = std::vector<T>;  ///< Values type container
using Channels = Values<int>;   ///< int container, available channel values

/// @brief kernel related informations, including stride and padding
struct KernelInfo {
    int height{1};
    int width{1};

    int pad_bottom{0};
    int pad_left{0};
    int pad_right{0};
    int pad_top{0};

    int stride_height{1};
    int stride_width{1};

    /// constructor based on information from a DPUWorkload
    explicit KernelInfo(const DPUWorkload& w)
            : height{static_cast<int>(w.kernels[Dim::Grid::H])},
              width{static_cast<int>(w.kernels[Dim::Grid::W])},
              pad_bottom{static_cast<int>(w.padding[Dim::Padding::BOTTOM])},
              pad_left{static_cast<int>(w.padding[Dim::Padding::LEFT])},
              pad_right{static_cast<int>(w.padding[Dim::Padding::RIGHT])},
              pad_top{static_cast<int>(w.padding[Dim::Padding::TOP])},
              stride_height{static_cast<int>(w.strides[Dim::Grid::H])},
              stride_width{static_cast<int>(w.strides[Dim::Grid::W])} {
    }
    KernelInfo() = default;
};

/// @brief local type describing a workload
/// easy to change and adapt without touching the DPUWorkload interface
/* coverity[rule_of_three_violation:FALSE] */
struct DPUOperation {
    VPUDevice device{};  ///< device family, VPU2_0, 2_7, ...
    Operation operation{};

    TensorInfo input_0;  ///< activators compute tensor
    TensorInfo input_1;  ///< weights. NOte: different operations will use differently the tensor shape to represent
                         ///< weights

    TensorInfo output_0;  //< compute tensor

    ExecutionMode execution_order{};  ///< execution mode

    KernelInfo kernel;

    ActivationFunction activation_function{ActivationFunction::NONE};

    int output_write_tiles{1};  //< broadcast policy
    ISIStrategy isi_strategy{ISIStrategy::CLUSTERING};

    // new halo
    HaloWorkload halo{};                 ///< halo aspects
    TensorInfo input_0_memory_dense{};   ///< activators memory tensor. no sparsity or SEP, only halo influence
    TensorInfo output_0_memory_dense{};  ///< memory tensor for output. no sparsity , only halo influence

    SEPModeInfo sep_activators;  // SEP mode , input via Storage elements. Not compatible with halo?

    bool weightless_operation{false};  ///< operation does not have weights

    /// output tensor can be the same as the input tensor, for Elementwise ops only
    bool in_place_output_memory{false};

    /// Superdense memory. ODU specific?
    /// optional. By default(no value) is considered as missing = false
    bool superdense{false};

    bool input_autopad{false};

    bool output_autopad{false};

    /// hint about what cost provider to use
    CostSourceHint cost_source_hint{CostSourceHint::AUTO};

    /// hint about what profiling service backend to use
    ProfilingServiceBackend profiling_service_backend_hint{ProfilingServiceBackend::__size};

    ///< MPE engine to be used, SCL by default
    MPEEngine mpe_engine{MPEEngine::SCL};

    /// this flag indicates if current operation does also the reduce min/max along with the main operation
    bool reduce_minmax_op{false};

    void set_intended_split(ISIStrategy strategy, unsigned int nTiles) {
        isi_strategy = strategy;
        output_write_tiles = static_cast<int>(nTiles);
    }

    /// constructor from a DPUWorkload.
    /// input_1 (weights) tensor is not filled with shape
    /// @todo:  provide a version that is able to init also the wts
    explicit DPUOperation(const DPUWorkload& w)
            : device{w.device},
              operation{w.op},
              input_0{w.inputs[0]},
              output_0{w.outputs[0]},
              execution_order{w.execution_order},
              kernel{w},
              activation_function{w.activation_function},
              output_write_tiles{static_cast<int>(w.output_write_tiles)},
              isi_strategy{w.isi_strategy},
              halo{w.halo},                      // copy halo
              sep_activators{w.sep_activators},  // copy sep
              weightless_operation{w.is_weightless_operation()},
              in_place_output_memory{w.is_inplace_output_memory()},
              superdense{w.is_superdense()},
              input_autopad{w.is_input_autopad()},
              output_autopad{w.is_output_autopad()},
              cost_source_hint{w.cost_source_hint},
              profiling_service_backend_hint{w.profiling_service_backend_hint},
              mpe_engine{w.mpe_engine},
              reduce_minmax_op{w.reduce_minmax_op} {
        // from WL to tensors
        input_0.swizzling = w.input_swizzling[0];
        input_0.sparsity = w.act_sparsity;

        {                                                                 // partial filling of input 1
            input_1.datatype = w.weight_type.value_or(input_0.datatype);  // default to activator type

            input_1.swizzling = w.input_swizzling[1];

            input_1.sparsity_enabled = w.weight_sparsity_enabled;
            input_1.sparsity = w.weight_sparsity;
        }
        output_0.swizzling = w.output_swizzling[0];

        input_0_memory_dense = compute_dense_input_memory_tensor(input_0, halo);
        output_0_memory_dense = compute_dense_output_memory_tensor(output_0, halo);
    }
    DPUOperation() = default;
    DPUOperation(const DPUOperation& r)
            : device{r.device},
              operation{r.operation},
              input_0{r.input_0},
              input_1{r.input_1},
              output_0{r.output_0},
              execution_order{r.execution_order},
              kernel{r.kernel},
              activation_function{r.activation_function},
              output_write_tiles{r.output_write_tiles},
              isi_strategy{r.isi_strategy},
              halo{r.halo},
              input_0_memory_dense{r.input_0_memory_dense},
              output_0_memory_dense{r.output_0_memory_dense},
              sep_activators{r.sep_activators}, /*_member_map{}*/
              weightless_operation{r.weightless_operation},
              in_place_output_memory{r.in_place_output_memory},
              superdense{r.superdense},
              input_autopad(r.input_autopad),
              output_autopad(r.output_autopad),
              cost_source_hint{r.cost_source_hint},
              profiling_service_backend_hint{r.profiling_service_backend_hint},
              mpe_engine{r.mpe_engine},
              reduce_minmax_op{r.reduce_minmax_op} {
    }

    DPUOperation(DPUOperation&) = delete;
    DPUOperation(const DPUOperation&&) = delete;
    DPUOperation(DPUOperation&&) = delete;

    DPUOperation& operator=(const DPUOperation&) = delete;
    DPUOperation& operator=(DPUOperation&) = delete;
    DPUOperation& operator=(DPUOperation) = delete;

    ~DPUOperation() = default;

    size_t hash() const;

    /// @brief checks if input/output memory preconditions permit in-place output.
    bool is_preconditions_for_inplace_output() const {
        const TensorInfo& in{input_0};
        const TensorInfo& out{output_0};
        return WorkloadSemanticsInfo::is_preconditions_for_inplace_output(in.layout, in.datatype, out.layout,
                                                                           out.datatype);
    }

    /// @brief determines if the operation can be treated as weightless in elementwise-like cases.
    bool is_special_No_weights_situation() const {
        const TensorInfo& in{input_0};
        const TensorInfo& out{output_0};

        return WorkloadSemanticsInfo::is_special_no_weights_situation(in.layout, in.datatype, out.layout,
                                                                       out.datatype);
    }

    /// @brief checks if the operation is elementwise-like (ELTWISE or ELTWISE_MUL).
    bool is_elementwise_like_operation() const {
        return WorkloadSemanticsInfo::is_elementwise_like_operation(operation);
    }

    DPUWorkload clone_as_DPUWorkload() const {
        const auto& in = input_0;
        const auto& out = output_0;

        DPUWorkload wl{
                device,
                operation,
                {in.toVPUTensor()},   // input dimensions
                {out.toVPUTensor()},  // output dimensions
                {static_cast<unsigned int>(kernel.width), static_cast<unsigned int>(kernel.height)},  // kernels
                {static_cast<unsigned int>(kernel.stride_width),
                 static_cast<unsigned int>(kernel.stride_height)},  // strides
                {static_cast<unsigned int>(kernel.pad_top), static_cast<unsigned int>(kernel.pad_bottom),
                 static_cast<unsigned int>(kernel.pad_left), static_cast<unsigned int>(kernel.pad_right)},  // padding
                execution_order  // execution mode
        };  // looks like local  object , but  hope  for Return Value Optimization (RVO)

        wl.activation_function = activation_function;

        wl.act_sparsity = input_0.sparsity;
        wl.weight_sparsity = input_1.sparsity;

        wl.input_swizzling[0] = input_0.swizzling;
        wl.input_swizzling[1] = input_1.swizzling;

        wl.output_swizzling[0] = output_0.swizzling;

        // wl.offsets;  // NOT SET remains zero/init
        wl.output_write_tiles = output_write_tiles;
        wl.isi_strategy = isi_strategy;

        wl.weight_sparsity_enabled = input_1.sparsity_enabled;

        wl.halo = halo;  // halo aspects
        wl.sep_activators = sep_activators;

        wl.weight_type = input_1.datatype;  // this will make the optional as existing!

        wl.weightless_operation = weightless_operation;  // the optional will be set as existing with the value
        wl.set_inplace_output_memory(in_place_output_memory);

        wl.set_superdense(superdense);

        wl.input_autopad = input_autopad;    // the optional will be set as existing with the value
        wl.output_autopad = output_autopad;  // the optional will be set as existing with the value

        wl.cost_source_hint = cost_source_hint;
        wl.profiling_service_backend_hint = profiling_service_backend_hint;
        wl.mpe_engine = mpe_engine;

        wl.reduce_minmax_op = reduce_minmax_op;

        return wl;
    }
    /// knowing input compute tensor and halo will calculate the input memory tensor, without considering sparsity
    /// or other indirection like SEP. Is like what would be the memory tensor if dense and no SEP (pointer
    /// indirection) tricks
    static TensorInfo compute_dense_input_memory_tensor(const TensorInfo& compute_t, const HaloWorkload& halo) {
        TensorInfo t{compute_t};  // same as compute tensor first

        const auto& in_halo{halo.input_0_halo};
        //  extension will be negative(memory reduction) if halo(positive halo),
        // or positive (memory increase) ,memory is larger, if negative halo, but we consume less (prev layer wrote
        //  more)

        auto newDimension = [](const long long crtDimension, const int oneEndHalo, const int otherEndHalo) {
            const int oneExt = -oneEndHalo;
            const int twoExt = -otherEndHalo;
            const long long newDim = crtDimension + (oneExt + twoExt);
            return (newDim > 0 ? newDim : 0);  // limit to zero
        };

        t.height = newDimension(t.height, in_halo.top, in_halo.bottom);
        t.width = newDimension(t.width, in_halo.left, in_halo.right);
        t.channels = newDimension(t.channels, in_halo.front, in_halo.back);

        return t;
    }

    /// knowing output compute tensor and halo will calculate the output memory tensor, without considering sparsity
    /// or other indirection
    /// COnsiders also negative inbound halo = forget halo
    static TensorInfo compute_dense_output_memory_tensor(const TensorInfo& compute_t, const HaloWorkload& halo) {
        TensorInfo t{compute_t};  // same as compute tensor first
        const auto& inbound_halo{halo.output_0_inbound_halo};
        //  extension can be only positive = how many elements the other tiles are writing here
        //  more)

        auto newDimensionWithInbound = [](const long long crtDimension, int oneEndHalo, int otherEndHalo) {
            // if halo is negative means we write less here, so memory dim is smaller, but we cannot have
            // negative dim if both negative , cannot overlap a negative one cannot forget more than existing
            // dim finally the new Dim has to be >=0
            if (oneEndHalo < 0) {
                if (-oneEndHalo >= crtDimension) {
                    oneEndHalo = -static_cast<int>(crtDimension);  // limit to max forget
                }
            }
            if (otherEndHalo < 0) {
                if (-otherEndHalo >= crtDimension) {
                    otherEndHalo = -static_cast<int>(crtDimension);  // limit to max forget
                }
            }
            // if both negative , cannot overlap

            // if both negative, their sum cannot "forget" more than the current dimension
            if (oneEndHalo < 0 && otherEndHalo < 0) {
                int total_forget = -oneEndHalo + -otherEndHalo;
                if (total_forget >= crtDimension) {
                    // limit so that we do not go below zero
                    oneEndHalo = -static_cast<int>(crtDimension);
                    otherEndHalo = 0;
                }
            }

            const long long newDim = crtDimension + oneEndHalo + otherEndHalo;

            return newDim >= 0 ? newDim : 0;  // can be zero? but should not be!
        };

        t.height = newDimensionWithInbound(t.height, inbound_halo.top, inbound_halo.bottom);
        t.width = newDimensionWithInbound(t.width, inbound_halo.left, inbound_halo.right);
        t.channels = newDimensionWithInbound(t.channels, inbound_halo.front, inbound_halo.back);

        return t;
    }

    /// update memory tensors in accordance with compute tensors and Halo
    void resyncronize_memory_tensors() {
        input_0_memory_dense = compute_dense_input_memory_tensor(input_0, halo);
        output_0_memory_dense = compute_dense_output_memory_tensor(output_0, halo);
    }

    /// @brief constructor based on DPUWorkload and a device valid values for initializing also the input_1 tensor
    ///
    /// @param w the source DPUWorkload
    /// @param config the device valid values for this device
    /// @throws std::runtime_error if the operation  is not supported
    DPUOperation(const DPUWorkload& w, const IDeviceValidValues& config);

    friend std::ostream& operator<<(std::ostream& stream, const DPUOperation& d);
};

inline std::ostream& operator<<(std::ostream& stream, const KernelInfo& d) {
    stream << "KernelInfo: \n"                                                              //
           << " kernels: [W,H]  \t{" << d.width << "," << d.height << "} ;\n"               //
           << " strides: [W,H]  \t{" << d.stride_width << "," << d.stride_width << "} ;\n"  //
           << " padding: [TBLR] \t{" << d.pad_top << "," << d.pad_bottom << ","             //
           << d.pad_left << "," << d.pad_right << "} ;\n"                                   //
            ;
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const DPUOperation& d) {
    stream << "DPUOperation-Workload: \n"                                                                           //
           << " device: \t" << (int)d.device << " : " << VPUDevice_ToText.at(static_cast<int>(d.device)) << " ;\n"  //
           << " Operation: \t" << (int)d.operation << " : " << Operation_ToText.at(static_cast<int>(d.operation))
           << " ;\n"  //

           // inputs and oytputs tensors
           << " input act: \t{\n"
           << d.input_0 << " } ;\n"  //
           << " input w: \t{\n"
           << d.input_1 << " } ;\n"  //
           << " output: \t{\n"
           << d.output_0 << " } ;\n"  //

           << d.kernel << "\n"  //

           << " execution_order: \t" << (int)d.execution_order << " : "
           << ExecutionMode_ToText.at(static_cast<int>(d.execution_order)) << " ;\n"  //

           << " output_write_tiles: \t" << d.output_write_tiles << " ;\n"  //

           << " isi_strategy: \t" << (int)d.isi_strategy << " : "
           << ISIStrategy_ToText.at(static_cast<int>(d.isi_strategy)) << " ;\n"  //
           << d.halo                                                             //
           << " input  dense memo: \t{" << d.input_0_memory_dense << " } ;\n"
           << " output dense memo: \t{" << d.output_0_memory_dense << " } ;\n"
           << d.sep_activators                                                                     //
           << " weightless_operation: \t" << std::to_string(d.weightless_operation) << " ;\n"      //
           << " in_place_output_memory: \t" << std::to_string(d.in_place_output_memory) << " ;\n"  //
           << " superdense: \t" << std::to_string(d.superdense) << " ;\n"                          //
           << " input_autopad: \t" << std::to_string(d.input_autopad) << " ;\n"                    //
           << " output_autopad: \t" << std::to_string(d.output_autopad) << " ;\n"                  //
           << " mpe_engine: \t" << (int)d.mpe_engine << " : " << MPEEngine_ToText.at(static_cast<int>(d.mpe_engine))
           << " ;\n"
           << " reduce_minmax_op: \t" << (d.reduce_minmax_op ? "true" : "false") << " ;\n"  //
           << " cost source hint: \t" << (int)d.cost_source_hint << " : "
           << CostSourceHint_ToText.at(static_cast<int>(d.cost_source_hint)) << " ;\n";
    return stream;
}
}  // namespace VPUNN

#endif  //
