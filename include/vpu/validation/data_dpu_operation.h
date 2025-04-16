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

#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "core/serializer.h"
#include "vpu/dpu_defaults.h"
#include "vpu/dpu_types.h"
#include "vpu/dpu_workload.h"
#include "vpu/serializer_utils.h"

namespace VPUNN {

class IDeviceValidValues;  // cannot include the header here, circular dependency

template <class T>
using Values = std::vector<T>;  ///< Values type container
using Channels = Values<int>;   ///< int container, available channel values

/// @brief holds info for a tensor.
struct TensorInfo {
    long long height{0};
    long long width{0};
    long long channels{0};
    long long batch{1};
    DataType datatype{DataType::UINT8};
    Layout layout{Layout::ZXY};  // same as ZMAJOR
    float sparsity{0.0F};
    bool sparsity_enabled{false};
    Swizzling swizzling{default_init_swizzling()};

    /// constructor based on DPUworkload related VPUTensor structure
    explicit TensorInfo(const VPUTensor& t)
            : height{static_cast<int>(t.height())},  // Y
              width{static_cast<int>(t.width())},    // X
              channels{static_cast<int>(t.z())},
              batch{static_cast<int>(t.b())},
              datatype{t.get_dtype()},
              layout{t.get_layout()},
              sparsity_enabled{t.get_sparsity()} {
    }

    TensorInfo() = default;

    /// @brief Get the size in samples
    /// @return how many elements are in this tensor shape
    long long numberOfElements() const {
        return height * width * channels * batch;
    }
};

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

    using _ref_supported_type =
            std::variant<std::reference_wrapper<VPUDevice>, std::reference_wrapper<Operation>,
                         std::reference_wrapper<DataType>, std::reference_wrapper<Layout>,
                         std::reference_wrapper<Swizzling>, std::reference_wrapper<ActivationFunction>,
                         std::reference_wrapper<ExecutionMode>, std::reference_wrapper<ISIStrategy>,
                         std::reference_wrapper<DimType>, std::reference_wrapper<long long>,
                         std::reference_wrapper<int>, std::reference_wrapper<float>, std::reference_wrapper<bool>,
                         VPUNN::SetGet_MemberMapValues>;

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
              superdense{w.is_superdense()} {
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
              superdense{r.superdense} {
    }

    DPUOperation(DPUOperation&) = delete;
    DPUOperation(const DPUOperation&&) = delete;
    DPUOperation(DPUOperation&&) = delete;

    DPUOperation& operator=(const DPUOperation&) = delete;
    DPUOperation& operator=(DPUOperation&) = delete;
    DPUOperation& operator=(DPUOperation) = delete;

    ~DPUOperation() = default;

    DPUWorkload clone_as_DPUWorkload() const {
        const auto& in = input_0;
        const auto& out = output_0;

        DPUWorkload wl{
                device,
                operation,
                {VPUTensor({static_cast<unsigned int>(in.width), static_cast<unsigned int>(in.height),
                            static_cast<unsigned int>(in.channels), static_cast<unsigned int>(in.batch)},
                           in.datatype, in.layout, in.sparsity_enabled)},  // input dimensions
                {VPUTensor({static_cast<unsigned int>(out.width), static_cast<unsigned int>(out.height),
                            static_cast<unsigned int>(out.channels), static_cast<unsigned int>(out.batch)},
                           out.datatype, out.layout, out.sparsity_enabled)},  // output dimensions
                {static_cast<unsigned int>(kernel.width), static_cast<unsigned int>(kernel.height)},  // kernels
                {static_cast<unsigned int>(kernel.stride_width),
                 static_cast<unsigned int>(kernel.stride_height)},  // strides
                {static_cast<unsigned int>(kernel.pad_top), static_cast<unsigned int>(kernel.pad_bottom),
                 static_cast<unsigned int>(kernel.pad_left), static_cast<unsigned int>(kernel.pad_right)},  // padding
                execution_order  // execution mode
        };                       // looks like local  object , but  hope  for Return Value Optimization (RVO)

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

        wl.weightless_operation = weightless_operation;
        wl.set_inplace_output_memory(in_place_output_memory);

        wl.set_superdense(superdense);

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
    static TensorInfo compute_dense_output_memory_tensor(const TensorInfo& compute_t, const HaloWorkload& halo) {
        TensorInfo t{compute_t};  // same as compute tensor first
        const auto& inbopund_halo{halo.output_0_inbound_halo};
        //  extension can be only positive = how many elements the other tiles are writing here
        //  more)

        auto newDimensionWithInbound = [](const long long crtDimension, const int oneEndHalo, const int otherEndHalo) {
            const long long newDim = crtDimension + (oneEndHalo + otherEndHalo);
            return newDim;
        };

        t.height = newDimensionWithInbound(t.height, inbopund_halo.top, inbopund_halo.bottom);
        t.width = newDimensionWithInbound(t.width, inbopund_halo.left, inbopund_halo.right);
        t.channels = newDimensionWithInbound(t.channels, inbopund_halo.front, inbopund_halo.back);

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

    const std::unordered_map<std::string, _ref_supported_type> _member_map{
            {"device", std::ref(device)},
            {"operation", std::ref(operation)},
            {"input_0_batch", std::ref(input_0.batch)},
            {"input_0_channels", std::ref(input_0.channels)},
            {"input_0_height", std::ref(input_0.height)},
            {"input_0_width", std::ref(input_0.width)},
            {"input_1_batch", std::ref(input_1.batch)},
            {"input_1_channels", std::ref(input_1.channels)},
            {"input_1_height", std::ref(input_1.height)},
            {"input_1_width", std::ref(input_1.width)},
            {"input_sparsity_enabled", std::ref(input_0.sparsity_enabled)},
            {"weight_sparsity_enabled", std::ref(input_1.sparsity_enabled)},
            {"input_sparsity_rate", std::ref(input_0.sparsity)},
            {"weight_sparsity_rate", std::ref(input_1.sparsity)},
            {"execution_order", std::ref(execution_order)},
            {"activation_function", std::ref(activation_function)},
            {"kernel_height", std::ref(kernel.height)},
            {"kernel_width", std::ref(kernel.width)},
            {"kernel_pad_bottom", std::ref(kernel.pad_bottom)},
            {"kernel_pad_left", std::ref(kernel.pad_left)},
            {"kernel_pad_right", std::ref(kernel.pad_right)},
            {"kernel_pad_top", std::ref(kernel.pad_top)},
            {"kernel_stride_height", std::ref(kernel.stride_height)},
            {"kernel_stride_width", std::ref(kernel.stride_width)},
            {"output_0_batch", std::ref(output_0.batch)},
            {"output_0_channels", std::ref(output_0.channels)},
            {"output_0_height", std::ref(output_0.height)},
            {"output_0_width", std::ref(output_0.width)},
            {"input_0_datatype", std::ref(input_0.datatype)},
            {"input_0_layout", std::ref(input_0.layout)},
            {"input_0_swizzling", std::ref(input_0.swizzling)},
            {"input_1_datatype", std::ref(input_1.datatype)},
            {"input_1_layout", std::ref(input_1.layout)},
            {"input_1_swizzling", std::ref(input_1.swizzling)},
            {"output_0_datatype", std::ref(output_0.datatype)},
            {"output_0_layout", std::ref(output_0.layout)},
            {"output_0_swizzling", std::ref(output_0.swizzling)},
            {"output_sparsity_enabled", std::ref(output_0.sparsity_enabled)},
            {"isi_strategy", std::ref(isi_strategy)},
            {"output_write_tiles", std::ref(output_write_tiles)},

            {"input_0_halo_top", std::ref(halo.input_0_halo.top)},
            {"input_0_halo_bottom", std::ref(halo.input_0_halo.bottom)},
            {"input_0_halo_left", std::ref(halo.input_0_halo.left)},
            {"input_0_halo_right", std::ref(halo.input_0_halo.right)},
            {"input_0_halo_front", std::ref(halo.input_0_halo.front)},
            {"input_0_halo_back", std::ref(halo.input_0_halo.back)},

            {"output_0_halo_top", std::ref(halo.output_0_halo.top)},
            {"output_0_halo_bottom", std::ref(halo.output_0_halo.bottom)},
            {"output_0_halo_left", std::ref(halo.output_0_halo.left)},
            {"output_0_halo_right", std::ref(halo.output_0_halo.right)},
            {"output_0_halo_front", std::ref(halo.output_0_halo.front)},
            {"output_0_halo_back", std::ref(halo.output_0_halo.back)},

            {"output_0_halo_broadcast_top", std::ref(halo.output_0_halo_broadcast_cnt.top)},
            {"output_0_halo_broadcast_bottom", std::ref(halo.output_0_halo_broadcast_cnt.bottom)},
            {"output_0_halo_broadcast_left", std::ref(halo.output_0_halo_broadcast_cnt.left)},
            {"output_0_halo_broadcast_right", std::ref(halo.output_0_halo_broadcast_cnt.right)},
            {"output_0_halo_broadcast_front", std::ref(halo.output_0_halo_broadcast_cnt.front)},
            {"output_0_halo_broadcast_back", std::ref(halo.output_0_halo_broadcast_cnt.back)},

            {"output_0_halo_inbound_top", std::ref(halo.output_0_inbound_halo.top)},
            {"output_0_halo_inbound_bottom", std::ref(halo.output_0_inbound_halo.bottom)},
            {"output_0_halo_inbound_left", std::ref(halo.output_0_inbound_halo.left)},
            {"output_0_halo_inbound_right", std::ref(halo.output_0_inbound_halo.right)},
            {"output_0_halo_inbound_front", std::ref(halo.output_0_inbound_halo.front)},
            {"output_0_halo_inbound_back", std::ref(halo.output_0_inbound_halo.back)},

            {"sep_enabled", std::ref(sep_activators.sep_activators)},
            {"sep_w",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.storage_elements_pointers.set_width(value);
                     }
                 }
                 return sep_activators.storage_elements_pointers.width();
             }},
            {"sep_h",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.storage_elements_pointers.set_height(value);
                     }
                 }
                 return sep_activators.storage_elements_pointers.height();
             }},
            {"sep_c",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.storage_elements_pointers.set_channels(value);
                     }
                 }
                 return sep_activators.storage_elements_pointers.channels();
             }},
            {"sep_b",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.storage_elements_pointers.set_batches(value);
                     }
                 }
                 return sep_activators.storage_elements_pointers.batches();
             }},
            {"sep_act_w",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.actual_activators_input.set_width(value);
                     }
                 }
                 return sep_activators.actual_activators_input.width();
             }},
            {"sep_act_h",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.actual_activators_input.set_height(value);
                     }
                 }
                 return sep_activators.actual_activators_input.height();
             }},
            {"sep_act_c",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.actual_activators_input.set_channels(value);
                     }
                 }
                 return sep_activators.actual_activators_input.channels();
             }},
            {"sep_act_b",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     VPUNN::DimType value;
                     if (is_unsigned_int(s, value)) {
                         sep_activators.actual_activators_input.set_batches(value);
                     }
                 }
                 return sep_activators.actual_activators_input.batches();
             }},
            {"sep_no_sparse_map", std::ref(sep_activators.no_sparse_map)},
            {"in_place_input1",  // weightless_operation
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     setWeightlessOperation(s);
                 }
                 return weightless_operation;
             }},
            {"in_place_output",
             [this](bool set_mode, std::string s) -> VPUNN::DimType {
                 if (set_mode) {
                     setInPlaceOutputMemory(s);
                 }
                 return in_place_output_memory;
             }},
            {"superdense_output", std::ref(superdense)},
    };

    static const std::vector<std::string>& _get_member_names() {
        static std::vector<std::string> names = std::vector<std::string>{
                "device",
                "operation",
                "input_0_batch",
                "input_0_channels",
                "input_0_height",
                "input_0_width",
                "input_1_batch",
                "input_1_channels",
                "input_1_height",
                "input_1_width",
                "input_sparsity_enabled",
                "weight_sparsity_enabled",
                "input_sparsity_rate",
                "weight_sparsity_rate",
                "execution_order",
                "activation_function",
                "kernel_height",
                "kernel_width",
                "kernel_pad_bottom",
                "kernel_pad_left",
                "kernel_pad_right",
                "kernel_pad_top",
                "kernel_stride_height",
                "kernel_stride_width",
                "output_0_batch",
                "output_0_channels",
                "output_0_height",
                "output_0_width",
                "input_0_datatype",
                "input_0_layout",
                "input_0_swizzling",
                "input_1_datatype",
                "input_1_layout",
                "input_1_swizzling",
                "output_0_datatype",
                "output_0_layout",
                "output_0_swizzling",
                "output_sparsity_enabled",
                "isi_strategy",
                "output_write_tiles",
                "input_0_halo_top",
                "input_0_halo_bottom",
                "input_0_halo_left",
                "input_0_halo_right",
                "input_0_halo_front",
                "input_0_halo_back",
                "output_0_halo_top",
                "output_0_halo_bottom",
                "output_0_halo_left",
                "output_0_halo_right",
                "output_0_halo_front",
                "output_0_halo_back",
                "output_0_halo_broadcast_top",
                "output_0_halo_broadcast_bottom",
                "output_0_halo_broadcast_left",
                "output_0_halo_broadcast_right",
                "output_0_halo_broadcast_front",
                "output_0_halo_broadcast_back",
                "output_0_halo_inbound_top",
                "output_0_halo_inbound_bottom",
                "output_0_halo_inbound_left",
                "output_0_halo_inbound_right",
                "output_0_halo_inbound_front",
                "output_0_halo_inbound_back",
                "sep_enabled",
                "sep_w",
                "sep_h",
                "sep_c",
                "sep_b",
                "sep_act_w",
                "sep_act_h",
                "sep_act_c",
                "sep_act_b",
                "sep_no_sparse_map",
                "in_place_input1",    // not the same name as the attribute weightless_operation
                "in_place_output",    // not the same name as the attribute in_place_output_memory
                "superdense_output",  // not the same name as the attribute superdense
        };

        return names;
    }

    size_t hash() const {
        std::size_t combined_hash = 0;
        for (const auto& [key, value] : _member_map) {
            std::visit(
                    [&combined_hash](auto&& arg) {
                        if constexpr (std::is_same_v<std::decay_t<std::remove_reference_t<decltype(arg)>>,
                                                     VPUNN::SetGet_MemberMapValues>) {
                            // arg have two parameters first one is false and that means that arg is in get_mode
                            // (function will just return a value), second parameter could be any value, in get_mode
                            // its value doesn't matter
                            combined_hash ^= std::hash<int>{}(arg(false, "")) + 0x9e3779b9 + (combined_hash << 6) +
                                             (combined_hash >> 2);
                        } else {
                            using argtype = std::decay_t<std::remove_reference_t<decltype(arg.get())>>;
                            combined_hash ^= std::hash<argtype>{}(arg) + 0x9e3779b9 + (combined_hash << 6) +
                                             (combined_hash >> 2);
                        }
                    },
                    value);
        }
        return combined_hash;
    }

    /// detect if operation is elementwise fammily
    bool is_elementwise_like_operation() const {
        return ((operation == Operation::ELTWISE) ||  //
                (operation == Operation::ELTWISE_MUL));
    }

protected:
    // test if a number is an unsigned int, if true we assign to result the value
    // if number is not an unsigned int, we return false, else return true
    static bool is_unsigned_int(const std::string& s, VPUNN::DimType& result) {
        // empty string
        if (s.empty())
            return false;

        int value;

        try {
            value = std::stoi(s);  // string to int

            // if result is negative => invalid value
            if (value < 0)
                return false;

        } catch (const std::invalid_argument&) {  // s is not a valid number, can contain characters that are not digits
                                                  // or number sign
            return false;
        } catch (const std::out_of_range&) {
            return false;
        }

        result = static_cast<unsigned int>(value);
        return true;
    }

    void setInPlaceOutputMemory(const std::string& s) {
        in_place_output_memory = false;  // default value

        VPUNN::DimType value;
        if (is_unsigned_int(s, value)) {  // valid input
            if (value != 0) {
                in_place_output_memory = true;
            }
        } else {  // no input so we compute internal the value
            if (is_elementwise_like_operation() && is_preconditions_for_inplace_output()) {
                in_place_output_memory = true;
            }
        }
    }

    void setWeightlessOperation(const std::string& s) {
        weightless_operation = false;  // default value

        VPUNN::DimType value;
        if (is_unsigned_int(s, value)) {  // valid input
            if (value != 0) {
                weightless_operation = true;
            }
        } else {  // no input so we compute internal the value
            if (is_elementwise_like_operation() && is_special_No_weights_situation()) {
                weightless_operation = true;
            }
        }
    }

    /// @brief checks if the memory for input and output have the preconditions to be 1-1 in order to support in place
    /// does not look at operation specific fields, like kernels, etc
    ///
    /// @param w is the workload for which the memory to be computed
    /// @returns true if the preconditions are met, this does not imply that is possible
    bool is_preconditions_for_inplace_output() const {
        const TensorInfo& in{input_0};
        const TensorInfo& out{output_0};
        if ((in.layout == out.layout)                                   // same layout
            && (is_same_datatype_footprint(in.datatype, out.datatype))  // same type size
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
        const TensorInfo& in{input_0};
        const TensorInfo& out{output_0};

        // this is a temporary speculative(contextual) implementation. The final solution will have a explicit field in
        // the workload specifying that the weights are not present

        if ((in.layout != out.layout)  // layout change
            || (!is_same_datatype_footprint(in.datatype,
                                            out.datatype))  // from a type size to another, not only F16 to [u]int8
        ) {
            return true;
        } else {
            return false;
        }
    }
};

inline std::ostream& operator<<(std::ostream& stream, const TensorInfo& d) {
    stream << "TensorInfo: \n"                                                                                        //
           << " shape: \t{" << d.width << "," << d.height << ","                                                      //
           << d.channels << "," << d.batch << "} ;\n"                                                                 //
           << " dtype: \t" << (int)d.datatype << " : " << DataType_ToText.at(static_cast<int>(d.datatype)) << " ;\n"  //
           << " layout: \t" << (int)d.layout << " : " << Layout_ToText.at(static_cast<int>(d.layout)) << " ;\n"       //
           << " sparsity enabled: \t" << (d.sparsity_enabled ? "true" : "false") << " ;\n"                            //
           << " sparsity value: \t" << d.sparsity << " ;\n"                                                           //
           << " swizzling: \t{" << (int)d.swizzling << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.swizzling)) << "} ;\n"  //
            ;
    return stream;
}

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
            ;
    return stream;
}
}  // namespace VPUNN

#endif  //
