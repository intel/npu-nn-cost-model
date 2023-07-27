// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_VALIDATOR_DATA_DPU_OPERATION_H
#define VPUNN_VPU_VALIDATOR_DATA_DPU_OPERATION_H

#include "vpu/types.h"

namespace VPUNN {

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
struct DPUOperation {
    VPUDevice device{};  ///< device family, VPU2_0, 2_7, ...
    Operation operation{};

    TensorInfo input_0;  ///< activators
    TensorInfo input_1;  ///< weights

    TensorInfo output_0;

    ExecutionMode execution_order{};  ///< execution mode

    KernelInfo kernel;

    int output_write_tiles{1};  //< broadcast policy
    ISIStrategy isi_strategy{ISIStrategy::CLUSTERING};

    void set_intended_split(ISIStrategy strategy, unsigned int nTiles) {
        isi_strategy = strategy;
        output_write_tiles = static_cast<int>(nTiles);
    }

    /// constructor from a DPUWorkload.
    /// input_1 (weights) tensor is not filled with shape
    explicit DPUOperation(const DPUWorkload& w)
            : device{w.device},
              operation{w.op},
              input_0{w.inputs[0]},
              output_0{w.outputs[0]},
              execution_order{w.execution_order},
              kernel{w},
              output_write_tiles{static_cast<int>(w.output_write_tiles)},
              isi_strategy{w.isi_strategy} {
        // from WL to tensors
        input_0.swizzling = w.input_swizzling[0];
        input_0.sparsity = w.act_sparsity;

        {  // input 1 is left empty . the WL does not have info yet
            // some extra aspects can be recovered based on operation and  in/out/Kernel size, but is not the scope
            // here

            input_1.swizzling = w.input_swizzling[1];
            input_1.datatype = input_0.datatype;
            input_1.sparsity_enabled = w.weight_sparsity_enabled;
            input_1.sparsity = w.weight_sparsity;
        }
        output_0.swizzling = w.output_swizzling[0];
    }
    DPUOperation() = default;

    DPUWorkload clone_as_DPUWorkload() const {
        DPUWorkload wl;  // looks like local  object , but  hope  for Return Value Optimization (RVO)

        wl.device = device;
        wl.op = operation;

        // data type restriction is mandatory, but should be also for all DPU workloads not just generated ones
        {
            auto& in = input_0;
            wl.inputs[0] = VPUTensor({static_cast<unsigned int>(in.width), static_cast<unsigned int>(in.height),
                                      static_cast<unsigned int>(in.channels), static_cast<unsigned int>(in.batch)},
                                     in.datatype, in.layout, in.sparsity_enabled);
        }
        {
            auto& out = output_0;
            wl.outputs[0] = VPUTensor({static_cast<unsigned int>(out.width), static_cast<unsigned int>(out.height),
                                       static_cast<unsigned int>(out.channels), static_cast<unsigned int>(out.batch)},
                                      out.datatype, out.layout, out.sparsity_enabled);
        }

        wl.kernels[Dim::Grid::W] = kernel.width;
        wl.kernels[Dim::Grid::H] = kernel.height;

        wl.strides[Dim::Grid::W] = kernel.stride_width;
        wl.strides[Dim::Grid::H] = kernel.stride_height;

        wl.padding[Dim::Padding::TOP] = kernel.pad_top;
        wl.padding[Dim::Padding::BOTTOM] = kernel.pad_bottom;
        wl.padding[Dim::Padding::LEFT] = kernel.pad_left;
        wl.padding[Dim::Padding::RIGHT] = kernel.pad_right;

        wl.execution_order = execution_order;

        // wl.activation_function = dpu.  // WHERE IS IT?
        wl.activation_function = ActivationFunction::NONE;  //

        wl.act_sparsity = input_0.sparsity;
        wl.weight_sparsity = input_1.sparsity;

        wl.input_swizzling[0] = input_0.swizzling;
        wl.input_swizzling[1] = input_1.swizzling;

        wl.output_swizzling[0] = output_0.swizzling;

        // wl.offsets;  // NOT SET remains zero/init
        wl.output_write_tiles = output_write_tiles;
        wl.isi_strategy = isi_strategy;

        wl.weight_sparsity_enabled = input_1.sparsity_enabled;

        return wl;
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

            ;
    return stream;
}

}  // namespace VPUNN

#endif  //
