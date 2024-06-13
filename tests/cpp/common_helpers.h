// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_UT_COMMON_HELPERS_H
#define VPUNN_UT_COMMON_HELPERS_H

#include <string>
#include <vector>
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

#ifndef VPU_2_7_MODEL_PATH
#define VPU_2_7_MODEL_PATH "../../../models/vpu_2_7.vpunn"
#endif

#ifndef VPU_2_0_MODEL_PATH
#define VPU_2_0_MODEL_PATH "../../../models/vpu_2_0.vpunn"
#endif

#ifndef VPU_4_0_MODEL_PATH
#define VPU_4_0_MODEL_PATH VPU_2_7_MODEL_PATH
#endif

namespace VPUNN_unit_tests {

/// @brief class to help extracting paths and names of neural network model
class NameHelperNN {
public:
    /// @brief gets the folder where the models are
    static std::string get_model_root() {
        const std::string m{VPU_2_0_MODEL_PATH};
        return get_model_root(m);
    }

    /// @brief gets the folder where this model is
    static std::string get_model_root(const std::string& model_file) {
        const std::string m{model_file};
        std::string model_root = m.substr(0, m.find_last_of('/') + 1);
        return model_root;
    }

    /// @brief appends .fast before .vpunn file suffix
    static std::string make_fast_version(const std::string& model_file) {
        const std::string m{model_file};
        std::string model_base = m.substr(0, m.rfind(".vpunn"));

        std::string fast_name = model_base + ".fast" + ".vpunn";
        return fast_name;
    };
};

/// @brief Contains the lists of available models.
/// Is aware of fast or normal files, and knows the associated devices for each NN
class VPUNNModelsFiles {
public:
    using ModelDescriptor = std::pair<std::string, VPUNN::VPUDevice>;
    const std::vector<ModelDescriptor> standard_model_paths{{VPU_2_0_MODEL_PATH, VPUNN::VPUDevice::VPU_2_0},
                                                            {VPU_2_7_MODEL_PATH, VPUNN::VPUDevice::VPU_2_7}};
    const std::vector<ModelDescriptor> fast_model_paths{
            {NameHelperNN::make_fast_version(VPU_2_0_MODEL_PATH), VPUNN::VPUDevice::VPU_2_0},
            {NameHelperNN::make_fast_version(VPU_2_7_MODEL_PATH), VPUNN::VPUDevice::VPU_2_7}};

    const std::vector<ModelDescriptor> all_model_paths{concat(standard_model_paths, fast_model_paths)};

    static const VPUNNModelsFiles& getModels() {
        static const VPUNNModelsFiles the_NN_models;
        return the_NN_models;
    }

private:
    std::vector<ModelDescriptor> concat(const std::vector<ModelDescriptor>& v1,
                                        const std::vector<ModelDescriptor>& v2) const {
        std::vector<ModelDescriptor> v(v1);
        v.insert(v.end(), v2.begin(), v2.end());
        return v;
    }
};

/// Value
inline VPUNN::CyclesInterfaceType V(const VPUNN::CyclesInterfaceType v) {
    return v;
}

namespace wlh {  // keep this isolated. Better would have been a static consts in a class, but implies a cpp file.
const std::string eq{"="};
const std::string end{",\n"};
}  // namespace wlh

class WLHelp {
public:
    static std::string toDictString(const VPUNN::VPUTensor& d, std::string prefix = "input_x") {
        std::stringstream buffer;
        buffer << ""                                                                                            //
               << prefix + "_batch" << wlh::eq << d.batches() << wlh::end                                       //
               << prefix + "_channels" << wlh::eq << d.channels() << wlh::end                                   //
               << prefix + "_height" << wlh::eq << d.height() << wlh::end                                       //
               << prefix + "_width" << wlh::eq << d.width() << wlh::end                                         //
                                                                                                                //
               << prefix + "_sparsity_enabled" << wlh::eq << (d.get_sparsity() ? "True" : "False") << wlh::end  //

               << prefix + "_datatype" << wlh::eq << "DataType."
               << VPUNN::DataType_ToText.at(static_cast<int>(d.get_dtype())) << wlh::end  //
               << prefix + "_layout" << wlh::eq << "Layout."
               << VPUNN::Layout_ToText.at(static_cast<int>(d.get_layout())) << wlh::end;
        return buffer.str();
    }

    static std::string toDictString(const VPUNN::DPUWorkload& d) {
        std::stringstream buffer;
        buffer << "\n\n"  //
               << "device" << wlh::eq << "VPUDevice." << VPUNN::VPUDevice_ToText.at(static_cast<int>(d.device))
               << wlh::end  //
               << "operation" << wlh::eq << "Operation." << VPUNN::Operation_ToText.at(static_cast<int>(d.op))
               << wlh::end  //

               // inputs and oytputs tensors

               << toDictString(d.inputs[0], "input_0")  //
               << "input_0_swizzling" << wlh::eq << "Swizzling."
               << VPUNN::Swizzling_ToText.at(static_cast<int>(d.input_swizzling[0])) << wlh::end  //

               << "weight_sparsity_enabled" << wlh::eq << (d.weight_sparsity_enabled ? "True" : "False") << wlh::end  //

               << "input_sparsity_rate" << wlh::eq << d.act_sparsity << wlh::end      //
               << "weight_sparsity_rate" << wlh::eq << d.weight_sparsity << wlh::end  //

               << "input_1_swizzling" << wlh::eq << "Swizzling."
               << VPUNN::Swizzling_ToText.at(static_cast<int>(d.input_swizzling[1])) << wlh::end  //

               << "execution_order" << wlh::eq << "ExecutionMode."
               << VPUNN::ExecutionMode_ToText.at(static_cast<int>(d.execution_order)) << wlh::end  //
               << "activation_function" << wlh::eq << "ActivationFunction."
               << VPUNN::ActivationFunction_ToText.at(static_cast<int>(d.activation_function)) << wlh::end  //

               << "kernel_height" << wlh::eq << d.kernels[VPUNN::Dim::Grid::H] << wlh::end  //
               << "kernel_width" << wlh::eq << d.kernels[VPUNN::Dim::Grid::W] << wlh::end   //

               << "kernel_pad_bottom" << wlh::eq << d.padding[VPUNN::Dim::BOTTOM] << wlh::end  //
               << "kernel_pad_left" << wlh::eq << d.padding[VPUNN::Dim::LEFT] << wlh::end      //
               << "kernel_pad_right" << wlh::eq << d.padding[VPUNN::Dim::RIGHT] << wlh::end    //
               << "kernel_pad_top" << wlh::eq << d.padding[VPUNN::Dim::TOP] << wlh::end        //

               << "kernel_stride_height" << wlh::eq << d.strides[VPUNN::Dim::Grid::H] << wlh::end  //
               << "kernel_stride_width" << wlh::eq << d.strides[VPUNN::Dim::Grid::W] << wlh::end   //

               << toDictString(d.outputs[0], "output_0")  //

               << "output_0_swizzling" << wlh::eq << "Swizzling."
               << VPUNN::Swizzling_ToText.at(static_cast<int>(d.output_swizzling[0])) << wlh::end  //

               << "isi_strategy" << wlh::eq << "ISIStrategy."
               << VPUNN::ISIStrategy_ToText.at(static_cast<int>(d.isi_strategy)) << wlh::end  //
               << "output_write_tiles" << wlh::eq << d.output_write_tiles << wlh::end         //
               << std::endl
               << "--------------\n";

        return buffer.str();
    }
};

}  // namespace VPUNN_unit_tests

#endif  // !VPUNN_UT_COMMON_HELPERS_H
