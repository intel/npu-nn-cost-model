// Copyright © 2024 Intel Corporation
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
#include <unordered_map>
#include "nn_models.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN_unit_tests {

/// Value
inline VPUNN::CyclesInterfaceType V(const VPUNN::CyclesInterfaceType v) {
    return v;
}
const VPUNN::Swizzling swz_def{VPUNN::Swizzling::KEY_5};  // default enabled for 2.7 onwards

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

class CompareValues {
protected:
    /// example:  if value is 100 and tol_percent is 10, the function returns 10 (which is 10% of 100)
    /// all values provided as parameters to this function must be positive otherwise undefined behavior
    /// @param value is the value for which we determine the desired percentage part
    /// @param tol_percent is a percentage , if 13 it means 13%
    /// @return that specific percentage (tol_percent) of the total value
    template <typename T>
    static float compute_maximum_difference(T value, float tol_percent) {
        // parameters must be positive otherwise undefined behavior
        assert(value >= 0 && tol_percent >= 0);

        return (tol_percent * value) / 100;
    }

public:
    /// this function compares v1 and v2 and see if they can be considered equal or not
    ///   For example, this function was designed for the following situation:
    /// -v1 is a standard value,
    /// -v2 is a value obtained from some functions,
    /// -min_necessary_abs_diff is the maximum difference for which these values are still considered equal
    /// -tol_percent represents how much percent the difference can be from the smallest value, bassically tell me by
    /// how much percent a value can be greater or smaller than the other, We use this tol_percent only if the absolute
    ///  difference between v1 and v2 is greater than diff.
    /// all values provided as parameters to this function must be positive otherwise undefined behavior
    /// @param v1 a value we want to compare
    /// @param v2 the other value we want to compare with
    /// @param min_necessary_abs_diff if the absolute difference between v1 and v2 is smaller than
    /// min_necessary_abs_diff, tol_percent no longer matters, we consider v1=v2
    /// @param tol_percent represents how much percent the difference can be from the smallest value, tol_percent
    /// represents a tolerance threshold expressed as a gross value, not as a percentage, eg: if tol_percent=5 it
    /// means 5%
    /// @return true if values are considered to be equal, false if not
    template <typename T>
    static bool isEqual(T v1, T v2, T min_necessary_abs_diff, float tol_percent) {
        // parameters must be positive otherwise undefined behavior
        assert(v1 >= 0 && v2 >= 0 && min_necessary_abs_diff >= 0 && tol_percent >= 0);

        const auto difference = (v1 > v2) ? (v1 - v2) : (v2 - v1);

        if (difference < min_necessary_abs_diff)
            return true;

        // worst case scenario: the maximum difference should be calculated based on the smallest value we want to
        // compare
        const float max_diff =
                std::min(compute_maximum_difference(v1, tol_percent), compute_maximum_difference(v2, tol_percent));

        return (difference <= max_diff);
    }

private:
};

/**
 * @brief Template class for managing VPU model instances by device type
 *
 * Provides a device-to-model mapping with automatic fallback to a default model
 * for unregistered devices. Uses unique_ptr for automatic memory management.
 *
 * @tparam ModelType The type of model being managed (VPUCostModel, VPULayerCostModel, etc.)
 */
template <typename ModelType>
class ModelMap {
public:
    /**
     * @brief Constructor that creates a fallback model for unregistered devices
     * @param args Constructor arguments for the fallback ModelType instance
     */
    template<typename... Args>
    ModelMap(Args&&... args) {
        models_map.emplace(VPUNN::VPUDevice::__size, std::make_unique<ModelType>(std::forward<Args>(args)...));
    }

    /**
     * @brief Register a model for a specific VPU device
     * @param device The VPU device identifier (e.g., VPU_2_7, VPU_4_0)
     * @param args Constructor arguments for the ModelType instance
     */
    template<typename... Args>
    void addModel(const VPUNN::VPUDevice device, Args&&... args) {
        models_map.emplace(device, std::make_unique<ModelType>(std::forward<Args>(args)...));
    }

    /**
     * @brief Get a model instance for the specified device
     * @param device The VPU device to get the model for
     * @return Reference to the model (guaranteed non-null)
     *
     * Returns the registered model if available, otherwise returns the fallback model
     * created during construction.
     */
    ModelType& getModel(const VPUNN::VPUDevice device) {
        auto model_it = models_map.find(device);
        if (model_it != models_map.end()) {
            return *(model_it->second);
        }
        return *models_map[VPUNN::VPUDevice::__size];
    }
    
    void clear() {
        models_map.clear();
    }

    ~ModelMap() {
        models_map.clear();
    }

private: 
    /// Map storing device-specific model instances
    std::unordered_map<VPUNN::VPUDevice, std::unique_ptr<ModelType>> models_map;
};

}  // namespace VPUNN_unit_tests

#endif  // !VPUNN_UT_COMMON_HELPERS_H
