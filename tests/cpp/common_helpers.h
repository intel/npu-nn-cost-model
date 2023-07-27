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
    static std::string get_model_root(std::string model_file) {
        const std::string m{model_file};
        std::string model_root = m.substr(0, m.find_last_of('/') + 1);
        return model_root;
    }

    /// @brief appends .fast before .vpunn file suffix
    static std::string make_fast_version(std::string model_file) {
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

}  // namespace VPUNN_unit_tests

#endif  // !VPUNN_UT_COMMON_HELPERS_H
