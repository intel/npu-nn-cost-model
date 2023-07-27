// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_LAYER_SANITIZER_H
#define VPUNN_VPU_LAYER_SANITIZER_H

#include <numeric>

#include "sanity_report.h"
#include "vpu/types.h"

#include "behaviors_and_devices_containers.h"
#include "device_valid_values.h"
#include "dpu_operations_valid_behaviours.h"
#include "dpu_operations_validator.h"
#include "vpu_layer_validator.h"

#include "vpu/layer.h"

namespace VPUNN {

/// @brief Layer validation mechanisms for split and un-split layers
class LayersValidation {
protected:
    VPU_LayerValidator layer_validator;

    /// configuration bundle for Workloads that are no finally split. Are just Layers on a Tile
    using SplitLayersContext = Behavior_Device_Mapping<OperationsBehaviour,  // operations for workloads
                                                       VPU2_0_LayerOnTileValidValues, VPU2_7_LayerOnTileValidValues>;
    using DPU_SplitLayersValidator = DPU_ConfigurableOperationValidator<SplitLayersContext>;
    DPU_SplitLayersValidator splitLayer_validator;

public:
    /// @brief checks the layer validity against the rules of an unsplit Layer
    void check_completeLayer_consistency(const DPULayer& layer, SanityReport& result, ISIStrategy strategy,
                                         unsigned int nTiles) const {
        result.resetOK();  // all OK
        if (!layer_validator.is_supported(layer.device)) {
            result.mark_unknown_device();
            return;
        }

        const auto& config = layer_validator.get_config(layer.device);  // previous if prevents throwing

        try {
            auto& operation_behaviour = config.get_specific_behaviour(layer.op);  // will throw if unknown op
            DPUOperation w(layer);                                                // workload internal representation

            w.set_intended_split(strategy, nTiles);

            operation_behaviour.deduce_input_1(w.input_0, w.output_0, config, w.kernel, w.input_1);

            // no memory size check?

            layer_validator.check_layer_consistency(w, config, operation_behaviour, result);

        } catch (const std::runtime_error&) {
            result.mark_unknown_operation();  // most probably
        }
        // result will keep its content to outside
    }

    void check_splitLayer_consistency(const DPULayer& layer, SanityReport& result) const {
        result.resetOK();  // all OK
        if (!splitLayer_validator.is_supported(layer.device)) {
            result.mark_unknown_device();
            return;
        }
        const auto& config = splitLayer_validator.get_config(layer.device);  // previous if prevents throwing

        try {
            auto& operation_behaviour = config.get_specific_behaviour(layer.op);  // will throw if unknown op
            DPUOperation w(layer);                                                // workload internal representation
            operation_behaviour.deduce_input_1(w.input_0, w.output_0, config, w.kernel, w.input_1);

            const WorkloadMemorySizeCalculator memory_calculator;  // ignore cmx overhead
            const auto cmx_memory = memory_calculator.compute_memory(w, config);

            const int avaialable_cmx_memo{config.get_cmx_size(layer.device)};
            const auto necesarry_cmx_memo = cmx_memory.cmx;

            if (avaialable_cmx_memo < necesarry_cmx_memo) {
                result.mark_size_too_big();
                std::stringstream buffer;
                buffer << "Memory request bigger than available: \n Requested: " << cmx_memory
                       << "\n available : " << avaialable_cmx_memo << "\n";
                result.info += buffer.str();
            } else {
                splitLayer_validator.check_workload_consistency(w, config, operation_behaviour, result);
            }

        } catch (const std::runtime_error&) {
            result.mark_unknown_operation();
        }
    }

    void sanitize_preconditions(DPULayer& layer) const {
        if (!layer_validator.is_supported(layer.device)) {
            return;
        }

        const auto& config = layer_validator.get_config(layer.device);

        // change swizzlings in case we are in a device that has less/no swizzlings
        {
            DPUWorkload& wl{layer};
            wl.input_swizzling[0] = config.adapt_device_comaptible_swizzling(wl.input_swizzling[0]);
            wl.input_swizzling[1] = config.adapt_device_comaptible_swizzling(wl.input_swizzling[1]);

            wl.output_swizzling[0] = config.adapt_device_comaptible_swizzling(wl.output_swizzling[0]);
        }
    }
};

}  // namespace VPUNN

#endif  //
