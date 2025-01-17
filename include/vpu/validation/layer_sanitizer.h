// Copyright © 2024 Intel Corporation
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
#include "device_valid_valuesVPU2.h"
#include "device_valid_valuesVPU2_7.h"
#include "device_valid_valuesVPU4.h"
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
                                                       VPU2_0_LayerOnTileValidValues, VPU2_7_LayerOnTileValidValues,
                                    VPU4_0_LayerOnTileValidValues>;
    using DPU_SplitLayersValidator = DPU_ConfigurableOperationValidator<SplitLayersContext>;
    DPU_SplitLayersValidator splitLayer_validator;

public:
    const IDeviceValidValues& getDeviceConfiguratorForTiles(VPUDevice device) const {
        return splitLayer_validator.get_config(device);  // for now
    }

    /// @brief checks the layer validity against the rules of an unsplit Layer
    void check_completeLayer_consistency(const DPULayer& layer, SanityReport& result, ISIStrategy presumed_strategy,
                                         unsigned int nTiles, VPUTilingStrategy strategy=VPUTilingStrategy::NONE) const {
        result.resetOK();  // all OK
        if (!layer_validator.is_supported(layer.device)) {
            result.mark_unknown_device();
            return;
        }

        const auto& config = layer_validator.get_config(layer.device);  // previous if prevents throwing

        try {
            DPUOperation w(layer, config);  // workload internal representation
            w.set_intended_split(presumed_strategy,
                                 nTiles);  // here we remain with SOH even if we do SOHO because it is still a split ,
                                           // not clustering this has to be carefully redesigned when we get rid of ISI.

            // no memory size check?

            auto& operation_behaviour = config.get_specific_behaviour(layer.op);  // will throw if unknown op
            layer_validator.check_layer_consistency(w, nTiles, strategy, config, operation_behaviour, result);

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
            const DPUOperation w(layer, config);  // workload internal representation (throws if bad op)

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
                auto& operation_behaviour = config.get_specific_behaviour(layer.op);
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
