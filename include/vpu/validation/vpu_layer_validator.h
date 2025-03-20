// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_LAYER_VALIDATOR_H
#define VPUNN_VPU_LAYER_VALIDATOR_H

#include <algorithm>

// #include <sstream>  // for error formating
#include <stdexcept>

#include <iostream>

#include "vpu/layer.h"
#include "vpu/types.h"

#include <tuple>
#include "behaviors_and_devices_containers.h"
#include "device_valid_valuesVPU2.h"
#include "device_valid_valuesVPU2_7.h"
#include "device_valid_valuesVPU4.h"
#include "interface_operations_behavior.h"
#include "interface_valid_values.h"
#include "layer_operations_valid_behaviours.h"
#include "sanity_report.h"

namespace VPUNN {

/// layer level operations dynamic behavior
using LayerOperationsBehaviour =
        Behaviours<CONVOLUTION_Constraints_Layer, DW_CONVOLUTION_Constraints_Layer, CM_CONVOLUTION_Constraints_Layer,
                   ELTWISE_Constraints_Layer, MAXPOOL_Constraints_Layer, LAYERNORM_Constraints_Layer, ELTWISE_MUL_Constraints_Layer>;

/// @brief services for Layer validation
class VPU_LayerValidator :
        public Behavior_Device_Mapping<LayerOperationsBehaviour,  // operations
                                       VPU2_0_LayerValidValues, VPU2_7_LayerValidValues, VPU4_0_LayerValidValues> {
protected:
public:
    /// nTiles: number of tiles we split a layer
    /// strategy: split layer strategy
    void check_layer_consistency(const DPUOperation& w, const unsigned int nTiles, VPUTilingStrategy strategy,
                                 const IDeviceValidValues& config,
                                 const IOperationDynamicConstraints& operation_behaviour, SanityReport& result) const {
        result.resetOK();  // all OK

        /// @brief: compute an array of coefficients by which we can multiply the upper limit of the allowed range (this
        /// is equal with 8192) for the dimensions (width, height, channels) of a layer, this help us to maximize WHC at
        /// high level, in that way by splitting a layer, at middle layer, we can have layers with maximum allowed
        /// values
        /// (=8192) for WHC
        ///
        /// @param strategy: tiling strategy
        /// @param nTiles: number of tiles
        /// @return: an array that contains the coefficients' values, the order of the coefficients in array is WHCB
        auto border_coeff = [](VPUTilingStrategy strategy, const unsigned int nTiles) -> WHCBTensorShape {
            switch (strategy) {
            case VPUTilingStrategy::NONE:
                return {1, 1, 1, 1};  // order WHCB
            case VPUTilingStrategy::SOH_Overlapped:
                return {1, nTiles, 1, 1};
            case VPUTilingStrategy::SOK:
                return {1, 1, nTiles, 1};
            case VPUTilingStrategy::SOW:
                return {nTiles, 1, 1, 1};
            case VPUTilingStrategy::SOHW:
                return {nTiles, nTiles, 1, 1};
            case VPUTilingStrategy::SOHK:
                return {1, nTiles, nTiles, 1};
            case VPUTilingStrategy::SOH_HaloRead:
                return {1, nTiles, 1, 1};
            case VPUTilingStrategy::SOHO_K_SWITCH:
                return {1, nTiles, 1, 1};
            case VPUTilingStrategy::SOH_K_SWITCH:
                return {1, nTiles, 1, 1};
            case VPUTilingStrategy::SOK_NO_BROADCAST:
                return {1, 1, nTiles, 1};
            default:
                return {1, 1, 1, 1};
            }
        };

        const WHCBTensorShape extended_border_coeff{border_coeff(strategy, nTiles)};

        Checker checker;
        try {
            checker.check_is_in_list(w.device, config.get_devices(), "Device");
            checker.check_is_in_list(w.operation, config.get_valid_operations(), "Operation");

            // todo: this is just the desired one to be applied at tile splitting?
            checker.check_is_in_list(w.output_write_tiles, config.get_output_write_tile_options(),
                                     "output_write_tiles");
            // dep on out tile and op  todo: ISI is just the desired one to be applied at tile splitting
            checker.check_is_in_list(w.isi_strategy, config.get_ISI_Strategy_Range(w),
                                     "ISI_strategy");  // no ELMWISE and SOK

            {  // kernel aspects
                const auto kernel_options{config.get_kernel_range(w)};
                checker.check_is_in_list(w.kernel.width, kernel_options, "kernel.width");
                checker.check_is_in_list(w.kernel.height, kernel_options, "kernel.height");

                auto k = w.kernel;  // check if equal (SOH + DW_CONV)
                if (operation_behaviour.normalize_kernel_dimension(w.isi_strategy, k)) {
                    checker.add_check_failed("Kernel dimension are not normalized properly!(maybe not equal?)");
                }
            }
            {  // padding aspects
                const auto kernel_pad_horz_options{config.get_pad_horz_range(w)};
                const auto kernel_pad_vert_options{config.get_pad_vert_range(w)};
                const auto k{w.kernel};

                checker.check_is_in_list(k.pad_left, kernel_pad_horz_options, "kernel.pad_left");
                checker.check_is_in_list(k.pad_top, kernel_pad_vert_options, "kernel.pad_top");
                checker.check_is_in_list(k.pad_right, kernel_pad_horz_options, "kernel.pad_right");
                checker.check_is_in_list(k.pad_bottom, kernel_pad_vert_options, "kernel.pad_bottom");
            }
            {  // input/activation dimensions and tensor properties
                const auto& in0{w.input_0};
                // what to do with batch??

                std::pair<int, int> height_interval{config.get_input_height_interval(w)};
                height_interval.second = height_interval.second * extended_border_coeff.height();
                checker.check_is_in_interval((int)in0.height, height_interval, "input_0.height");

                std::pair<int, int> width_interval{config.get_input_width_interval(w)};
                width_interval.second = width_interval.second * extended_border_coeff.width();
                checker.check_is_in_interval((int)in0.width, width_interval, "input_0.width");

                const auto range = config.get_input_channels_restriction(w);
                const auto input_range{range.multiply_upper(extended_border_coeff.channels())};
                checker.check_is_in_requirements((int)in0.channels, input_range, "input_0.channels");

                checker.check_is_in_list(in0.datatype, config.get_input_valid_datatypes(w), "input_0.datatype");
                checker.check_is_in_list(in0.layout, config.get_valid_layouts(), "input_0.layout");
                checker.check_is_in_list(in0.swizzling, config.get_valid_swizzlings(), "input_0.swizzling");
            }
            {  // stride , depends on input zero, and operation sometimes
                const auto k{w.kernel};
                const auto stride_options{config.get_strides_range(w)};

                checker.check_is_in_list(k.stride_width, stride_options.first, "kernel.stride_width");
                checker.check_is_in_list(k.stride_height, stride_options.second, "kernel.stride_height");
            }
            {  // output dims, non random,  depend on input, padding, kernel, stride
               // batch in out to be equal
                if (w.output_0.batch != w.input_0.batch) {
                    checker.add_check_failed("Output.batch different than input_0.batch!");
                }

                const auto expected_out_width =
                        config.compute_output_dim((int)w.input_0.width, w.kernel.pad_left, w.kernel.pad_right,
                                                  w.kernel.width, w.kernel.stride_width);
                const auto expected_out_height =
                        config.compute_output_dim((int)w.input_0.height, w.kernel.pad_top, w.kernel.pad_bottom,
                                                  w.kernel.height, w.kernel.stride_height);

                checker.check_is_in_interval((int)w.output_0.width,
                                             std::make_pair(expected_out_width, expected_out_width), "output_0.width");

                checker.check_is_in_interval((int)w.output_0.height,
                                             std::make_pair(expected_out_height, expected_out_height),
                                             "output_0.height");

                // TODO: consider also the Split  H aspect
                auto range = config.get_output_channels_restriction(w);
                const auto output_range{range.multiply_upper(extended_border_coeff.channels())};
                checker.check_is_in_requirements((int)w.output_0.channels, output_range, "output_0.channels");

                {  // special SOH situation ;  to do: should be captured in the generic approach
                    if (w.isi_strategy ==
                        ISIStrategy::SPLIT_OVER_H) {  // SOH or SOHO, be careful to set this correctly for layer
                        if (w.output_0.height <= 1) {
                            // cannot do split
                            checker.add_check_failed("can't do SPLIT_OVER_H if output_0_.height is <= 1");
                        } else {
                            // check special corner case  for se_sp_size_se_size se_sp_size1_se_size
                            //   @todo ?
                        }
                    }
                }

                {  // sparsity and types  for output_0
                    const auto& out0{w.output_0};
                    checker.check_is_in_list(out0.datatype, config.get_output_valid_datatypes(w), "output_0.datatype");
                    checker.check_is_in_list(out0.layout, config.get_valid_layouts(), "output_0.layout");
                    checker.check_is_in_list(out0.swizzling, config.get_valid_swizzlings(), "output_0.swizzling");
                }
            }
            {
                // types for input_1
                const auto& in1{w.input_1};
                checker.check_is_in_list(in1.datatype, config.get_weights_valid_datatypes(w), "input_1.datatype");
            }
            {  // sparsity check on all channels

                if ((w.input_0.sparsity < 0.0F) || (w.input_0.sparsity > 1.0F)) {
                    checker.add_check_failed("input_0.sparsity not in interval [0.0, 1.0] !");
                }

                if ((w.input_1.sparsity < 0.0F) || (w.input_1.sparsity > 1.0F)) {
                    checker.add_check_failed("output_1.sparsity not in interval [0.0, 1.0] !");
                }
                {
                    std::string info_out{};
                    if (!operation_behaviour.check_sparsity_rules(config, w, info_out))
                        checker.add_check_failed(info_out);
                }
            }
            { checker.check_is_in_list(w.execution_order, config.get_valid_execution_order(), "Execution_Order"); }
            // no padding optimization checked

            {  // check correlation between in-out tensors
                std::string info_out{};
                if (!operation_behaviour.check_input_output_tensor_corelation(config, w, info_out))
                    checker.add_check_failed(info_out);
            }

        } catch (const std::exception& e) {
            checker.add_check_failed(e.what());
        }
        // draw a final conclusion based on what was accumulated into the checker
        if (!checker.is_clean()) {
            result.mark_invalid_LayerConfiguration();
            result.info = checker.findings();
        }
    }

protected:
};

}  // namespace VPUNN

#endif  //
