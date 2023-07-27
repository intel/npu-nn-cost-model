// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_RANDOM_TASK_GENERATOR_H
#define VPUNN_RANDOM_TASK_GENERATOR_H

#include <algorithm>

#include <sstream>  // for error formating
#include <stdexcept>

#include <iostream>
#include <string>

#include "vpu/types.h"

#include "sample_generator.h"
#include "vpu/validation/device_valid_values.h"
#include "vpu/validation/dpu_operations_valid_behaviours.h"

#include "vpu/validation/dpu_operations_validator.h"

namespace VPUNN {

class DPU_OperationGenerator : public DPU_OperationValidator {
public:
    /// @brief provides the behavior associated with the desired operation
    ///
    /// @param op the desired operation, will throw if not supported
    /// @throws runtime_error if the operation is not supported or known
    /// @returns IOperationsConstrainsts to be used for this operation
    const IOperationDynamicGenerator& get_nodevice_operation_specific_generator(const Operation op) const {
        return specific_behaviours.get_operation_specific_<IOperationDynamicGenerator>(op);
    }

    /// constructor with special configuration for generation
    DPU_OperationGenerator() {
        memory_calculator.set_ignore_cmx_overhead(false);
    }
};

/// @brief Generates a random Workload  based on device rules
class DPU_OperationCreator {
private:
    DPU_OperationGenerator my_validator{};

    /// @brief  Generated a valid (no checks on size) DPUOperation, the device is kept unchanged and obtained from dpu.
    /// Does not check for memory size or if it fits into the memory.
    ///
    /// @param sampler the object used to chose a random sample from an interval or a discrete list
    /// @param config the static configuration of the option for this generator
    /// @dpu [in , out] provides the device and will hold the generated information
    /// @throws runtime_error if the operation selected from config is not supported or known
    ///
    /// @returns true in case dpu contains a valid DPUOPeration, false in case there were problems and it was not
    /// possible to generate
    bool sample_tentative(Sampler& sampler, const IDeviceValidValues& config, DPUOperation& dpu) const {
        // 1. operation
        dpu.operation = sampler.sample_list(config.get_valid_operations_range());

        // 2. select Output write tiles
        {  // check again if depends on op?: yes depends, EMLwise cannot be !=1
            dpu.output_write_tiles = sampler.sample_list(config.get_output_write_tile_Range(dpu));
        }
        // 3. ISI strategy
        { dpu.isi_strategy = sampler.sample_list(config.get_ISI_Strategy_Range(dpu)); }

        {  // kernel , input, padding,
            {
                const auto kernel_options{config.get_kernel_range(dpu)};
                dpu.kernel.width = sampler.sample_list(kernel_options);
                dpu.kernel.height = sampler.sample_list(kernel_options);

                auto& operation_behaviour = config.get_specific_behaviour(dpu.operation);  // might throw
                operation_behaviour.normalize_kernel_dimension(dpu.isi_strategy, dpu.kernel);
            }
            {  // padding
                const auto kernel_pad_horz_options{config.get_pad_horz_range(dpu)};
                const auto kernel_pad_vert_options{config.get_pad_vert_range(dpu)};

                dpu.kernel.pad_left = sampler.sample_list(kernel_pad_horz_options);
                dpu.kernel.pad_right = dpu.kernel.pad_left;  // artificial constraint

                dpu.kernel.pad_top = sampler.sample_list(kernel_pad_vert_options);
                dpu.kernel.pad_bottom = dpu.kernel.pad_top;  // artificial constraint
            }

            {  // input activation dimensions
                dpu.input_0.batch = 1;

                dpu.input_0.height = sampler.sample_list_decrease_prob(config.get_input_height_range(dpu));
                dpu.input_0.width = sampler.sample_list_decrease_prob(config.get_input_width_range(dpu));

                // MAKE DYNAMIC CHANNELS
                dpu.input_0.channels = sampler.sample_list_decrease_prob(config.get_input_channels_range(dpu));
            }

            {  // stride , depends on input zero, and operation sometimes
                const auto stride_options{config.get_strides_range(dpu)};

                dpu.kernel.stride_width = sampler.sample_list_decrease_prob(stride_options.first);    // Width
                dpu.kernel.stride_height = sampler.sample_list_decrease_prob(stride_options.second);  // Height

                // normalize to be equal for generated samples
                const auto stride = std::min(dpu.kernel.stride_width, dpu.kernel.stride_height);
                dpu.kernel.stride_width = stride;
                dpu.kernel.stride_height = stride;
            }
            {  // output dims, non random,  depend on input, padding, kernel, stride
                dpu.output_0.batch = dpu.input_0.batch;
                dpu.output_0.width =
                        config.compute_output_dim((int)dpu.input_0.width, dpu.kernel.pad_left, dpu.kernel.pad_right,
                                                  dpu.kernel.width, dpu.kernel.stride_width);

                dpu.output_0.height =
                        config.compute_output_dim((int)dpu.input_0.height, dpu.kernel.pad_top, dpu.kernel.pad_bottom,
                                                  dpu.kernel.height, dpu.kernel.stride_height);
            }

            {  // sanitize  padding (when kernel_stride_>1, different  padding can lead to the same output
               // dim, choosing the smallest padding)

                dpu.kernel.pad_left =
                        config.check_trailing_padding((int)dpu.input_0.width, (int)dpu.output_0.width,
                                                      dpu.kernel.pad_right, dpu.kernel.width, dpu.kernel.stride_width);

                dpu.kernel.pad_top = config.check_trailing_padding((int)dpu.input_0.height, (int)dpu.output_0.height,
                                                                   dpu.kernel.pad_bottom, dpu.kernel.height,
                                                                   dpu.kernel.stride_height);
            }

            dpu.input_0.datatype = sampler.sample_list(config.valid_datatypes);
            dpu.input_0.datatype =
                    config.restrict_datatype(dpu.input_0.datatype);  // do not generate for untrained data

            dpu.input_0.layout = config.valid_layouts[0];      // NOT COVERED
            dpu.input_0.swizzling = config.default_swizzling;  // NOT COVERED

            dpu.input_1.swizzling = config.default_swizzling;  // NOT COVERED

            dpu.output_0.datatype =
                    dpu.input_0.datatype;  // SAME as input, but will be overwritten in operation specific

            dpu.output_0.layout = config.valid_layouts[0];      // NOT COVERED
            dpu.output_0.swizzling = config.default_swizzling;  // NOT COVERED
        }

        {  // Operation dependent

            auto& operation_generator = my_validator.get_nodevice_operation_specific_generator(dpu.operation);

            // generate input/output
            operation_generator.generate_operation_dependent_tensors(sampler, config, dpu);
            dpu.output_0.datatype =
                    config.restrict_datatype(dpu.output_0.datatype);  // do not generate for untrained data

            //+ sparsity
            operation_generator.generate_sparsity(sampler, config, dpu);
        }
        {                                                                             // execution order
            dpu.execution_order = sampler.sample_list(config.valid_execution_order);  // uniform
        }

        return true;
    }

public:
    ///@brief Creates a random valid DPUWorkload, that fits in memory, for device or will throw
    /// The created workload will fit into the device memory
    ///
    /// @param device for which the DPUWOrkload will be created. it is set also into output
    /// @throws runtime_error if device is not supported or if too many tries to obtain a valid , memory constrained
    /// workload
    DPUWorkload create(VPUNN::VPUDevice device) const {
        if (!my_validator.is_supported(device)) {
            std::stringstream buffer;
            buffer << "[ERROR]: Workload Generator: Device Not Supported: " << static_cast<int>(device)
                   << "  t:" << VPUDevice_ToText.at(static_cast<int>(device)) << " File: " << __FILE__
                   << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::runtime_error(details);
        }

        Sampler sampler;
        const auto& config = my_validator.get_config(device);  // will not throw

        DPUOperation dpu;
        dpu.device = device;

        const int avaialable_cmx_memo{config.get_cmx_size(device)};

        bool sample_made_OK = false;
        for (int tries = 1; (false == sample_made_OK) && (tries < 100); ++tries) {
            auto sampled =
                    sample_tentative(sampler, config, dpu);  // restrictions(due to training) must be handled inside

            const auto cmx_mem{my_validator.compute_wl_memory(dpu)};  // memory is stricter
            if (sampled && (cmx_mem.cmx <= avaialable_cmx_memo)) {
                sample_made_OK = true;
            }
        }

        if (sample_made_OK) {  // generation done, move generated info to the actual WL
            const DPUWorkload wl{dpu.clone_as_DPUWorkload()};
            return wl;
        } else {
            std::stringstream buffer;
            buffer << "[ERROR]: Workload Generator: Cannot generate (probably memory limitations): "
                   << static_cast<int>(device) << "  t:" << VPUDevice_ToText.at(static_cast<int>(device))
                   << " File: " << __FILE__ << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::runtime_error(details);
        }
    }
};

/**
 * @brief A structure to generate random DPU workloads
 * @details Useful to generate random DPU workloads
 * Example: VPUNN::randDPUWorkload(VPUNN::VPUDevice::VPU_2_0)
 */
struct randDPUWorkload {
    /**
     * @brief randDPUWorkload VPUDevice
     *
     */
    const VPUNN::VPUDevice device;
    const DPU_OperationCreator generator;

public:
    /**
     * @brief Construct a new random DPUWorkload object
     *
     * @param device a VPUDevice object
     */
    randDPUWorkload(VPUNN::VPUDevice device): device(device) {
    }

    /**
     * @brief overloaded operator () to enable calling the struct
     *
     * @return VPUNN::DPUWorkload
     */
    VPUNN::DPUWorkload operator()() {
        return generator.create(device);
    }
};

}  // namespace VPUNN

#endif  //
