// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPU_COST_MODEL_H
#define VPU_COST_MODEL_H

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include "core/cache.h"
#include "core/logger.h"

#include "inference/preprocessing.h"
#include "inference/preprop_factory.h"

#include "vpu/performance.h"
#include "vpu/power.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpunn.h"

namespace VPUNN {

/**
 * @brief The VPUCostModel class
 *
 */
class VPUNN_API(VPUCostModel): public VPUNNPerformanceModel, VPUNNPowerModel {
private:
    const RuntimeProcessingFactory preprocessing_factory;  ///< provides Preprocessing objects
    Runtime vpunn_runtime;                ///< the loaded inference model is here, used for FW propagation
    Preprocessing<float>& preprocessing;  ///< prepares the input vector for the runtime, configured at ctor
    LRUCache<float> cache;                ///< cache for inferred values, used only for single workload, not for batches

    /// @brief Configuration options concerning the interpretation and post processing of inferred values
    struct PostProcessConfig {
        bool cycles_not_hw_overhead;  ///< true: cycles network output, false: hw_overhead  NN output
        bool may_limit_hw_overhead;   ///< true: sometimes extra limitations are done on hw_overhead
    };

    /// output versions
    static constexpr int OUT_LATEST = 0;
    static constexpr int OUT_HW_OVERHEAD_BOUNDED = 1;
    static constexpr int OUT_CYCLES = 2;
    static constexpr int OUT_HW_OVERHEAD_UNBOUNDED = 3;

    const PostProcessConfig latest_results_config{true, false};  ///< will be use in case latest(00) version
    const PostProcessConfig
            results_config;  ///< active config, must be set up accordingly in the constructor, based on loaded NN

    /// @brief obtains the actual preprocessing instance from factory. The factory must live longer than the instance
    /// created. warning: Throws if not possible
    static Preprocessing<float>& init_preproc(const RuntimeProcessingFactory& factory,
                                              const ModelVersion& version_service, const std::string& filename) {
        // let's initialize the preproc aspects based on version
        const int version = version_service.get_input_interface_version();
        if (factory.exists_preprocessing(version)) {
            auto& preproc = factory.make_preprocessing(version);
            return preproc;

        } else {
            std::stringstream buffer;
            buffer << "Cannot create preprocessing stage!.Preprocessing with version (" << version
                   << ") is not known/supported. Filename: " << filename
                   << " , Version info (raw): " << version_service.get_raw_name();
            std::string details = buffer.str();
            Logger::error() << details;

            throw std::runtime_error(details);
        }
    }

    ///@brief establish what o do at post-processing (cycles versus hw_overhaed)
    static PostProcessConfig init_post_config(const ModelVersion& v, const PostProcessConfig& latest_config) {
        PostProcessConfig config{latest_config};
        const auto out_intf = v.get_output_interface_version();
        switch (out_intf) {
        case OUT_LATEST: {
            config = latest_config;
        } break;
        case OUT_HW_OVERHEAD_BOUNDED: {
            config.cycles_not_hw_overhead = false;  // hw overhead
            config.may_limit_hw_overhead = true;    // some limitations will apply, this is old mode
        } break;
        case OUT_CYCLES: {
            config.cycles_not_hw_overhead = true;  // cycles
            config.may_limit_hw_overhead = false;  // no limitations
        } break;
        case OUT_HW_OVERHEAD_UNBOUNDED: {
            config.cycles_not_hw_overhead = false;  // hw overhead
            config.may_limit_hw_overhead = false;   // no limitations
        } break;
        default:
            config = latest_config;
            break;
        }
        return config;
    }

    /**
     * @brief Sanitize invalid workloads
     *
     * @param workload a DPUWorkload
     */
    void sanitize_workload(DPUWorkload& workload) const {
        if (workload.op == Operation::ELTWISE || workload.op == Operation::DW_CONVOLUTION ||
            workload.op == Operation::MAXPOOL || workload.op == Operation::AVEPOOL)
            if (workload.inputs[0].channels() != workload.outputs[0].channels()) {
                Logger::warning() << "Changed channels from " << workload.inputs[0].channels() << " to "
                                  << workload.outputs[0].channels();
                workload.inputs[0].set_shape({workload.inputs[0].x(), workload.inputs[0].y(),
                                              workload.outputs[0].channels(), workload.inputs[0].b()});
            }
    }

    ///@brief changes the hw_overhead value according to some limits
    void sanitize_hw_overhead_result(float& hw_overhead, const DPUWorkload& workload) const {
        if (results_config.may_limit_hw_overhead) {
            if (hw_overhead < 1.0F && workload.act_sparsity == 0 && workload.weight_sparsity == 0) {
                // If there is no sparsity, hw_overhead should be > 1.
                // This check avoid edge cases where NN predicts a smaller than 1 overhead.
                // In that case, the network is obviously wrong.
                // This check fallback the cost to the theoretical cost model by forcing hw_overhead = 1
                hw_overhead = 1.0F;
            }
        }
    }

    /// @brief  check and try to make the preprocessing output to be the same as what model expects
    /// This mechanism is unsafe at this moment and lets you change the preprocessing output to a bigger or smaller size
    /// The result may be impossibility to run (if smaller) or leaving empty zeros at the end (only partial fill)
    ///
    void correlate_preprocessor_with_model_inputs() {
        const auto model_input_size = vpunn_runtime.input_tensors()[0]->shape()[1];
        const auto preprocessing_output_size = preprocessing.output_size();
        if (model_input_size != preprocessing_output_size) {
            Logger::warning() << "Changing preprocessing output size (" << preprocessing_output_size
                              << ") to the model input size (" << model_input_size << ")";
            preprocessing.set_size(model_input_size);
        }
    }

public:
    /**
     * @brief Construct a new VPUCostModel object
     *
     * @param filename the name of the .vpunn model
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit VPUCostModel(const std::string& filename = "", bool profile = false, const unsigned int cache_size = 16384,
                          const unsigned int batch_size = 1)
            : vpunn_runtime(filename, batch_size, profile),
              preprocessing(init_preproc(preprocessing_factory, vpunn_runtime.model_version_info(), filename)),
              cache(cache_size),
              results_config(init_post_config(vpunn_runtime.model_version_info(), latest_results_config)) {
        Logger::initialize();
        if (!vpunn_runtime.initialized()) {
            return;
        }

        correlate_preprocessor_with_model_inputs();
    }

    /**
     * @brief Construct a new VPUCostModel object
     *
     * @param model_data a buffer containing a .vpunn model
     * @param model_data_length the size of the model_data buffer
     * @param copy_model_data enable/disable the memcopy of the buffer
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit VPUCostModel(const char* model_data, size_t model_data_length, bool copy_model_data, bool profile = false,
                          const unsigned int cache_size = 16384, const unsigned int batch_size = 1)
            : vpunn_runtime(model_data, model_data_length, copy_model_data, batch_size, profile),
              preprocessing(init_preproc(preprocessing_factory, vpunn_runtime.model_version_info(), "ConstCharInit")),
              cache(cache_size),
              results_config(init_post_config(vpunn_runtime.model_version_info(), latest_results_config)) {
        Logger::initialize();
        if (!vpunn_runtime.initialized()) {
            return;
        }
        correlate_preprocessor_with_model_inputs();
    }

    /**
     * @brief Compute the Hardware Overhead/(NN Output) of a specific DPUWorkload
     *
     * @param workload a DPUWorkload
     * @return float
     */
    float compute_hw_overhead(DPUWorkload& workload) {
        // Here is where the magic happens
        if (!vpunn_runtime.initialized()) {
            return 1;
        }
        sanitize_workload(workload);
        auto vector = preprocessing.transform(workload);
        // Check for cache hit
        float* hw_overhead = cache.get(vector);
        if (hw_overhead == nullptr) {
            // run the model in case of a cache miss
            hw_overhead = (float*)vpunn_runtime.predict(&(vector[0]), preprocessing.output_size());
            // Add result to the cache
            cache.add(vector, *hw_overhead);
        }
        sanitize_hw_overhead_result(*hw_overhead, workload);
        return *hw_overhead;
    }

    /**
     * @brief Compute the Hardware Overhead/(NN Output) of multiple DPUWorkloads
     *
     * @param workloads a std::vector of DPUWorkload
     * @return float
     */
    std::vector<float> compute_hw_overhead(std::vector<DPUWorkload>& workloads) {
        // Pre-initialize the results
        auto result = std::vector<float>(workloads.size(), 1.0f);

        if (!vpunn_runtime.initialized()) {
            return result;
        }

        // Sanitize workloads
        for (auto& workload : workloads) {
            sanitize_workload(workload);
        }

        // Preprocess the workloads to generate descriptors
        auto model_batch_size = vpunn_runtime.input_tensors()[0]->shape()[0];
        auto vector = preprocessing.transform(workloads, model_batch_size);

        for (unsigned int batch_idx = 0; batch_idx < workloads.size(); batch_idx += model_batch_size) {
            // Slice the workload descriptors and predict on a single batch
            float* hw_overhead = (float*)vpunn_runtime.predict(&(vector[batch_idx * preprocessing.output_size()]),
                                                               preprocessing.output_size() * model_batch_size);
            auto end_idx =
                    batch_idx + model_batch_size > workloads.size() ? workloads.size() : batch_idx + model_batch_size;
            for (unsigned int idx = batch_idx; idx < end_idx; idx++) {
                result[idx] = hw_overhead[idx - batch_idx];
                sanitize_hw_overhead_result(result[idx], workloads[idx]);
            }
        }
        return result;
    }

    /**
     * @brief Check if the internal VPUNN is initialized
     *
     * @return true the VPUNN neural network is initialized
     * @return false the VPUNN neural network is not initialized
     */
    bool nn_initialized() {
        return vpunn_runtime.initialized();
    }

    /**
     * @brief Return the number of cycles needed to compute a workload
     *
     * @param wl a DPUWorkload
     * @return unsigned int DPUWorkload execution cycles
     */
    unsigned int DPU(DPUWorkload wl) {
        // Compute the Hw utilization using the NN
        const auto hw_overhead = compute_hw_overhead(wl);
        // Compute the theoretical cycles
        const auto theoretical_cycles = DPUTheoreticalCycles(wl);

        auto result = 0.0F;
        if (results_config.cycles_not_hw_overhead) {  // just cycles
            result = vpunn_runtime.initialized() ? hw_overhead : theoretical_cycles;
        } else {  // output cycles are the theoretical ones times the hw overhead
            result = (theoretical_cycles * hw_overhead);
        }
        return static_cast<unsigned int>(ceil(result));
    }

    /**
     * @brief Return the number of cycles needed to compute multiple workloads
     *
     * @param workloads a std::vector of DPUWorkload
     * @return std::vector<unsigned int> the DPUWorklaods execution cycles
     */
    std::vector<unsigned int> DPU(std::vector<DPUWorkload> workloads) {
        std::vector<unsigned int> cycles = std::vector<unsigned int>(workloads.size());
        // Compute the Hw utilization using the NN
        std::vector<float> hw_overhead = compute_hw_overhead(workloads);
        for (unsigned int idx = 0; idx < workloads.size(); idx++) {
            const auto theoretical_cycles = DPUTheoreticalCycles(workloads[idx]);
            auto result = 0.0F;
            if (results_config.cycles_not_hw_overhead) {  // just cycles
                result = vpunn_runtime.initialized() ? hw_overhead[idx] : theoretical_cycles;
            } else {  // output cycles are the theoretical ones times the hw overhead
                result = (theoretical_cycles * hw_overhead[idx]);
            }

            cycles[idx] = static_cast<unsigned int>(ceil(result));
        }
        // output cycles are the theoretical ones times the hw utilization
        return cycles;
    }

    /**
     * @brief Compute DPUWorkload hw utilization
     *
     * @param workload a DPUWorkload
     * @return float DPUWorkload hardware utilization
     */
    float hw_utilization(DPUWorkload& workload) {
        // Compute the Hw utilization using the NN
        auto hw_overhead = compute_hw_overhead(workload);  // this is just the NN output
        auto utilization = 0.0F;
        if (results_config.cycles_not_hw_overhead) {  // need to go back from cycles to utilization
            const auto theoretical_cycles = DPUTheoreticalCycles(workload);
            const float cycles{hw_overhead};  //< cycles are what the NN gives/ inferred
            utilization = nn_initialized() ? theoretical_cycles / cycles : 1.0F;
        } else {  // inverse of overhead
            utilization = 1.0F / (hw_overhead + 0.001f);
        }

        if (results_config.may_limit_hw_overhead) {
            // Clamp the utilization between 0 and 1 only if we are also clamping the hw overhead
            utilization = std::max(0.0f, std::min(utilization, 1.0f));
        }
        return utilization;
    }

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param device DMA VPUDevice
     * @param input DMA input Tensor
     * @param output DMA output Tensor
     * @param output_write_tiles how many CMX tiles the DMA broadcast
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(VPUDevice device, const VPUTensor& input, const VPUTensor& output,
                     MemoryLocation input_location = MemoryLocation::DRAM,
                     MemoryLocation output_location = MemoryLocation::CMX, unsigned int output_write_tiles = 1) {
        // Call the helper function
        return DMATheoreticalCycles({device, input, output, input_location, output_location, output_write_tiles});
    }

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(const DMAWorkload& wl) {
        // Call the helper function
        return DMATheoreticalCycles(wl);
    }

    /**
     * @brief Return the number of cycles needed to compute a Shave kernel
     *
     * @param swl a Shave Kernel
     * @return unsigned int the number of cycles of the Shave kernel
     */
    unsigned int SHAVE(SWOperation& swl) {
        return SHAVETheoreticalCycles(swl);
    }

    /**
     * @brief Compute the activity factor of a DPUWorkload
     *
     * @param wl a DPUWorkload
     * @return float the DPUWorkload activity factor
     */
    float DPUActivityFactor(DPUWorkload& wl) {
        VPUDevice device = wl.device;
        Operation operation = wl.op;
        unsigned int input_ch = wl.inputs[0].channels();
        VPUPowerFactorLUT power_factor_lut = VPUPowerFactorLUT(input_ch, operation, device);

        float pf_value = power_factor_lut.getValue(wl.inputs[0].get_dtype());
        float hw_util = hw_utilization(wl);

        float power_af = hw_util * pf_value;

        return power_af;
    }

    /**
     * @brief Compute the power (in mW) of a DPUWorkload
     *
     * @param wl a DPUWorkload
     * @return float DPUWorkload consumed power in mW
     */
    float DPUPower(DPUWorkload& wl) {
        return DPUPower(wl, getDefaultDVFS(wl.device));
    }

    /**
     * @brief Compute the power (in mW) of a DMAWorkload
     *
     * @param wl a DMAWorkload
     * @return float DMAWorkload consumed power in mW
     */
    float DMAPower(DMAWorkload& wl) {
        return DMAPower(wl, getDefaultDVFS(wl.device));
    }

    /**
     * @brief Compute the power (in mW) of a Shave Kernel
     *
     * @param wl a SWOperation describing a Shave Kernel
     * @return float Shave Kernel consumed power in mW
     */
    float SHAVEPower(SWOperation& wl) {
        return SHAVEPower(wl, getDefaultDVFS(wl.device));
    }

    /**
     * @brief Compute the power (in mW) of a DPUWorkload
     *
     * @param wl a DPUWorkload
     * @param dvfs a dynamic voltage frequency scaling (DVFS) point
     * @return float DPUWorkload consumed power in mW
     */
    float DPUPower(DPUWorkload& wl, DVFS dvfs) {
        // Get the C_dyn for the DPU and the activity factor
        float c_dyn = getCDyn(wl.device, VPUSubsystem::VPU_DPU);
        float activity_factor = DPUActivityFactor(wl);
        // COmpute the dynamic power
        return DynamicPower(c_dyn, activity_factor, dvfs);
    }

    /**
     * @brief Compute the power (in mW) of a DMAWorkload
     *
     * @param wl a DMAWorkload
     * @param dvfs a dynamic voltage frequency scaling (DVFS) point
     * @return float DMAWorkload consumed power in mW
     */
    float DMAPower(DMAWorkload& wl, DVFS dvfs) {
        float c_dyn = getCDyn(wl.device, VPUSubsystem::VPU_DMA);
        return DMAPower(wl, c_dyn, dvfs.voltage, dvfs.frequency);
    }

    /**
     * @brief Compute the power (in mW) of a Shave Kernel
     *
     * @param wl a Shave Kernel
     * @param dvfs a dynamic voltage frequency scaling (DVFS) point
     * @return float Shave Kernel consumed power in mW
     */
    float SHAVEPower(SWOperation& wl, DVFS dvfs) {
        float c_dyn = getCDyn(wl.device, VPUSubsystem::VPU_SHV);
        return SHAVEPower(wl, c_dyn, dvfs.voltage, dvfs.frequency);
    }

    /**
     * @brief Compute the power (in mW) of a DPUWorkload
     *
     * @param wl a DPUWorkload
     * @param c_dyn dynamic capacitance parameter
     * @param voltage voltage in V
     * @param frequency frequency in MHz
     * @return float DPUWorkload consumed power in mW
     */
    float DPUPower(DPUWorkload& wl, float c_dyn, float voltage, float frequency) {
        // Get the DPU activity factor
        float activity_factor = DPUActivityFactor(wl);
        // COmpute the dynamic power
        return DynamicPower(c_dyn, activity_factor, voltage, frequency);
    }

    /**
     * @brief Compute the power (in mW) of a DMAWorkload
     *
     * @param wl a DMAWorkload
     * @param c_dyn dynamic capacitance parameter
     * @param voltage voltage in V
     * @param frequency frequency in MHz
     * @return float DMAWorkload consumed power in mW
     */
    float DMAPower(DMAWorkload& wl, float c_dyn, float voltage, float frequency) {
        UNUSED(wl);
        // Get the DMA activity factor
        float activity_factor = 1.0f;
        // COmpute the dynamic power
        return DynamicPower(c_dyn, activity_factor, voltage, frequency);
    }

    /**
     * @brief Compute the power (in mW) of a Shave Kernel
     *
     * @param wl a Shave Kernel
     * @param c_dyn dynamic capacitance parameter
     * @param voltage voltage in V
     * @param frequency frequency in MHz
     * @return float Shave Kernel consumed power in mW
     */
    float SHAVEPower(SWOperation& wl, float c_dyn, float voltage, float frequency) {
        UNUSED(wl);
        // Get the SHAVE activity factor
        float activity_factor = 1.0f;
        // COmpute the dynamic power
        return DynamicPower(c_dyn, activity_factor, voltage, frequency);
    }
};
}  // namespace VPUNN

#endif  // VPU_COST_MODEL_H
