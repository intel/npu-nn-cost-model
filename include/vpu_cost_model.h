// Copyright © 2023 Intel Corporation
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
#include <utility>
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

#include "vpu/cycles_interface_types.h"
#include "vpu/validation/dpu_operations_sanitizer.h"

namespace VPUNN {

/**
 * @brief The VPUCostModel class
 *
 * Has behind a loaded CostModel neural network that infers cycle times for DPUWOrkloads
 *
 */
class VPUNN_API(VPUCostModel): public VPUNNPerformanceModel, VPUNNPowerModel {
private:
    const RuntimeProcessingFactory preprocessing_factory;  ///< provides Preprocessing objects
    Runtime vpunn_runtime;                ///< the loaded inference model is here, used for FW propagation
    Preprocessing<float>& preprocessing;  ///< prepares the input vector for the runtime, configured at ctor
    LRUCache<float> cache;                ///< cache for inferred values, used only for single workload, not for batches

    DPU_OperationSanitizer sanitizer;  ///< sanitizer mechanisms

    const size_t prealloc_results{1000};          ///< how much results buffer to pre-alloc
    std::vector<float> workloads_results_buffer;  ///< buffer dedicated for workloads. avoids reallocation.
    const float default_NN_output{-1.0F};  ///< this is the value used in no NN output is present (like not loaded).

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
     * @brief Ensures that input channels are equal to output channels for channel preserving operations
     *
     * @param workload a DPUWorkload that is checked and changed
     */
    void channels_preserving_operations_consistency_check(DPUWorkload& workload) const {
        if (workload.op == Operation::ELTWISE || workload.op == Operation::DW_CONVOLUTION ||
            workload.op == Operation::MAXPOOL || workload.op == Operation::AVEPOOL)
            if (workload.inputs[0].channels() != workload.outputs[0].channels()) {
                Logger::warning() << "Changed channels from " << workload.inputs[0].channels() << " to "
                                  << workload.outputs[0].channels();
                workload.inputs[0].set_shape({workload.inputs[0].x(), workload.inputs[0].y(),
                                              workload.outputs[0].channels(), workload.inputs[0].b()});
            }
    }

protected:
    /// @brief simulates AVGPOOL with another equivalent operation (DW CONV)
    ///
    /// @param workload [in, out] that will be changed in case the input is AVGPOOL
    void avgpool_replace_by(DPUWorkload& workload) const {
        if (workload.op == Operation::AVEPOOL) {
            Logger::warning() << "Workload with AVEPOOL changed to DW_CONVOLUTION";
            workload.op = Operation::DW_CONVOLUTION;
        }
    }

private:
    ///@brief changes the hw_overhead value according to some limits
    void sanitize_hw_overhead_result(float& hw_overhead, const DPUWorkload& workload) const {
        if (results_config.may_limit_hw_overhead) {
            if (hw_overhead < 1.0F && workload.act_sparsity == 0 && workload.weight_sparsity == 0) {
                // This makes sense only when NN output is NOT cycles, but real HW overhead that means
                // RealCycles/TheoreticalBestCase
                // If there is no sparsity, hw_overhead should be > 1. This check avoid
                // edge cases where NN predicts a smaller than 1 overhead. In that case, the network is obviously wrong.
                // This check fall-backs the cost to the theoretical cost model by forcing hw_overhead = 1
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

    /// 4 billion, any value higher than this might not be representable on UINT32, and
    /// should be treated like a not in range value given by the NN
    const float high_threshold{4000000000.0F};

    /// less than this is not representable on UINT32, and has no meanings in
    /// cycles. zero is still left possible to be returned, it might be a special
    /// way of network to communicate something (like no answer)
    const float low_threshold{0.0F};

    /// @brief checks if the NN returned value is invalid, is outside of usable range
    /// @param nn_output_cycles , the value to be analyzed, this is assumed given by the NN inference
    /// @return true if invalid value
    bool is_NN_value_invalid(const float nn_output_cycles) const {
        bool validity = false;
        if ((nn_output_cycles > high_threshold) || (nn_output_cycles < low_threshold)) {
            validity = true;
        }
        return validity;
    }

public:
    /// @brief provides the value interval where the NN raw outputs are considered valid and will be used to further
    /// compute information
    ///
    /// @returns a pair containing (minimum_valid_value maximum_valid_value)
    std::pair<float, float> get_NN_Valid_interval() const noexcept {
        return std::make_pair(low_threshold, high_threshold);
    }

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
        preprocessing.set_probable_batch(batch_size);
        workloads_results_buffer.reserve(prealloc_results);
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
        preprocessing.set_probable_batch(batch_size);
        workloads_results_buffer.reserve(prealloc_results);
    }

protected:
    ///@ brief checks some validity criteria and performs sanitization that does not alter relevance
    ///
    /// @sa DPU_OperationSanitizer::check_and_sanitize for details
    /// from legacy behavior is ensures that input channels are equal to output channels for channel preserving
    /// operations
    /// @param workload [in, out] to be checked and changed
    /// @param result [out] holds error code
    /// @returns true if checks were OK, false if this wl is not to be used
    bool sanitize_workload(DPUWorkload& workload, SanityReport& result) const {
        avgpool_replace_by(workload);  // AVEPOOL will be transformed to something equivalent

        channels_preserving_operations_consistency_check(workload);  // old style sanitation

        sanitizer.check_and_sanitize(workload, result);
        return result.is_usable();
    }

public:
    /**
     * @brief Compute the NN Output of a specific DPUWorkload
     * takes in consideration the cache
     * no sanitation is done
     * no check if network exists
     *
     * @param workload a DPUWorkload
     * @return float the NN raw output, not filtered
     */
    float run_NN(const DPUWorkload& workload) {
        const auto& vector = preprocessing.transform(workload);
        // Check for cache hit
        const float* const cached_value = cache.get(vector);
        if (cached_value == nullptr) {
            // run the model in case of a cache miss
            const auto infered_value = vpunn_runtime.predict<float>(vector)[0];
            // Add result to the cache
            cache.add(vector, infered_value);
            return infered_value;
        }
        return *cached_value;
    }

    /**
     * @brief Compute the NN Output of multiple DPUWorkloads
     * NOT taking in consideration the cache
     * no sanitation is done
     * no check if network exists
     *
     * @param workloads a std::vector of DPUWorkloads
     * @return a reference to the results vector. Do not store, will be invalid after next call to this method.
     */
    const std::vector<float>& run_NN(const std::vector<DPUWorkload>& workloads) {
        workloads_results_buffer.resize(workloads.size());

        if (!nn_initialized()) {  // do not run for bad NN
            std::fill(workloads_results_buffer.begin(), workloads_results_buffer.end(), default_NN_output);
            return workloads_results_buffer;
        }

        // Pre-process the workloads to generate descriptors
        const auto model_batch_size{vpunn_runtime.input_tensors()[0]->shape()[0]};  // how many wlds in a batch
        // transforms all at once, potential optimization is to do batch by batch
        const auto& vector = preprocessing.transform(workloads, model_batch_size);

        const auto descriptor_size{preprocessing.output_size()};
        const auto inputs_to_process_in_batch{descriptor_size * model_batch_size};

        for (unsigned int wl_idx = 0; wl_idx < workloads.size(); wl_idx += model_batch_size) {
            // Slice the workload descriptors and predict on a single batch
            const float* hw_overhead_arr =
                    vpunn_runtime.predict(&(vector[wl_idx * descriptor_size]), inputs_to_process_in_batch);

            const auto complete_batch_end_idx{wl_idx + model_batch_size};
            auto end_idx{(complete_batch_end_idx > workloads.size()) ? workloads.size() : complete_batch_end_idx};

            // fill into result the data for this batch
            for (unsigned int idx = wl_idx; idx < end_idx; ++idx) {
                workloads_results_buffer[idx] = hw_overhead_arr[idx - wl_idx];
            }
        }
        return workloads_results_buffer;
    }

public:
    /**
     * @brief Check if the internal VPUNN is initialized
     *
     * @return true the VPUNN neural network is initialized
     * @return false the VPUNN neural network is not initialized
     */
    bool nn_initialized() const {
        return vpunn_runtime.initialized();
    }

    /**
     * @brief Return the number of cycles needed to compute a workload
     *
     * Important: If no NN is available it will return Theoretical cycles for the workload. Check if NN is loaded with
     * nn_initialized()
     *
     * A sanity check will be performed on the workload and in case it is not suitable the method will return an error
     * code without running the inference on the NN. @sa DPU_OperationSanitizer::check_and_sanitize() explanations.
     * Some checks examples:
     * - if the device is supported
     * - if workload fits in CMX memory
     * - if the operation is supported
     *
     * List of Error codes is available in CyclesInterfaceType doc.
     *
     * A sanity check will be performed also on the NN output, in case the NN  raw value is not reliable it will not be
     * returned but an error code will be given, e.g. ERROR_INVALID_OUTPUT_RANGE
     *
     * To see the limits of valid NN values interval , use @sa get_NN_Valid_interval().  Zero is a value that will NOT
     * be filtered out.
     *
     * Behind the DPU computation is a trained Neural Network that might give unexpected results in case is asked about
     * a workload that is odd/(not well formed) or was not trained in that area or workloads.
     * The workload passed as parameter for inference should be a valid one, a one that makes sense, we are checking
     * some sanity, but ,for now, not a strict/extensive sanity check is performed. A workload with unrealistic
     * combinations of  parameters (eg DW_CONV with 7 input/output channels ) will not be detected.
     *
     * In case the wl configuration is unrealistic the network will give undefined(aberrant) results (it was not trained
     * on invalid data). The NN raw output is filtered for  generic valid interval (no negatives, no huge , e.g. 4bilion
     * cycles) but the user can also be aware of this behavior and use its own narrower ranges
     *
     * e.g.  Depending on the wl a cycle values of 10 might be unrealistic, also a value of 100milion cycles (@1Ghz is
     * ~100ms),  The user should be aware that not all aberrant/unrealistic NN outputs are handled inside.
     *
     *
     * @param wl a DPUWorkload to be evaluated.
     * @return unsigned int DPUWorkload execution cycles or an error code.
     *
     * @throws out_of_range : cache problems, cannot pre-process data , generate the NN descriptor due to data unknown
     * @throws runtime_error: cannot generate the NN descriptor, e.g expected sizes do not match
     *
     */
    CyclesInterfaceType DPU(DPUWorkload wl) {
        std::string dummy_info{};
        return DPU(wl, dummy_info);
    }

    /// @brief @see DPU(DPUWorkload wl)
    CyclesInterfaceType DPU(DPUWorkload wl, std::string& info) {
        // sanitize and check the input.
        const auto is_inference_posible = nn_initialized();
        SanityReport problems{};
        const auto is_inference_relevant = sanitize_workload(wl, problems);
        info = problems.info;

        // Compute the theoretical cycles
        const auto theoretical_cycles = DPUTheoreticalCycles(wl);

        CyclesInterfaceType cycles{problems.value()};  // neutral value or reported problems at sanitization
        if (is_inference_relevant) {
            if (is_inference_posible) {
                if (results_config.cycles_not_hw_overhead) {  // just cycles
                    const auto nn_output_cycles = run_NN(wl);
                    if (is_NN_value_invalid(nn_output_cycles)) {
                        cycles = Cycles::ERROR_INVALID_OUTPUT_RANGE;
                        // std::cout << "\n Problematic inf response: "<<nn_output_cycles<<std::endl<<wl<<"\n";
                    } else {
                        cycles = static_cast<CyclesInterfaceType>(ceil(nn_output_cycles));  // NORMAL CASE
                    }

                } else {  // output cycles are the theoretical ones times the hw overhead, old mode
                    auto hw_overhead = run_NN(wl);
                    sanitize_hw_overhead_result(hw_overhead, wl);  // old style
                    const auto cycles_computed = (theoretical_cycles * hw_overhead);
                    cycles = static_cast<CyclesInterfaceType>(ceil(cycles_computed));  // NORMAL CASE
                }

            } else {  // NN not available, use theoretical cycles
                cycles = theoretical_cycles;
            }
        }

        return cycles;
    }

    /**
     * @brief Return the number of cycles needed to compute multiple workloads
     *
     * @param workloads a std::vector of DPUWorkload
     * @return std::vector<CyclesInterfaceType> the DPUWorklaods execution cycles, @sa DPU for single wl for more
     * explanations
     */
    std::vector<CyclesInterfaceType> DPU(std::vector<DPUWorkload> workloads) {
        const auto number_of_workloads{workloads.size()};  ///< fixed value remembered here, workloads is non const
        std::vector<CyclesInterfaceType> cycles_vector = std::vector<CyclesInterfaceType>(number_of_workloads);
        const auto is_inference_posible = nn_initialized();

        /// @brief sanitization result element
        struct sanitizationOutcome {
            bool inference_relevance{false};
            SanityReport problems{};
        };
        std::vector<sanitizationOutcome> sanitization_results{number_of_workloads};

        // sanitize the input vector.
        for (unsigned int idx = 0; idx < number_of_workloads; ++idx) {
            auto& wl{workloads[idx]};
            auto& sanity{sanitization_results[idx]};
            sanity.inference_relevance = sanitize_workload(wl, sanity.problems);  // workloads are changed
        }

        // Compute using NN. Should not run if not initialized (fills a default value)
        const std::vector<float>& NN_results = run_NN(workloads);  // always tentative run

        // parse all and decide individually
        for (unsigned int idx = 0; idx < workloads.size(); ++idx) {
            auto& wl{workloads[idx]};
            const SanityReport& problems{sanitization_results[idx].problems};
            const auto is_inference_relevant{sanitization_results[idx].inference_relevance};
            const auto theoretical_cycles = DPUTheoreticalCycles(wl);

            CyclesInterfaceType cycles{problems.value()};  // neutral value or sanitization error
            if (is_inference_relevant) {
                if (is_inference_posible) {
                    if (results_config.cycles_not_hw_overhead) {  // just cycles
                        const auto nn_output_cycles = NN_results[idx];
                        if (is_NN_value_invalid(nn_output_cycles)) {
                            cycles = Cycles::ERROR_INVALID_OUTPUT_RANGE;
                        } else {
                            cycles = static_cast<CyclesInterfaceType>(ceil(nn_output_cycles));  // NORMAL CASE
                        }

                    } else {  // output cycles are the theoretical ones times the hw overhead, old mode
                        auto hw_overhead = NN_results[idx];
                        sanitize_hw_overhead_result(hw_overhead, wl);  // old style
                        const auto cycles_computed = (theoretical_cycles * hw_overhead);
                        cycles = static_cast<CyclesInterfaceType>(ceil(cycles_computed));
                    }

                } else {  // NN not available, use theoretical cycles
                    cycles = theoretical_cycles;
                }
            }

            cycles_vector[idx] = cycles;
        }

        return cycles_vector;
    }

    /**
     * @brief Compute DPUWorkload hw utilization
     *
     * @param workload a DPUWorkload
     * @return float DPUWorkload hardware utilization
     */
    float hw_utilization(DPUWorkload& workload) {
        // Compute the Hw utilization using the NN

        // sanitize and check the input.
        SanityReport problems{};
        const auto is_inference_relevant = sanitize_workload(workload, problems);
        const auto is_inference_posible = nn_initialized();

        auto utilization = 0.0F;  // signals problems

        // first priority is given to NN existence. No matter the WL if NN not available use ideal/theoretical
        if (is_inference_posible) {
            if (is_inference_relevant) {
                if (results_config.cycles_not_hw_overhead) {  // just cycles as output
                    const auto theoretical_cycles = DPUTheoreticalCycles(workload);
                    const auto nn_output_cycles = run_NN(workload);
                    if (is_NN_value_invalid(nn_output_cycles)) {
                        utilization = 0.0;
                    } else {
                        if (nn_output_cycles != 0.0F) {
                            utilization = theoretical_cycles / nn_output_cycles;  // NORMAL CASE
                        } else {
                            utilization = 0.0F;
                        }
                    }

                } else {  // inverse of overhead, old mode, overhead from NN
                    auto hw_overhead = run_NN(workload);
                    sanitize_hw_overhead_result(hw_overhead, workload);  // old style
                    utilization = (1.0F / (hw_overhead + 0.001f));       // NORMAL CASE
                }
            }  // if not relevant keep the default value for utilization

        } else {  // NN not available, use theoretical utilization =1
            utilization = (1.0F / (1.0F + 0.001f));
            // special formula from legacy code, not returning exactly one
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
     * @param input_location where is the source memory
     * @param output_location where is the destination memory
     * @param output_write_tiles how many CMX tiles the DMA broadcast
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(VPUDevice device, const VPUTensor& input, const VPUTensor& output,
                     MemoryLocation input_location = MemoryLocation::DRAM,
                     MemoryLocation output_location = MemoryLocation::CMX, unsigned int output_write_tiles = 1) const {
        // Call the helper function
        return DMATheoreticalCycles({device, input, output, input_location, output_location, output_write_tiles});
    }

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(const DMAWorkload& wl) const {
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
