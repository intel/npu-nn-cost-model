// Copyright © 2024 Intel Corporation
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
#include <type_traits>
#include <utility>
#include <vector>

#include <mutex>
#include <thread>

#include "core/cache.h"
#include "core/logger.h"
#include "core/serializer.h"

#include "inference/post_process.h"
#include "inference/postprocessing_factory.h"
#include "inference/preprop_factory.h"

#include "core/utils.h"
#include "inference/vpunn_runtime.h"
#include "vpu/performance.h"
#include "vpu/power.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpu/validation/checker_utils.h"
#include "vpu/vpu_performance_model.h"

#include "vpu/cycles_interface_types.h"
#include "vpu/dpu_info_pack.h"
#include "vpu/validation/dpu_operations_sanitizer.h"

#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"

#include "inference/postprocessing_mocks.h"

#include "vpu_shave_cost_model.h"

#include "vpu/nn_cost_provider.h"

#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"
#include "vpu/dpu_theoretical_cost_provider.h"

#include "vpu/serialization/l1_cost_serialization_wrapper.h"

#ifdef VPUNN_BUILD_HTTP_CLIENT
#include "http_client/http_cost_provider.h"
#endif

#include "vpu/energy_interface.h"
#include "vpu/vpu_mutex.h"

namespace VPUNN {

/**
 * @brief The VPUCostModel class
 *
 * Has behind a loaded CostModel neural network that infers cycle times for DPUWOrkloads
 * ALso behind it need to have a dCIm model for the ops that support dCIM
 *
 */
/* coverity[rule_of_three_violation:FALSE] */
class VPUNN_API VPUCostModel {
private:
    mutable CSVSerializer serializer{};  ///< serializer for workloads, has its own file to save data

    const HWPerformanceModel performance{};  // performance instance, not used here

protected:
    // DPU cost providers
    const NNCostProvider dpu_nn_cost_provider;                      ///< NN cost provider for DPU
    const DPUTheoreticalCostProvider dpu_theoretical{performance};  ///< theoretical cost provider for DPU

    // DMA cost providers
    // No NN or measured DMA cost provider available
    const DMATheoreticalCostProvider dma_theoretical{};  ///< theoretical cost provider for DMA

    // SHAVE cost providers
    std::shared_ptr<SHAVECostModel> ptr_internal_shave_cost_model;  ///< shared ownership of SHAVE cost model
    SHAVECostModel& internal_shave_cost_model{*ptr_internal_shave_cost_model};

    // Energy interface
    const IEnergy my_energy{*this, internal_shave_cost_model,
                            performance};  ///< energy aspects, not used here but instantiated

#ifdef VPUNN_BUILD_HTTP_CLIENT
    std::unique_ptr<HttpDPUCostProvider> http_dpu_cost_provider;  ///< HTTP cost provider for DPU
#endif
    const std::string default_host = "irlccggpu04.ir.intel.com";  ///< default host for HTTP cost provider
    const int default_port = 5000;                                ///< default port for HTTP cost provider
    // should be const and init at ctor!
    std::string profiling_backend{"silicon"};  ///< backend for profiling service [silicon, vpuem]
    bool is_profiling_service_enabled{false};  ///< true if profiling service is enabled

    const DPU_OperationSanitizer sanitizer;  ///< sanitizer mechanisms

private:
    /**
     * @brief Ensures that input channels are equal to output channels for channel preserving operations
     *
     * @param workload a DPUWorkload that is checked and changed
     */
    void channels_preserving_operations_consistency_check(DPUWorkload& workload) const {
        if (workload.op == Operation::ELTWISE || workload.op == Operation::DW_CONVOLUTION ||
            workload.op == Operation::MAXPOOL || workload.op == Operation::AVEPOOL) {
            if (!workload.output_autopad && workload.outputs[0].channels() >= 16) {
                if (workload.inputs[0].channels() != workload.outputs[0].channels()) {
                    Logger::warning() << "Changed channels from " << workload.inputs[0].channels() << " to "
                                      << workload.outputs[0].channels();

                    workload.inputs[0].set_shape({workload.inputs[0].x(), workload.inputs[0].y(),
                                                  workload.outputs[0].channels(), workload.inputs[0].b()});
                }
            }
        }
    }

    /**
     * @brief Initializes the profiling service.
     *
     * This function checks environment variables to determine if the profiling service should be enabled.
     * If enabled, it initializes the HTTP DPU cost provider with the specified host, port, and backend.
     *
     * @return true if the profiling service was successfully initialized and is available, false otherwise.
     */
    bool init_profiling_service() {
#ifdef VPUNN_BUILD_HTTP_CLIENT
        bool use_profiling_service{
                get_env_vars({"ENABLE_VPUNN_PROFILING_SERVICE"}).at("ENABLE_VPUNN_PROFILING_SERVICE") == "TRUE"};
        if (use_profiling_service) {
            if (http_dpu_cost_provider == nullptr) {
                std::string env_host =
                        get_env_vars({"VPUNN_PROFILING_SERVICE_HOST"}).at("VPUNN_PROFILING_SERVICE_HOST");
                int env_port = 0;
                try {
                    env_port = std::stoi(
                            get_env_vars({"VPUNN_PROFILING_SERVICE_PORT"}).at("VPUNN_PROFILING_SERVICE_PORT"));
                } catch (const std::exception&) {
                    env_port = 0;
                }
                std::string env_backend =
                        get_env_vars({"VPUNN_PROFILING_SERVICE_BACKEND"}).at("VPUNN_PROFILING_SERVICE_BACKEND");

                std::string host = env_host.empty() ? default_host : env_host;
                int port = env_port == 0 ? default_port : env_port;
                profiling_backend = env_backend.empty() ? profiling_backend : env_backend;

                http_dpu_cost_provider = std::make_unique<HttpDPUCostProvider>(host, port);
            }
        }

        return (http_dpu_cost_provider != nullptr) && (http_dpu_cost_provider->is_available(profiling_backend));
#else
        return false;
#endif
    }

public:
    /// returns a reference of energy object
    /// owned by the current costmodel
    /// the lifetime of this pointer is bound to the lifetime of costmodel object
    /// if costmodel is destroyed or goes out of scope, the reference will also become invalid
    const IEnergy& getEnergyInterface() const {
        return my_energy;
    }

    // choose a better name
    /// returns a reference of performance object
    const HWPerformanceModel& getPerformanceModel() const {
        return performance;
    }

    const IDeviceValidValues& getSanitizerDeviceConfiguration(VPUDevice device) const {
        return sanitizer.getDeviceConfiguration(device);  // just a forwarder
    }

    /// @brief simulates AVGPOOL with another equivalent operation (DW CONV), depends also on Device
    ///
    /// @param workload [in, out] that will be changed in case the input is AVGPOOL
    void avgpool_replace_by(DPUWorkload& workload) const {
        if (Operation::AVEPOOL == workload.op) {
            if (VPUDevice::NPU_5_0 > workload.device) {
                Logger::warning() << "Workload with AVEPOOL changed to DW_CONVOLUTION";
                workload.op = Operation::DW_CONVOLUTION;
            } else {
                Logger::warning() << "Workload with AVEPOOL  NOT changed!, Device should support it!, ";
            }
        }
    }

    /// @brief Presumes any VPU27++ CONV with IC <16 to be compressed CONV. This is known by NN as CM_CONV
    ///
    /// @param workload [in, out] that will be changed in case the input is presumed compressed CONV
    void compressConv_replace_by_CM_CONV_VPU27(DPUWorkload& workload) const {
        if (workload.device >= VPUDevice::VPU_2_7) {
            if ((workload.op == Operation::CONVOLUTION) &&
                ((workload.inputs[0].channels() >= 1) && (workload.inputs[0].channels() < 16) &&
                 !workload.input_autopad)) {
                Logger::warning() << "Workload with CONVOLUTION compressed IC[1..15] transformed to CM_CONV ";
                workload.op = Operation::CM_CONVOLUTION;
            }
        }
    }

public:
    /// @brief Turns OFF the swizzling
    ///
    /// @param workload [in, out] that will be changed in case the conditions are met
    void swizzling_turn_OFF(DPUWorkload& workload) const {
        if constexpr (false == PerformanceMode::allowLegacySwizzling_G5) {
            // only for some devices
            if (workload.device >= VPUDevice::NPU_5_0) {
                workload.set_all_swizzlings(Swizzling::KEY_0);
            }
        }
    }

public:
    /// @brief Get a reference to the serializer
    /// temporary only for testing aspects (extra save ). TO BE REFACTORED
    CSVSerializer& get_serializer() noexcept {
        return serializer;
    }

    const NNCostProvider& get_NN_cost_provider() const noexcept {
        return dpu_nn_cost_provider;
    }

    /**
     * @brief Construct a new VPUCostModel object
     *
     * @param filename the name of the .vpunn model
     * @param dpu_cache_filename the name of the cache file
     * @param shave_cache_filename the name of the shave cache file
     *
     */
    explicit VPUCostModel(const std::string& filename, const std::string& dpu_cache_filename,
                          const std::string& shave_cache_filename, bool use_shave_2_api = false,
                          bool tryToLoadPairedCache = false)
            : VPUCostModel(filename, false, 16384, 1, dpu_cache_filename, shave_cache_filename, use_shave_2_api,
                           tryToLoadPairedCache) {
        /* coverity[uninit_member] */
    }

    /**
     * @brief Construct a new VPUCostModel object
     *
     * @param filename the name of the .vpunn model
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     * @param dpu_cache_filename the name of the cache file
     * @param shave_cache_filename the name of the shave cache file
     * @param tryToLoadPairedCache , special condition: if main file empty, try to load the paired cache, generated name
     * based on NN file. (fro DPU, for others?) DRAFT!
     *
     */
    explicit VPUCostModel(const std::string& filename = "", bool profile = false, const unsigned int cache_size = 16384,
                          const unsigned int batch_size = 1, const std::string& dpu_cache_filename = "",
                          const std::string& shave_cache_filename = "", bool use_shave_2_api = false,
                          bool tryToLoadPairedCache = false)
            : dpu_nn_cost_provider(filename, batch_size, profile, cache_size, dpu_cache_filename, tryToLoadPairedCache),
              ptr_internal_shave_cost_model(
                      std::make_shared<SHAVECostModel>(shave_cache_filename, cache_size, use_shave_2_api)) {
        Logger::initialize();

        if (!dpu_nn_cost_provider.is_initialized()) {
            return;
        }
        is_profiling_service_enabled = init_profiling_service();
        serializer.initialize("l1_dpu_workloads", FileMode::READ_WRITE,
                              dpu_nn_cost_provider.get_names_for_serializer());
    }

    VPUCostModel(const VPUCostModel&) = delete;
    // VPUCostModel(VPUCostModel&&) = default; //explicitly defaulted move constructor is implicitly deleted
    virtual ~VPUCostModel() = default;

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
                          const unsigned int cache_size = 16384, const unsigned int batch_size = 1,
                          const char* dpu_cache_data = nullptr, size_t dpu_cache_data_length = 0,
                          const char* shave_cache_data = nullptr, size_t shave_cache_data_length = 0,
                          bool use_shave_2_api = false)
            : dpu_nn_cost_provider(model_data, model_data_length, batch_size, copy_model_data, profile, cache_size,
                                   dpu_cache_data, dpu_cache_data_length),
              ptr_internal_shave_cost_model(std::make_shared<SHAVECostModel>(shave_cache_data, shave_cache_data_length,
                                                                             cache_size, use_shave_2_api)) {
        Logger::initialize();

        if (!dpu_nn_cost_provider.is_initialized()) {
            return;
        }
        is_profiling_service_enabled = init_profiling_service();
        serializer.initialize("l1_dpu_workloads", FileMode::READ_WRITE,
                              dpu_nn_cost_provider.get_names_for_serializer());
    }

    // protected:
public:
    /// @brief checks some validity criteria and performs sanitization that does not alter relevance
    ///
    /// @sa DPU_OperationSanitizer::check_and_sanitize for details
    /// from legacy behavior is ensures that input channels are equal to output channels for channel preserving
    /// operations
    /// @param workload [in, out] to be checked and changed
    /// @param result [out] holds error code
    /// @returns true if checks were OK, false if this wl is not to be used
    bool sanitize_workload(DPUWorkload& workload, SanityReport& result) const {
        avgpool_replace_by(workload);  // AVEPOOL will be transformed to something equivalent
        compressConv_replace_by_CM_CONV_VPU27(workload);

        channels_preserving_operations_consistency_check(workload);  // old style sanitation

        sanitizer.check_and_sanitize(workload, result);
        return result.is_usable();
    }

protected:
    /**
     * @brief Compute runtime when both sparsity (input and weight) are active.
     * runtime will be a combined value (now is minimum) of the pair obtained by activating only one sparsity.
     *
     * @param workload is the workload to be inferred
     * @return the runtime or error
     */
    CyclesInterfaceType get_cost_dualsparsity(const DPUWorkload& workload) const {
        // run twice
        std::string info;
        const auto wl_noAct{cloneDeactivateActSparsity(workload)};
        const CyclesInterfaceType act_off = run_cost_providers(wl_noAct, info);

        const auto wl_noWt{cloneDeactivateWeightSParsity(workload)};
        const CyclesInterfaceType wt_off = run_cost_providers(wl_noWt, info);

        // case when act_off is invalid, also this catch the case when both of values are invalid
        //   does not count which of them we return if both are invalid
        if (Cycles::isErrorCode(act_off)) {
            return act_off;
        }

        // case when wt_off is invalid
        if (Cycles::isErrorCode(wt_off)) {
            return wt_off;
        }

        // MIN.. their combined runtime reduction should be at least like their independent one?
        // IN general CostMOdelis a best case (due to memory contention, DMA ), but here  we keep this not so best case
        // (too many variations) alternative: more policies and we chose which one has best FPS over models optimization
        // on min:  run only once , for max sparsity (maybe balanced...)
        CyclesInterfaceType ret{std::min(act_off, wt_off)};  // algorithm here

        return ret;  // exit point
    }
    /**
     * @brief Wrapper over run_cost_providers for handling situations where a workload cannot be resolved with only one
     * inference. Now it is handling the dual input sparsity (activation and weights): compute the runtime using the
     * algorithm @sa get_cost_dualsparsity() for more explanations.
     *
     * @param workload is the workload to be inferred
     * @return the runtime or error
     */
    CyclesInterfaceType get_cost(const DPUWorkload& workload, std::string& info,
                                 std::string* cost_source = nullptr) const {
        // @todo : impact on energy?, CHeck if energy/DPUINfo/Theoretical cycles/ops considers this situation to reduce
        // energy

        // if weight and input (SEP irrelevant?) sparsity ON.
        if (is_dualsparsity_active(workload)) {
            return get_cost_dualsparsity(workload);
        }

        return run_cost_providers(workload, info, cost_source);  // normal wl handling
    }

    /**
     * @brief Run the cost providers for a given workload.
     *
     * This function checks if the DPU NN cost provider is initialized and retrieves the cost from the cache or
     * profiling service. If the profiling service is not available, it falls back to the DPU NN cost provider or
     * theoretical cycles.
     *
     * @param workload The DPU workload to be processed.
     * @param info A string to store additional information about the cost source.
     * @param cost_source A string to store the source of the cost.
     * @return The number of cycles required for the workload.
     */
    CyclesInterfaceType run_cost_providers(const DPUWorkload& workload, std::string& info,
                                           std::string* cost_source = nullptr) const {
        CyclesInterfaceType cycles{Cycles::NO_ERROR};
        const bool is_inference_possible = dpu_nn_cost_provider.is_initialized();

        const auto try_cache = [&]() -> CyclesInterfaceType {
            return dpu_nn_cost_provider.get_cached(workload, cost_source);
        };
        const auto try_profiling = [&]() -> CyclesInterfaceType {
            if (!is_profiling_service_enabled) {
                return Cycles::ERROR_PROFILING_SERVICE;
            }

            auto get_profiling_backend_string = [&](ProfilingServiceBackend backend) -> std::string {
                if (backend != ProfilingServiceBackend::__size) {
                    return mapToText<ProfilingServiceBackend>().at(static_cast<int>(backend));
                }
                return profiling_backend;
            };

            if (cost_source) {
                *cost_source =
                        "profiling_service_" + get_profiling_backend_string(workload.profiling_service_backend_hint);
            }

            auto dpu_op = DPUOperation(workload, sanitizer.getDeviceConfiguration(workload.device));

#ifdef VPUNN_BUILD_HTTP_CLIENT
            return http_dpu_cost_provider->getCost(
                    dpu_op, info, get_profiling_backend_string(workload.profiling_service_backend_hint));
#else
            (void)dpu_op;
            (void)info;
            return Cycles::ERROR_PROFILING_SERVICE;
#endif
        };
        const auto try_nn = [&]() -> CyclesInterfaceType {
            if (!is_inference_possible) {
                return Cycles::ERROR_INFERENCE_NOT_POSSIBLE;
            }
            if (cost_source) {
                *cost_source = "nn_" + dpu_nn_cost_provider.get_model_nickname();
            }
            return dpu_nn_cost_provider.get_cost(workload);
        };
        const auto try_theoretical = [&]() -> CyclesInterfaceType {
            if (cost_source) {
                *cost_source = "theoretical";
            }
            return dpu_theoretical.DPUTheoreticalCycles(workload);
        };

        if (workload.cost_source_hint == CostSourceHint::AUTO) {
            // 1. Cache
            const auto cached = try_cache();
            if (!Cycles::isErrorCode(cached)) {
                return cached;
            }

            // 2. Profiling service
            cycles = try_profiling();

            // 3. Fallbacks (NN then theoretical)
            if (Cycles::isErrorCode(cycles) || cycles == Cycles::NO_ERROR) {
                cycles = try_nn();
                if (Cycles::isErrorCode(cycles)) {
                    cycles = try_theoretical();
                }
            }

            // Share result with NN cache if needed (only if cache had no valid entry)
            if (Cycles::isErrorCode(dpu_nn_cost_provider.get_cached(workload)) && !Cycles::isErrorCode(cycles)) {
                dpu_nn_cost_provider.add_to_cache(workload, static_cast<float>(cycles));
            }
            return cycles;

        } else if (workload.cost_source_hint == CostSourceHint::PROFILING_SERVICE) {
            cycles = try_profiling();

        } else if (workload.cost_source_hint == CostSourceHint::NN) {
            cycles = try_nn();

        } else if (workload.cost_source_hint == CostSourceHint::THEORETICAL) {
            cycles = try_theoretical();
        }

        return cycles;
    }

    /**
     * @brief checks if both input tensor sparsity(activation) and wl weight sparsity are active
     *
     * @param wl is the workload
     * @return true if both sparsities are active (sparsity enable true)
     */
    bool static is_dualsparsity_active(const DPUWorkload& wl) {
        return (wl.inputs[0].get_sparsity() && wl.weight_sparsity_enabled);
    }

    /**
     * @brief Deactivates input sparsity for a workload
     *
     * @param wl is the workload
     * @return a clone of the original wl with deactivated sparsity
     */
    DPUWorkload cloneDeactivateActSparsity(const DPUWorkload& wl) const {
        DPUWorkload out{wl};
        out.act_sparsity = 0.0;
        out.inputs[0].set_sparsity(false);
        return out;
    }

    /**
     * @brief Deactivates weight sparsity for a workload
     *
     * @param wl is teh workload
     * @return a clone of the original wl with deactivated sparsity
     */
    DPUWorkload cloneDeactivateWeightSParsity(const DPUWorkload& wl) const {
        DPUWorkload out{wl};
        out.weight_sparsity = 0.0;
        out.weight_sparsity_enabled = false;
        return out;
    }

    /**
     * @brief Wrapper over run_NN for batches for handling situations where a workload cannot be resolved with only one
     * inference. Now it is handling the dual input sparsity (activation and weights): this situation is not supported,
     * would complicate the implementation too much
     *
     *
     * @param workloads a std::vector of DPUWorkloads
     * @return a vector of runtimes
     * @throws runtime_error: when input sparsity and weight sparsity are active at the same time for at least one
     * workload (first workload)
     */
    const std::vector<CyclesInterfaceType> get_cost(const std::vector<DPUWorkload>& workloads) const {
        // here we check if at least one wl in vector workloads has act sparsity and weight sparsity active
        const bool exists_dual_spars{
                std::any_of(workloads.cbegin(), workloads.cend(), VPUCostModel::is_dualsparsity_active)};

        if (exists_dual_spars) {
            std::vector<CyclesInterfaceType> cycles_vector;
            cycles_vector.reserve(workloads.size());
            std::transform(workloads.cbegin(), workloads.cend(), cycles_vector.begin(), [this](const DPUWorkload& wl) {
                std::string info, source;
                return get_cost(wl, info, &source);
            });

            return cycles_vector;
        } else {
            return dpu_nn_cost_provider.get_cost(workloads);  // normal execution
        }
    }

protected:
public:
    /**
     * @brief Check if the internal VPUNN is initialized
     *
     * @return true the VPUNN neural network is initialized
     * @return false the VPUNN neural network is not initialized
     */
    bool nn_initialized() const {
        return dpu_nn_cost_provider.is_initialized();
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
    /* coverity[pass_by_value] */
    CyclesInterfaceType DPU(DPUWorkload wl) const {
        std::string dummy_info{};
        return DPU(std::move(wl), dummy_info);
    }

    /// @brief same like  @see DPU(DPUWorkload wl) , the extra param is to have as output the textual errors/findings
    /// discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param li will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    std::tuple<CyclesInterfaceType, std::string> DPUMsg(DPUWorkload wl) const {
        std::string dummy_info{};
        auto previous_print_mode{Checker::set_print_tags(false)};
        const auto r{DPU(wl, dummy_info)};
        Checker::set_print_tags(previous_print_mode);
        return std::make_tuple(r, dummy_info);
    }

    /// @brief same like  @see DPU(DPUWorkload wl) , the extra param is to have as output the textual errors/findings
    /// discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param info [out] will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    CyclesInterfaceType DPU(DPUWorkload wl, std::string& info) const {
        return DPU_and_sanitize(wl, info);
    }

protected:
    /* @brief DPU + wl param is also output as sanitized one.
     *  Provides workload outside so we know on what(post sanitization) was done the inference
     */
    CyclesInterfaceType DPU_and_sanitize(DPUWorkload& wl, std::string& info) const {
        swizzling_turn_OFF(wl);  // swizz guard sanitization


        L1CostSerializationWrap serialization_handler(serializer);

        serialization_handler.serializeInfoAndComputeWorkloadUid(wl);

        // sanitize and check the input.
        SanityReport problems{};
        const auto is_inference_relevant = sanitize_workload(wl, problems);
        info = problems.info;

        std::string cost_source = "unknown";
        CyclesInterfaceType cycles{problems.value()};  // neutral value or reported problems at sanitization

        if (is_inference_relevant) {
            cycles = get_cost(wl, info, &cost_source);
        }

        serialization_handler.serializeCyclesAndCostInfo_closeLine(cycles, std::move(cost_source), info);

        return cycles;
    }

public:
    /**
     * @brief Return the number of cycles needed to compute multiple workloads
     *
     * @param workloads a std::vector of DPUWorkload
     * @return std::vector<CyclesInterfaceType> the DPUWorklaods execution cycles, @sa DPU for single wl for more
     * explanations
     */
    std::vector<CyclesInterfaceType> DPU(std::vector<DPUWorkload> workloads) const {
        std::vector<DPUWorkload> serializer_orig_wls;  // should be const
        if (serializer.is_serialization_enabled()) {   // has to be factored out
            serializer_orig_wls = std::vector<DPUWorkload>(workloads.size());
            std::copy(workloads.begin(), workloads.end(), serializer_orig_wls.begin());
        }

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
            swizzling_turn_OFF(wl);                                               // swizz guard sanitization
            sanity.inference_relevance = sanitize_workload(wl, sanity.problems);  // workloads are changed
        }

        // Compute using NN. Should not run if not initialized (fills a default value)
        const auto costs = get_cost(workloads);  // always tentative run, what if throws?

        // parse all and decide individually
        for (unsigned int idx = 0; idx < workloads.size(); ++idx) {
            auto& wl{workloads[idx]};
            const SanityReport& problems{sanitization_results[idx].problems};
            const auto is_inference_relevant{sanitization_results[idx].inference_relevance};
            const auto theoretical_cycles = dpu_theoretical.DPUTheoreticalCycles(wl);

            CyclesInterfaceType cycles{problems.value()};  // neutral value or sanitization error
            if (is_inference_relevant) {
                if (is_inference_posible) {
                    cycles = costs[idx];

                } else {  // NN not available, use theoretical cycles
                    cycles = theoretical_cycles;
                }
            }

            cycles_vector[idx] = cycles;
        }

        const std::string dpu_nickname{get_NN_cost_provider().get_model_nickname()};
        L1CostSerializationWrap serialization_handler(serializer);
        serialization_handler.serializeCyclesAndComputeWorkloadUid_closeLine(std::move(serializer_orig_wls),
                                                                             cycles_vector, dpu_nickname);

        return cycles_vector;
    }

public:
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
     * @deprecated Will be removed in future releases
     */
    unsigned int DMA(VPUDevice device, const VPUTensor& input, const VPUTensor& output,
                     MemoryLocation input_location = MemoryLocation::DRAM,
                     MemoryLocation output_location = MemoryLocation::CMX, unsigned int output_write_tiles = 1) const {
        // Call the helper function. TO DO Adjust theoretical based on some measured data!
        return dma_theoretical.get_cost({device, input, output, input_location, output_location, output_write_tiles});
    }

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     * @deprecated Will be removed in future releases
     */
    unsigned int DMA(const DMAWorkload& wl) const {
        return DMA(wl.device, wl.input, wl.output, wl.input_location, wl.output_location, wl.output_write_tiles);
    }

    /**
     * @brief Return the number of cycles needed to compute a Shave kernel
     *
     * @param shave_wl a Shave workload contains: name of kernel, device, in out tensor. PLus optional parameters of the
     * operations
     * @param infoOut  a string that will contain informative error information (in case of error)
     * @return the number of cycles of the Shave kernel, in DPU cycles of the desired device nominal frequency. OR ERROR
     */
    CyclesInterfaceType SHAVE(const SHAVEWorkload& shave_wl, std::string& infoOut) const {
        return internal_shave_cost_model.computeCycles(shave_wl, infoOut);
    }

    CyclesInterfaceType SHAVE(const SHAVEWorkload& shave_wl) const {
        return internal_shave_cost_model.computeCycles(shave_wl);
    }

    /**
     * @brief Return the number of cycles needed to compute a Shave kernel without posibility of skipping Cache
     *
     * @param shave_wl a Shave workload contains: name of kernel, device, in out tensor. PLus optional parameters of the
     * operations
     * @param infoOut  a string that will contain informative error information (in case of error)
     * @return the number of cycles of the Shave kernel, in DPU cycles of the desired device nominal frequency. OR ERROR
     */
    CyclesInterfaceType SHAVE(const SHAVEWorkload& shave_wl, std::string& infoOut, bool skipCacheValues) const {
        return internal_shave_cost_model.computeCycles(shave_wl, infoOut, skipCacheValues);
    }

    /// gets the list of names of supported operators on a specified device. Each device has own operators
    ///
    /// @param device  for what device?
    /// @returns container with the name of operators
    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const {
        return internal_shave_cost_model.getShaveSupportedOperations(device);
    };

    /// provides a reference to the operator executor specified by name
    /// The executor can be used to execute (run) the runtime prediction on different tensors/parameters
    /// Or can be asked to print information about its implementation parameters
    ///
    /// @param name of the operator
    /// @param device name
    /// @returns a ref (no ownership transfered. exists as long as this VPUCostModel instance exists)
    const ShaveOpExecutor& getShaveInstance(std::string name, VPUDevice device) const {
        const std::string original_name = name;  // Store the name before moving
        auto result = internal_shave_cost_model.getShaveInstance(std::move(name), device);
        if (!result.has_value()) {
            throw std::runtime_error("Shave instance not found for name: " + original_name);
        }
        return result.value().get();
    }

    /**
     * @brief Checks if the SHAVE Gen 2 API is being used.
     * @return True if the SHAVE Gen 2 API is used, otherwise false.
     */
    bool isShave2ApiUsed() const {
        return internal_shave_cost_model.isShave2APIused();
    }

public:
    /**
     * @brief Compute the energy of a DPUWorkload.
     * @details This is a relative energy metric with a time base in DPU clock cyles. Energy of
     * 1000 would mean energy of worst case power for 1000 DPU clock cycles at reference dynamic power (power virus
     * for INT08 operations). measured in PowerVirusJoules = PowerVirus*cycle
     * @param wl a DPUWorkload
     * @return float the DPUWorkload energy, measured  PowerVirus*cycle
     */
    float DPUEnergy(const DPUWorkload& wl) const {
        return getEnergyInterface().DPUEnergy(wl);
    }

public:
    /** @brief Compute the energy of a SHAVE SHAVEWorkload.
     * @details Energy here is a relative metric, but the activity factor of the operation multiplied by
     *          its cost (number of clock cycles). We assume a constant activity factor of 0.5 for all and a max
     *          power of 5% of the DPU max power.
     *
     * @param swl a SHAVEWorkload
     * @return float the operation energy, in units relative to DPU PowerVirus. WIl return zero in case of error
     */
    float SHAVEEnergy(const SHAVEWorkload& swl) const {
        return getEnergyInterface().SHAVEEnergy(swl);
    }

    /// @brief same like  @see DPU(DPUWorkload wl) but return a Pack of information regarding the workload
    /// The purpose of this Method is to replace several separate calls to individual informations about the same
    /// workload.
    /// For example , estimated  cycle-times, errors, energy, activity factor, all can be obtained in one call.
    /// This method has the potential to be more efficient that the collection of individual ones.
    /// @param wl the workload to infer on
    /// @returns a Structure with all info that L1 APi can provide about this Workload
    DPUInfoPack DPUInfo(const DPUWorkload& workload) const {
        DPUInfoPack allData;      // expect RVO when returning it!
        DPUWorkload w{workload};  // local clone

        allData.DPUCycles = DPU_and_sanitize(
                w, allData.errInfo);  // do this first, might change w. It considers both sparsities if activated

        getEnergyInterface().fillDPUInfo(allData, w);

        allData.hw_theoretical_cycles = dpu_theoretical.DPUTheoreticalCycles(w);

        return allData;  // rvo
    }

    /////// Section for dCIM interfaces
public:
    //// provides the interface that has methods for DCiM
    // const DCiMCostModelInterface<DCiM_Workload_Interface>& getDCiM_interface() const {
    //     return *this;
    // }

    const AccessCounter& getPreloadedCacheCounter() const {
        return dpu_nn_cost_provider.getPreloadedCacheCounter();
    }

    const AccessCounter& getPreloadedShaveCacheCounter() const {
        return internal_shave_cost_model.getPreloadedCacheCounter();
    }
};  // class
}  // namespace VPUNN

#endif  // VPU_COST_MODEL_H
