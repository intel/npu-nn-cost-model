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

#include <string>
#include <tuple>
#include <vector>

#include "core/persistent_cache.h"
#include "core/serializer.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"
#include "vpu/dpu_info_pack.h"
#include "vpu/dpu_theoretical_cost_provider.h"
#include "vpu/energy_interface.h"
#include "vpu/nn_cost_provider.h"
#include "vpu/performance.h"
#include "vpu/shave/shave_devices.h"
#include "vpu/types.h"
#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/vpu_performance_model.h"
#include "vpu_shave_cost_model.h"

#include "vpu/profiling_service.h"
#include "vpu/http_cost_provider_intf.h"
#include "vpu/http_cost_provider_factory.h"

namespace VPUNN {

/**
 * @brief The VPUCostModel class
 *
 * Has behind a loaded CostModel neural network that infers cycle times for DPUWorkloads
 * Also behind it need to have a dCIM model for the ops that support dCIM
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

    const std::unique_ptr<IHttpCostProvider> http_dpu_cost_provider; ///< HTTP cost provider for DPU
    
    const DPU_OperationSanitizer sanitizer;  ///< sanitizer mechanisms

private:
    /// cache to store linearly extrapolation property of the NN. does not change after ctor!
    const bool is_linearly_extrapolation_necessary_cache_capability{
            dpu_nn_cost_provider.get_preprocessing().supportsProperty("DW_MXP_AVP_SupportsMoreThan64Ch")};
    /**
     * @brief Ensures that input channels are equal to output channels for channel preserving operations
     *
     * @param workload a DPUWorkload that is checked and changed
     */
    void channels_preserving_operations_consistency_check(DPUWorkload& workload) const;

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
    void avgpool_replace_by(DPUWorkload& workload) const;

    /// @brief Presumes any VPU27++ CONV with IC <16 to be compressed CONV. This is known by NN as CM_CONV
    ///
    /// @param workload [in, out] that will be changed in case the input is presumed compressed CONV
    void compressConv_replace_by_CM_CONV_VPU27(DPUWorkload& workload) const;

public:
    /// @brief Turns OFF the swizzling
    ///
    /// @param workload [in, out] that will be changed in case the conditions are met
    void swizzling_turn_OFF(DPUWorkload& workload) const;

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
                          const std::string& shave_cache_filename, bool tryToLoadPairedCache = false);

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
                          const std::string& shave_cache_filename = "", bool tryToLoadPairedCache = false);

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
                          const char* shave_cache_data = nullptr, size_t shave_cache_data_length = 0);

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
    bool sanitize_workload(DPUWorkload& workload, SanityReport& result) const;

protected:
    /*
     * @brief Determines if a workload should use linear extrapolation for cost estimation.
     *
     * This function checks if the workload meets the criteria for linear extrapolation:
     * - The operation must be an intratile-like operation (AVEPOOL, DW_CONVOLUTION, or MAXPOOL)
     * - The output channels must exceed 64
     * - The NN preprocessing must support the property "DW_MXP_AVP_SupportsMoreThan64Ch"
     *
     * When these conditions are met, the cost is extrapolated linearly using reference points
     * at 32 and 64 channels, as these operations scale linearly with channel count.
     *
     * @param wl The DPU workload to check
     * @return true if linear extrapolation should be used, false otherwise
     */
    bool is_linearly_extrapolation_necessary(const DPUWorkload& wl) const;

    /**
     * @brief Compute runtime when both sparsity (input and weight) are active.
     * runtime will be a combined value (now is minimum) of the pair obtained by activating only one sparsity.
     *
     * @param workload is the workload to be inferred
     * @return the runtime or error
     */
    CyclesInterfaceType get_cost_dualsparsity(const DPUWorkload& workload) const;

    /**
     * @brief This method is used specifically for intra-tile like operations (AVEPOOL, DW_CONVOLUTION, MAXPOOL)
     * when the channel count exceeds 64. These operations scale linearly with channel count, allowing
     * reliable extrapolation from known reference points.
     *
     * ## How It Works:
     * The method uses two reference points to establish a linear relationship:
     * - Reference point 1: Cost at 32 channels → f(32) = cost_32ch
     * - Reference point 2: Cost at 64 channels → f(64) = cost_64ch
     *
     * From these points, we derive the linear function: f(ch) = slope × ch + intercept
     * Where:
     * - slope = (cost_64ch - cost_32ch) / (64 - 32) = (cost_64ch - cost_32ch) / 32
     * - intercept = cost_32ch - slope × 32
     *
     *If one or both reference workloads fail to get a cost from the cost providers, then:
     *    - If cost_32ch has an error → returned immediately (first check)
     *    - If cost_64ch has an error (but cost_32ch succeeded) → 64ch error returned (second check)
     *
     * @param workload The DPU workload with channel count > 64
     * @param info [out] String to accumulate error/warning information
     * @param cost_source [out] Optional pointer to receive the source identifier of the cost
     *
     * @return CyclesInterfaceType The extrapolated cost in cycles, or an error code if:
     */
    CyclesInterfaceType get_cost_linearly_extrapolated(const DPUWorkload& workload) const;

    /**
     * @brief Wrapper over run_cost_providers for handling situations where a workload cannot be resolved with only one
     * inference. Now it is handling the dual input sparsity (activation and weights): compute the runtime using the
     * algorithm @sa get_cost_dualsparsity() for more explanations.
     *
     * @param workload is the workload to be inferred
     * @return the runtime or error
     */
    CyclesInterfaceType get_cost(const DPUWorkload& workload, std::string& info,
                                 std::string* cost_source = nullptr) const;

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
                                           std::string* cost_source = nullptr) const;

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
    DPUWorkload cloneDeactivateActSparsity(const DPUWorkload& wl) const;

    /**
     * @brief Deactivates weight sparsity for a workload
     *
     * @param wl is the workload
     * @return a clone of the original wl with deactivated sparsity
     */
    DPUWorkload cloneDeactivateWeightSParsity(const DPUWorkload& wl) const;

    DPUWorkload cloneAndChangeInOutChannels(const DPUWorkload& wl_, const unsigned int channels) const;

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
    const std::vector<CyclesInterfaceType> get_cost(const std::vector<DPUWorkload>& workloads) const;

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

    std::vector<float> getDescriptor(const DPUWorkload& wl) const;

    /// @brief same like  @see DPU(DPUWorkload wl) , the extra param is to have as output the textual errors/findings
    /// discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param li will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    std::tuple<CyclesInterfaceType, std::string> DPUMsg(DPUWorkload wl) const;

    /// @brief same like  @see DPU(DPUWorkload wl) , the extra param is to have as output the textual errors/findings
    /// discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param info [out] will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    CyclesInterfaceType DPU(DPUWorkload wl, std::string& info) const;

protected:
    /* @brief DPU + wl param is also output as sanitized one.
     *  Provides workload outside so we know on what(post sanitization) was done the inference
     */
    CyclesInterfaceType DPU_and_sanitize(DPUWorkload& wl, std::string& info) const;

public:
    /**
     * @brief Return the number of cycles needed to compute multiple workloads
     *
     * @param workloads a std::vector of DPUWorkload
     * @return std::vector<CyclesInterfaceType> the DPUWorklaods execution cycles, @sa DPU for single wl for more
     * explanations
     */
    std::vector<CyclesInterfaceType> DPU(std::vector<DPUWorkload> workloads) const;

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
                     MemoryLocation output_location = MemoryLocation::CMX, unsigned int output_write_tiles = 1) const;

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     * @deprecated Will be removed in future releases
     */
    unsigned int DMA(const DMAWorkload& wl) const;

    /**
     * @brief Return the number of cycles needed to compute a Shave kernel
     *
     * @param shave_wl a Shave workload contains: name of kernel, device, in out tensor. PLus optional parameters of the
     * operations
     * @param infoOut  a string that will contain informative error information (in case of error)
     * @return the number of cycles of the Shave kernel, in DPU cycles of the desired device nominal frequency. OR ERROR
     */
    CyclesInterfaceType SHAVE(const SHAVEWorkload& shave_wl, std::string& infoOut) const;

    CyclesInterfaceType SHAVE(const SHAVEWorkload& shave_wl) const;

    /**
     * @brief Return the number of cycles needed to compute a Shave kernel without posibility of skipping Cache
     *
     * @param shave_wl a Shave workload contains: name of kernel, device, in out tensor. PLus optional parameters of the
     * operations
     * @param infoOut  a string that will contain informative error information (in case of error)
     * @return the number of cycles of the Shave kernel, in DPU cycles of the desired device nominal frequency. OR ERROR
     */
    CyclesInterfaceType SHAVE(const SHAVEWorkload& shave_wl, std::string& infoOut, bool skipCacheValues) const;

    /// gets the list of names of supported operators on a specified device. Each device has own operators
    ///
    /// @param device  for what device?
    /// @returns container with the name of operators
    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const;

    /// provides a reference to the operator executor specified by name
    /// The executor can be used to execute (run) the runtime prediction on different tensors/parameters
    /// Or can be asked to print information about its implementation parameters
    ///
    /// @param name of the operator
    /// @param device name
    /// @returns a ref (no ownership transfered. exists as long as this VPUCostModel instance exists)
    const ShaveOpExecutor& getShaveInstance(std::string name, VPUDevice device) const;

public:
    /**
     * @brief Compute the energy of a DPUWorkload.
     * @details This is a relative energy metric with a time base in DPU clock cyles. Energy of
     * 1000 would mean energy of worst case power for 1000 DPU clock cycles at reference dynamic power (power virus
     * for INT08 operations). measured in PowerVirusJoules = PowerVirus*cycle
     * @param wl a DPUWorkload
     * @return float the DPUWorkload energy, measured  PowerVirus*cycle
     */
    float DPUEnergy(const DPUWorkload& wl) const;

public:
    /** @brief Compute the energy of a SHAVE SHAVEWorkload.
     * @details Energy here is a relative metric, but the activity factor of the operation multiplied by
     *          its cost (number of clock cycles). We assume a constant activity factor of 0.5 for all and a max
     *          power of 5% of the DPU max power.
     *
     * @param swl a SHAVEWorkload
     * @return float the operation energy, in units relative to DPU PowerVirus. WIl return zero in case of error
     */
    float SHAVEEnergy(const SHAVEWorkload& swl) const;

    /// @brief same like  @see DPU(DPUWorkload wl) but return a Pack of information regarding the workload
    /// The purpose of this Method is to replace several separate calls to individual informations about the same
    /// workload.
    /// For example , estimated  cycle-times, errors, energy, activity factor, all can be obtained in one call.
    /// This method has the potential to be more efficient that the collection of individual ones.
    /// @param wl the workload to infer on
    /// @returns a Structure with all info that L1 APi can provide about this Workload
    DPUInfoPack DPUInfo(const DPUWorkload& workload) const;

    /// @brief Indicates whether the legacy SHAVE2 API is being used.
    /// Currently always returns false because this build targets the older SHAVE API.
    bool isShave2ApiUsed() const {
        return false;
    }

public:
    const AccessCounter& getPreloadedCacheCounter() const {
        return dpu_nn_cost_provider.getPreloadedCacheCounter();
    }

    const AccessCounter& getPreloadedShaveCacheCounter() const {
        return internal_shave_cost_model.getPreloadedCacheCounter();
    }
};  // class
}  // namespace VPUNN

#endif  // VPU_COST_MODEL_H
