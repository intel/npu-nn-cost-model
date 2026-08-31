// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPU_SHAVE_COST_MODEL_H
#define VPU_SHAVE_COST_MODEL_H

#include <functional>  // for std::reference_wrapper
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/cache.h"
#include "core/serializer.h"
#include "core/shave_map_type_selector.h"
#include "core/vpunn_api.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/http_cost_provider_factory.h"
#include "vpu/http_cost_provider_intf.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_cost_providers/shave_cost_provider_interface.h"
#include "vpu/shave/shave_prefetch_cost_strategy.h"
#include "vpu/validation/shave_workloads_sanitizer.h"

namespace VPUNN {
/// @brief High-level cost model for estimating execution cycles of SHAVE workloads on VPU devices
/// Provides flexibility in the way of what cost provider to use, either mathematical, priority based or NN based in
/// future rule of three violation is false because they present but no effect (=default doesn't mean implemented)
/* coverity[rule_of_three_violation:FALSE] */
class VPUNN_API SHAVECostModel {
private:
    SHAVE_Workloads_Sanitizer sanitizer;  ///< sanitizes the workload before processing
    mutable CSVSerializer serializer;     ///< serializes workloads to a CSV file

    std::shared_ptr<IShaveCostProvider>
            ptr_internal_shave_cost_provider;  ///< shared ownership of SHAVE cost provider
                                               ///< Currently is shared in case that in future we decide
                                               ///< to have an internal factory that will create the SHAVECM
                                               ///< it has to be revisited
    IShaveCostProvider& shave_cost_provider{*ptr_internal_shave_cost_provider};  ///< provides cycles

    std::shared_ptr<const IPrefetchCostStrategy> prefetch_cost_strategy;  ///< standalone prefetch lookup infrastructure

    mutable LRUCache<SHAVEWorkload, float> cache;  ///< all devices cache/LUT for shave ops. Populated in ctor
                                                   ///< this is a preloaded cache that features also a dynamic one
    const std::unique_ptr<const IHttpCostProvider> http_cost_provider;  ///< optional HTTP cost provider for remote cost
                                                                        ///< estimation (e.g., via a profiling service)

    /// @brief Gate for whether a cache-miss SHAVE workload query may use the online profiling service. Defaults
    /// from the environment; runtime overrides apply only when the environment did not force a value.
    ProfilingAutoHintGate profilingAutoHintGate;

    /**
     * @brief Creates the default SHAVE cost provider
     *
     * This static method centralizes the logic for creating the default cost provider.
     * Currently returns the device-mapped provider, but can be easily modified to
     * switch to old SHAVE-only provider or other implementations.
     *
     * @return std::shared_ptr<IShaveCostProvider> The created cost provider
     */
    static std::shared_ptr<IShaveCostProvider> createDefaultCostProvider();

    /// @brief Creates the default prefetch strategy.
    static std::shared_ptr<const IPrefetchCostStrategy> createDefaultPrefetchCostStrategy();

public:
    explicit SHAVECostModel(const std::string& cache_filename = "", const unsigned int cache_size = 16384);

    explicit SHAVECostModel(const char* cache_data, size_t cache_data_length, const unsigned int cache_size = 16384);

protected:
    // Constructor with IShaveCostProvider externally provided - to be used in future when we are going to feature a
    // CostModel based on an internal factory
    explicit SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider,
                            const std::string& cache_filename = "", const unsigned int cache_size = 16384);

    // Constructor with IShaveCostProvider externally provided - to be used in future when we are going to feature a
    // CostModel based on an internal factory
    explicit SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider, const char* cache_data,
                            size_t cache_data_length, const unsigned int cache_size = 16384);

public:
    SHAVECostModel(const SHAVECostModel&) = delete;
    virtual ~SHAVECostModel() = default;

    /**
     * @brief Estimate the execution cycles for a SHAVE workload.
     *
     * Resolution order: preloaded fixed cache → profiling service → mathematical/NN provider.
     * The first successful source wins and is cached in the dynamic LRU cache for future calls.
     *
     * @param swl the workload to evaluate; set swl.include_code_prefetch=true to model a cold invocation
     * @param infoOut optional diagnostic string filled by the model (e.g. which source was used)
     * @return estimated cycles, or an error code from Cycles if the operator cannot be resolved
     */
    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, [[maybe_unused]] std::string& infoOut) const;

    /**
     * @brief Convenience overload of computeCycles() without diagnostic output.
     * @see computeCycles(const SHAVEWorkload&, std::string&)
     */
    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl) const;

    const AccessCounter& getPreloadedCacheCounter() const {
        return cache.getPreloadedCacheCounter();
    }

    /// @brief Enables/disables the online profiling service for cache-miss queries, unless the environment
    /// forced a value (which takes precedence).
    void setProfilingEnabledForAutoHint(bool value) noexcept {
        profilingAutoHintGate.set(value);
    }

    bool isProfilingEnabledForAutoHint() const noexcept {
        return profilingAutoHintGate.isEnabled();
    }

protected:
    /**
    @brief Sanitizes the workload before processing

    @param swl the workload to sanitize
    @param result the report of the sanitization

    @return true if the workload is sanitized, false otherwise
    */
    bool sanitize_workload(const SHAVEWorkload& swl, SanityReport& result) const;

public:
    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const {
        return shave_cost_provider.get_shave_supported_ops(device);
    }

    /**
     * @brief Lightweight capability query that does not require SHAVECostModel construction.
     *
     * Returns supported SHAVE operation names for the requested device using the
     * default device-mapped provider composition.
     *
     * @param device Device to query
     * @return std::vector<std::string> Supported operation names
     */
    static std::vector<std::string> queryDeviceMappedSupportedOperations(VPUDevice device);

    std::optional<std::reference_wrapper<const ShaveOpExecutor>> getShaveInstance(std::string name,
                                                                                  VPUDevice device) const {
        return shave_cost_provider.get_shave_instance(name, device);
    }

    /**
     * @brief Returns the cold-start kernel code-prefetch cost in cycles for a named operator on a given device.
     *
     * This is the cost of loading the SHAVE kernel binary from DDR into CMX, modelling the one-time
     * overhead incurred on the first invocation of the operator on hardware.
     *
     * - This method is a pure, stateless query: it always returns the configured prefetch cost for the
     *   operator regardless of how many times the operator has been invoked via computeCycles(). The model
     *   does not perform any first-use tracking internally.
     *
     * - The prefetch term is applied AFTER cache lookup and is NOT stored in the cache.
     *   A cache hit for a workload with include_code_prefetch=true will therefore return
     *   cached_base_cycles + prefetch_cost, not a cached total that already includes prefetch.
     *   This guarantees that the same base result can serve both warm and cold estimates.
     *
     * - Cache entries and profiling-service results are assumed to represent warm (post-prefetch)
     *   execution cost only. If a cache entry or profiling backend returns cold-invocation cost
     *   (i.e. already including prefetch), enabling include_code_prefetch will double-count the
     *   prefetch term.
     *
     * @param swl workload carrying the operator name and target device
     * @return the cold-start prefetch cost in cycles based on the strategy configured at construction.
     *         The default strategy returns the generated-table value when available.
     *         If the device is covered but the operator is missing, it returns Cycles::ERROR_SHAVE_PREFETCH_NOT_FOUND.
     *         If the device is not covered, it returns Cycles::ERROR_SHAVE_PREFETCH_DEVICE_NOT_COVERED (and
     *         computeCycles() will propagate this error when include_code_prefetch=true).
     */
    CyclesInterfaceType getCodePrefetchCost(const SHAVEWorkload& swl) const {
        return prefetch_cost_strategy->get_code_prefetch_cost(swl);
    }
};

}  // namespace VPUNN

#endif  // VPU_SHAVE_COST_MODEL_H
