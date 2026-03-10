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

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <functional>  // for std::reference_wrapper

#include "vpu/shave/shave_cost_providers/shave_cost_provider_interface.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/validation/shave_workloads_sanitizer.h"
#include "core/cache.h"
#include "core/serializer.h"
#include "core/shave_map_type_selector.h"
#include "core/vpunn_api.h"
#include "vpu/cycles_interface_types.h"

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

    mutable LRUCache<SHAVEWorkload, float> cache;  ///< all devices cache/LUT for shave ops. Populated in ctor
                                                   ///< this is a preloaded cache that features also a dynamic one

    /**
     * @brief Creates the default SHAVE cost provider
     *
     * This static method centralizes the logic for creating the default cost provider.
     * Currently returns the old SHAVE-only provider, but can be easily modified to
     * switch to device-mapped provider or other implementations.
     *
     * @return std::shared_ptr<IShaveCostProvider> The created cost provider
     */
    static std::shared_ptr<IShaveCostProvider> createDefaultCostProvider();

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

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, [[maybe_unused]] std::string& infoOut,
                                      bool skipCacheSearch) const;

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut) const;

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl) const;

    const AccessCounter& getPreloadedCacheCounter() const {
        return cache.getPreloadedCacheCounter();
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

    std::optional<std::reference_wrapper<const ShaveOpExecutor>> getShaveInstance(std::string name,
                                                                                  VPUDevice device) const {
        return shave_cost_provider.get_shave_instance(name, device);
    }
};

}  // namespace VPUNN

#endif  // VPU_SHAVE_COST_MODEL_H
