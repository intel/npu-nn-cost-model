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

#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"
#include <vpu/shave/shave_cost_providers/shave_cost_provider_interface.h>
#include <vpu/shave/shave_cost_providers/shave_cost_providers.h>
#include <vpu/shave/shave_cost_providers/priority_shave_cost_provider.h>
#include <vpu/shave/shave_cost_providers/shave_provider_bundles.h>

#include "core/vpunn_api.h"

namespace VPUNN
{
/// @brief High-level cost model for estimating execution cycles of SHAVE workloads on VPU devices
/// Provides flexibility in the way of what cost provider to use, either mathematical, priority based or NN based in future
/// rule of three violation is false because they present but no effect (=default doesn't mean implemented)
/* coverity[rule_of_three_violation:FALSE] */
class VPUNN_API SHAVECostModel {
private:
    SHAVE_Workloads_Sanitizer sanitizer; ///< sanitizes the workload before processing
    mutable CSVSerializer serializer;    ///< serializes workloads to a CSV file
    bool use_shave_2_api;                ///< Flag indicating whether to use SHAVE 2 API instead of legacy SHAVE 1 API. 
                                         ///< This will be removed once the compiler takes ownership of shave cost model creation.

    std::shared_ptr<IShaveCostProvider> ptr_internal_shave_cost_provider;  ///< shared ownership of SHAVE cost provider
                                                                           ///< Currently is shared in case that in future we decide
                                                                           ///< to have an internal factory that will create the SHAVECM
                                                                           ///< it has to be revisited
    IShaveCostProvider& shave_cost_provider{
        *ptr_internal_shave_cost_provider
    };       ///< provides cycles

    mutable LRUCache<SHAVEWorkload, float> cache;  ///< all devices cache/LUT for shave ops. Populated in ctor
                                                   ///< this is a preloaded cache that features also a dynamic one
                                                   ///< and it is populated based on the new API entries only

public:

    explicit SHAVECostModel(const std::string& cache_filename = "", const unsigned int cache_size = 16384, const bool use_shave_2_api = false)
            : use_shave_2_api(use_shave_2_api),
              ptr_internal_shave_cost_provider(std::make_shared<PriorityShaveCostProvider>(use_shave_2_api ? 
                                                ShaveCostProviderBundles::createDefaultShaveCostProviders() : 
                                                ShaveCostProviderBundles::createOldShaveOnlyProviders())),
              cache(cache_size, cache_filename) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
    }

    explicit SHAVECostModel(const char* cache_data, size_t cache_data_length, const unsigned int cache_size = 16384, 
                        bool use_shave_2_api = false)
            : use_shave_2_api(use_shave_2_api),
              ptr_internal_shave_cost_provider(std::make_shared<PriorityShaveCostProvider>(use_shave_2_api ? 
                                                ShaveCostProviderBundles::createDefaultShaveCostProviders() : 
                                                ShaveCostProviderBundles::createOldShaveOnlyProviders())),
              cache(cache_size, cache_data, cache_data_length) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
    }

protected:
    // Constructor with IShaveCostProvider externally provided - to be used in future when we are going to feature a CostModel based on an internal factory 
    explicit SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider, const std::string& cache_filename = "", const unsigned int cache_size = 16384) 
            : use_shave_2_api(false), /*to be discussed if we want to do this now or comment the constructors until next iteration*/ 
              ptr_internal_shave_cost_provider(std::move(external_shave_cost_provider)),
              cache(cache_size, cache_filename) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
    }

    // Constructor with IShaveCostProvider externally provided - to be used in future when we are going to feature a CostModel based on an internal factory 
    explicit SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider, const char* cache_data, size_t cache_data_length, const unsigned int cache_size = 16384) 
            : use_shave_2_api(false), /*to be discussed if we want to do this now or comment the constructors until next iteration*/
              ptr_internal_shave_cost_provider(external_shave_cost_provider),
              cache(cache_size, cache_data, cache_data_length) {
        serializer.initialize("shave_workloads", FileMode::READ_WRITE, ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
    }

public:
    SHAVECostModel(const SHAVECostModel&) = delete;
    virtual ~SHAVECostModel() = default;

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, [[maybe_unused]]std::string& infoOut, bool skipCacheSearch) const {
        // finds func inmpl, executes it, handles errors
        SHAVECostSerializationWrap serialization_handler(serializer);
        std::string apiUsed{"unknown"};
        
        CyclesInterfaceType cycles{Cycles::NO_ERROR};  // Initialize with a default error value

        if (!skipCacheSearch) {  // before finding the shave imnpl check if already in cache for this request
                                 // This is a one cache for all
            const auto cachedData{cache.get(swl, &apiUsed)};
            if (cachedData) {
                cycles = static_cast<CyclesInterfaceType>(std::floor(*cachedData));
                serialization_handler.serializeShaveWorkloadWithCycles(swl, apiUsed, cycles);
                return cycles;
            }
        }

        cycles = shave_cost_provider.get_cost(swl, &apiUsed); 
        serialization_handler.serializeShaveWorkloadWithCycles(swl, apiUsed, cycles);
        return cycles;
    }

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl, std::string& infoOut) const {
        return computeCycles(swl, infoOut, false);  // do not skip cache
    }

    CyclesInterfaceType computeCycles(const SHAVEWorkload& swl) const {
        std::string infoOut;
        return computeCycles(swl, infoOut, false);  // do not skip cache
    }

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
    bool sanitize_workload(const SHAVEWorkload& swl, SanityReport& result) const {
        sanitizer.check_and_sanitize(swl, result);

        if (!result.is_usable()) {
            return false;
        }
        return true;
    }

public:
    std::vector<std::string> getShaveSupportedOperations(VPUDevice device) const {
       return shave_cost_provider.get_shave_supported_ops(device);
    };

    std::optional<std::reference_wrapper<const ShaveOpExecutor>> getShaveInstance(std::string name, VPUDevice device) const {
        return shave_cost_provider.get_shave_instance(name, device);
    }

    /**
     * @brief Checks if the SHAVE Gen 2 API is being used.
     * @return True if the SHAVE Gen 2 API is used, otherwise false.
     */
    bool isShave2APIused() const {
        return use_shave_2_api;
    }
};

} // namespace VPUNN


#endif  // VPU_SHAVE_COST_MODEL_H
