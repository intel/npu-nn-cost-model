// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu_shave_cost_model.h"

#include <cmath>
#include <memory>  // for std::shared_ptr, std::move
#include <string>  // for std::string
#include "vpu/cycles_interface_types.h"
#include "vpu/shave/shave_cost_providers/shave_provider_bundles.h"

namespace VPUNN {

std::shared_ptr<IShaveCostProvider> SHAVECostModel::createDefaultCostProvider() {
    return  // select by commenting or un-commenting the desired provider
            ShaveCostProviderBundles::createDeviceMappedProvider()  // for updated approach
            // ShaveCostProviderBundles::createOldShaveOnlyProvider()  // activate this for legacy non heuristic
            // behaviors
            ;
}

SHAVECostModel::SHAVECostModel(const std::string& cache_filename, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(createDefaultCostProvider()), cache(cache_size, cache_filename),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

SHAVECostModel::SHAVECostModel(const char* cache_data, size_t cache_data_length, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(createDefaultCostProvider()),
          cache(cache_size, cache_data, cache_data_length),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

SHAVECostModel::SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider,
                               const std::string& cache_filename, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(std::move(external_shave_cost_provider)), cache(cache_size, cache_filename),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

SHAVECostModel::SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider, const char* cache_data,
                               size_t cache_data_length, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(std::move(external_shave_cost_provider)),
          cache(cache_size, cache_data, cache_data_length),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

CyclesInterfaceType SHAVECostModel::computeCycles(const SHAVEWorkload& swl, [[maybe_unused]] std::string& infoOut) const {
    // finds func inmpl, executes it, handles errors
    SHAVECostSerializationWrap serialization_handler(serializer);
    std::string apiUsed{"unknown"};

    CyclesInterfaceType cycles{Cycles::NO_ERROR};  // Initialize with a default error value

    const auto try_cache = [&]() -> CyclesInterfaceType {
        const auto cachedData{cache.get(swl, &apiUsed)};
        if (cachedData) {
            cycles = static_cast<CyclesInterfaceType>(std::floor(*cachedData));
            return cycles;
        }

        return Cycles::ERROR_CACHE_MISS;  // if not found in cache, we return a cache miss error code
    };
    
    const auto try_profiling = [&]() -> CyclesInterfaceType {
        // now search with the http provider 
        if(http_cost_provider) {
            apiUsed = "profiling_service_" + http_cost_provider->profilingBackendToString(swl.get_profiling_service_backend());
            
            auto http_cost = http_cost_provider->getCost(swl, infoOut);
            if(!Cycles::isErrorCode(http_cost)) {
                return http_cost;
            }
        }

        return Cycles::ERROR_PROFILING_SERVICE;  // if no provider or if provider returns error, we return an error code
    };
    
    // First search in the cache for a pre-computed cost. If found, return it immediately.
    cycles = try_cache();

    // Second, if not found in cache, try to get the cost from the profiling service.
    if (Cycles::isErrorCode(cycles)) {
        cycles = try_profiling();
    }

    // Finally, if still not found, compute the cost using the SHAVE cost provider.
    if (Cycles::isErrorCode(cycles)) {
        cycles = shave_cost_provider.get_cost(swl, &apiUsed);
    }

    // Add the computed cost to the cache for future reuse
    if (cycles < Cycles::START_ERROR_RANGE) {
        cache.add(swl, static_cast<float>(cycles));
    }

    serialization_handler.serializeShaveWorkloadWithCycles(swl, apiUsed, cycles);
    return cycles;
}

CyclesInterfaceType SHAVECostModel::computeCycles(const SHAVEWorkload& swl) const {
    std::string infoOut;
    return computeCycles(swl, infoOut);  // do not skip cache
}

bool SHAVECostModel::sanitize_workload(const SHAVEWorkload& swl, SanityReport& result) const {
    sanitizer.check_and_sanitize(swl, result);

    if (!result.is_usable()) {
        return false;
    }
    return true;
}

}  // namespace VPUNN
