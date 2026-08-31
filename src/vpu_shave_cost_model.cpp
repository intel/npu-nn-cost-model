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
#include "core/utils.h"  // for get_env_vars
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

std::shared_ptr<const IPrefetchCostStrategy> SHAVECostModel::createDefaultPrefetchCostStrategy() {
    static const auto default_prefetch_strategy = std::make_shared<DeviceMappedPrefetchCostStrategy>(
            ShaveCostProviderBundles::createDeviceMappedPrefetchStrategy());
    //static is only to call only once the createDeviceMappedPrefetchStrategy() 
    return default_prefetch_strategy;
}

SHAVECostModel::SHAVECostModel(const std::string& cache_filename, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(createDefaultCostProvider()),
          prefetch_cost_strategy(createDefaultPrefetchCostStrategy()),
          cache(cache_size, cache_filename),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

SHAVECostModel::SHAVECostModel(const char* cache_data, size_t cache_data_length, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(createDefaultCostProvider()),
          prefetch_cost_strategy(createDefaultPrefetchCostStrategy()),
          cache(cache_size, cache_data, cache_data_length),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

SHAVECostModel::SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider,
                               const std::string& cache_filename, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(std::move(external_shave_cost_provider)),
          prefetch_cost_strategy(createDefaultPrefetchCostStrategy()),
          cache(cache_size, cache_filename),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

SHAVECostModel::SHAVECostModel(std::shared_ptr<IShaveCostProvider> external_shave_cost_provider, const char* cache_data,
                               size_t cache_data_length, const unsigned int cache_size)
        : ptr_internal_shave_cost_provider(std::move(external_shave_cost_provider)),
          prefetch_cost_strategy(createDefaultPrefetchCostStrategy()),
          cache(cache_size, cache_data, cache_data_length),
          http_cost_provider(HttpCostProviderFactory::create()) {
    serializer.initialize(
            "shave_workloads", FileMode::READ_WRITE,
            ShaveSerializerUtils::get_names_for_shave_serializer(shave_cost_provider.get_max_num_params()));
}

CyclesInterfaceType SHAVECostModel::computeCycles(const SHAVEWorkload& swl,
                                                  [[maybe_unused]] std::string& infoOut) const {
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
        if (http_cost_provider) {
            apiUsed = "profiling_service_" +
                      http_cost_provider->profilingBackendToString(swl.get_profiling_service_backend());

            auto http_cost = http_cost_provider->getCost(swl, infoOut);
            if (!Cycles::isErrorCode(http_cost)) {
                return http_cost;
            }
        }

        return Cycles::ERROR_PROFILING_SERVICE;  // if no provider or if provider returns error, we return an error code
    };

    // First search in the cache for a pre-computed cost. If found, return it immediately.
    cycles = try_cache();

    // Second, if not found in cache and profiling is enabled, try to get the cost from the profiling service.
    if (Cycles::isErrorCode(cycles) && profilingAutoHintGate.isEnabled()) {
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

    // Preserve historical serialization semantics: CSV stores warm/runtime cost.
    // Prefetch metadata columns may be added in the future to disambiguate warm vs. cold rows.
    const auto cycles_for_serialization = cycles;

    // Serialize the warm/base cost to CSV before attempting prefetch lookup.
    // This ensures the warm cost is always recorded regardless of prefetch success/failure.
    serialization_handler.serializeShaveWorkloadWithCycles(swl, apiUsed, cycles_for_serialization);

    // Code-prefetch is an optional additive term modelling a cold (first-time) invocation.
    // It is applied after the cache-insert step so that:
    //  - cache entries always represent the warm base cost, and
    //  - cache hits and cache misses add the prefetch term identically.
    // ASSUMPTION: the base cycles value at this point is a warm execution cost that does
    // NOT already include code-prefetch overhead.
    //
    // The lookup is delegated to a dedicated prefetch strategy infrastructure.
    if (!Cycles::isErrorCode(cycles) && swl.include_code_prefetch) {
        const auto prefetch_cost = getCodePrefetchCost(swl);

        // If the prefetch cost lookup itself failed with an error, propagate that error.
        if (Cycles::isErrorCode(prefetch_cost)) {
            return prefetch_cost;
        }

        cycles = Cycles::cost_adder(cycles, prefetch_cost);
    }

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

std::vector<std::string> SHAVECostModel::queryDeviceMappedSupportedOperations(VPUDevice device) {
    return ShaveCostProviderBundles::queryDeviceMappedSupportedOperations(device);
}

}  // namespace VPUNN
