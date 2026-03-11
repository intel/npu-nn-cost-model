// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu_dma_cost_model.h"

#include <algorithm>
#include <cmath>     // for std::floor
#include <memory>    // for std::make_shared, std::shared_ptr, std::dynamic_pointer_cast
#include <optional>  // for std::optional
#include <string>    // for std::string
#include <tuple>     // for std::tuple, std::make_tuple
#include "core/cache.h"
#include "core/dma_map_type_selector.h"  // need this to instantiate the template with specifics like DMANNWorkload_NPU27
#include "core/logger.h"
#include "core/serializer.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dma_cost_providers/dma_cost_provider_bundles.h"
#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"
#include "vpu/dma_cost_providers/dmann_cost_provider.h"
#include "vpu/dma_cost_providers/priority_dma_cost_provider.h"
#include "vpu/dma_types.h"
#include "vpu/dma_workload.h"
#include "vpu/serialization/dma_cost_serialization_wrapper.h"
#include "vpu/types.h"
#include "vpu/validation/checker_utils.h"

namespace VPUNN {

// DMATheoreticalCostModel implementations
DMATheoreticalCostModel::DMATheoreticalCostModel() {
    Logger::initialize();
}

unsigned int DMATheoreticalCostModel::DMA(VPUDevice device, const VPUTensor& input, const VPUTensor& output,
                                          MemoryLocation input_location, MemoryLocation output_location,
                                          unsigned int output_write_tiles) const {
    return dma_theoretical.get_cost({device, input, output, input_location, output_location, output_write_tiles});
}

unsigned int DMATheoreticalCostModel::DMA(const DMAWorkload& wl) const {
    return dma_theoretical.get_cost(wl);
}

// Template class implementations
template <class DMADesc>
DMACostModel<DMADesc>::DMACostModel(const std::string& filename, bool profile, const unsigned int cache_size,
                                    const unsigned int batch_size, const std::string& cache_filename)
        : ptr_internal_dma_cost_provider(std::make_shared<PriorityDMACostProvider<DMADesc>>(
                  DMACostProviderBundles::createDefaultDMACostProviders<DMADesc>(filename, batch_size, profile))),
          cache(cache_size, cache_filename),
          http_dma_cost_provider(HttpCostProviderFactory::create()) {
        Logger::initialize();
        interogation_serializer.initialize(DescType::get_wl_name(), FileMode::READ_WRITE, DMANNCostProvider<DMADesc>::get_names_for_serializer());
}

template <class DMADesc>
DMACostModel<DMADesc>::DMACostModel(const char* model_data, size_t model_data_length, bool copy_model_data,
                                     bool profile, const unsigned int cache_size, const unsigned int batch_size,
                                     const char* cache_data, size_t cache_data_length)
       : ptr_internal_dma_cost_provider(std::make_shared<PriorityDMACostProvider<DMADesc>>(
                  DMACostProviderBundles::createDefaultDMACostProviders<DMADesc>(
                      model_data, model_data_length, batch_size, copy_model_data, profile))),
         cache(cache_size, cache_data, cache_data_length), 
         http_dma_cost_provider(HttpCostProviderFactory::create()) {
        Logger::initialize();
        interogation_serializer.initialize(DescType::get_wl_name(), FileMode::READ_WRITE, DMANNCostProvider<DMADesc>::get_names_for_serializer());
}

template <class DMADesc>
bool DMACostModel<DMADesc>::nn_initialized() const {
    auto priority_provider =
            std::dynamic_pointer_cast<const PriorityDMACostProvider<DMADesc>>(ptr_internal_dma_cost_provider);
    if (priority_provider) {
        return priority_provider->has_nn_initialized();
    }
    return false;
}

template <class DMADesc>
const AccessCounter& DMACostModel<DMADesc>::getPreloadedCacheCounter() const {
    return cache.getPreloadedCacheCounter();
}

template <class DMADesc>
bool DMACostModel<DMADesc>::sanitize_workload(DMADesc&, SanityReport& result) const {
    return result.is_usable();
}

template <class DMADesc>
CyclesInterfaceType DMACostModel<DMADesc>::computeCycles(const DMADesc& wl) {
    std::string dummy_info{};
    return computeCycles(wl, dummy_info);
}

template <class DMADesc>
std::tuple<CyclesInterfaceType, std::string> DMACostModel<DMADesc>::computeCyclesMsg(DMADesc wl) {
    std::string dummy_info{};
    auto previous_print_mode{Checker::set_print_tags(false)};
    const auto r{computeCycles(wl, dummy_info)};
    Checker::set_print_tags(previous_print_mode);
    return std::make_tuple(r, dummy_info);
}

template <class DMADesc>
CyclesInterfaceType DMACostModel<DMADesc>::computeCycles(const DMADesc& wl, std::string& info) {
    return Execute_and_sanitize(wl, info);
}

template <class DMADesc>
CyclesInterfaceType DMACostModel<DMADesc>::Execute_and_sanitize(const DMADesc& wl, std::string& info) {
    DMACostSerializationWrap<DMADesc> serialization_handler(interogation_serializer);
    serialization_handler.serializeDMAWorkload(wl);

    SanityReport problems{};
    info = problems.info;

    std::string cost_source = "unknown";
    CyclesInterfaceType cycles{problems.value()};

    cycles = get_cost(wl, info, &cost_source);

    serialization_handler.serializeCyclesAndCostInfo_closeLine(cycles, std::move(cost_source), info);

    return cycles;
}

template <class DMADesc>
CyclesInterfaceType DMACostModel<DMADesc>::get_cost(const DMADesc& workload, std::string& info,
                                                    std::string* cost_source) const {
    return run_cost_providers(workload, info, cost_source);
}

template <class DMADesc>
CyclesInterfaceType DMACostModel<DMADesc>::run_cost_providers(const DMADesc& workload, std::string& info,
                                                               std::string* cost_source) const {
    auto cycles{Cycles::NO_ERROR};

        const auto try_cache = [&]() -> CyclesInterfaceType {
            const auto cached_cost = cache.get(workload, cost_source);
            if(cached_cost) {
                return static_cast<CyclesInterfaceType>(std::floor(*cached_cost));
            }
            else {
                return Cycles::ERROR_CACHE_MISS;
            }
        };

        const auto try_profiling = [&]() -> CyclesInterfaceType {
            // Check if the provided DMA Descriptor is of NN type
            if constexpr (std::is_same_v<DMADesc, DMAWorkload>) {
                // Profiling service is not applicable for DMAWorkload
                return Cycles::ERROR_PROFILING_SERVICE;
            } else {
                if (http_dma_cost_provider) {
                    if (cost_source) {
                        *cost_source = "profiling_service_" + http_dma_cost_provider->profilingBackendToString(workload.profiling_service_backend_hint);
                    }
                    return http_dma_cost_provider->getCost(workload, info);
                } else {
                    return Cycles::ERROR_PROFILING_SERVICE;
                }
            }
        };

        const auto try_priority_provider = [&]() -> CyclesInterfaceType {
            info = "";  // Avoid unreferenced var warning
            // Use priority-based provider (NN with fallback to theoretical)
            cycles = dma_cost_provider.get_cost(workload, cost_source);

            // Add valid cycles to cache for future lookups
            if (!Cycles::isErrorCode(cycles)) {
                cache.add(workload, static_cast<float>(cycles));
            }
            return cycles;
        };

        // 1. Cache lookup
        const auto cached_cost = try_cache();
        if(!Cycles::isErrorCode(cached_cost)) {
            return cached_cost;
        }

        // 2. Profiling service
        cycles = try_profiling();

        // 3. Priority-based provider (fallback if profiling fails)
        if (Cycles::isErrorCode(cycles)) {
            return try_priority_provider();
        }

        return cycles;
}

// Explicit template instantiations for known DMANN workload types
template class DMACostModel<DMANNWorkload_NPU27>;
template class DMACostModel<DMANNWorkload_NPU40_50>;

}  // namespace VPUNN
