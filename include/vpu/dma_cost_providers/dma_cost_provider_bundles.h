// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_COST_PROVIDER_BUNDLES_H
#define DMA_COST_PROVIDER_BUNDLES_H

#include "dma_cost_provider_interface.h"
#include "dmann_cost_provider.h"
#include "dma_theoretical_cost_provider.h"
#include "priority_dma_cost_provider.h"
#include <vector>
#include <memory>

namespace VPUNN {

/**
 * @brief Utility class for creating precomposed DMA cost provider bundles
 * 
 * This class provides static factory methods for creating different combinations
 * of DMA cost providers. DMA providers use adapters for workload type conversion.
 * The adapters store shared_ptr to the underlying providers for proper lifetime management.
 */
class DMACostProviderBundles {
public:
    /**
     * @brief Create a prebuilt list of DMA cost providers with default priority order
     * 
     * Creates a priority list with:
     * - Priority 0: DMANNCostProvider (NN-based cost model)
     * - Priority 1: DMATheoreticalCostProvider (fallback to theoretical model)
     * 
     * The adapter stores shared_ptr to the underlying providers for automatic lifetime management.
     * 
     * @tparam DMADesc The DMA descriptor type (e.g., DMANNWorkload_NPU27, DMANNWorkload_NPU40_50)
     * @param filename Model filename for NN provider
     * @param batch_size Batch size for NN inference
     * @param profile Enable profiling
     * @return DMACostProviderList The prebuilt provider list
     */
    template <typename DMADesc>
    static DMACostProviderList<DMADesc> createDefaultDMACostProviders(
            const std::string& filename = "", 
            const unsigned int batch_size = 1, 
            bool profile = false) {

        DMACostProviderList<DMADesc> providers;
        providers.push_back(std::make_shared<DMANNAdapter<DMADesc>>(
            std::shared_ptr<const IDMACostProvider<DMADesc>>(
                std::make_shared<DMANNCostProvider<DMADesc>>(filename, batch_size, profile, 0, "", false)
            )
        ));
        providers.push_back(std::make_shared<DMANNAdapter<DMADesc>>(
            std::shared_ptr<const IDMACostProvider<DMAWorkload>>(
                std::make_shared<DMATheoreticalCostProvider>()
            )
        ));
        
        return providers;
    }

    /**
     * @brief Create a prebuilt list of DMA cost providers with default priority order (buffer-based)
     * 
     * Creates a priority list with:
     * - Priority 0: DMANNCostProvider (NN-based cost model from buffer)
     * - Priority 1: DMATheoreticalCostProvider (fallback to theoretical model)
     * 
     * The adapter stores shared_ptr to the underlying providers for automatic lifetime management.
     * 
     * @tparam DMADesc The DMA descriptor type
     * @param model_data Buffer containing model data
     * @param model_data_length Size of model data buffer
     * @param batch_size Batch size for NN inference
     * @param copy_model_data Whether to copy model data
     * @param profile Enable profiling
     * @return DMACostProviderList The prebuilt provider list
     */
    template <typename DMADesc>
    static DMACostProviderList<DMADesc> createDefaultDMACostProviders(
            const char* model_data,
            size_t model_data_length,
            const unsigned int batch_size = 1,
            bool copy_model_data = true,
            bool profile = false) {
        
        DMACostProviderList<DMADesc> providers;
        providers.push_back(std::make_shared<DMANNAdapter<DMADesc>>(
            std::shared_ptr<const IDMACostProvider<DMADesc>>(
                std::make_shared<DMANNCostProvider<DMADesc>>(model_data, model_data_length, batch_size, copy_model_data, profile)
            )
        ));
        providers.push_back(std::make_shared<DMANNAdapter<DMADesc>>(
            std::shared_ptr<const IDMACostProvider<DMAWorkload>>(
                std::make_shared<DMATheoreticalCostProvider>()
            )
        ));
        
        return providers;
    }    
    /**
     * @brief Create a bundle with only the NN cost provider
     * 
     * @tparam DMADesc The DMA descriptor type
     * @param filename Model filename for NN provider
     * @param batch_size Batch size for NN inference
     * @param profile Enable profiling
     * @return DMACostProviderList Provider list with only NN provider
     */
    template <typename DMADesc>
    static DMACostProviderList<DMADesc> createNNOnlyProviders(
            const std::string& filename = "", 
            const unsigned int batch_size = 1, 
            bool profile = false) {

        DMACostProviderList<DMADesc> providers;
        providers.push_back(std::make_shared<DMANNAdapter<DMADesc>>(
            std::shared_ptr<const IDMACostProvider<DMADesc>>(
                std::make_shared<DMANNCostProvider<DMADesc>>(filename, batch_size, profile, 0, "", false)
            )
        ));
        
        return providers;
    }

    /**
     * @brief Create a bundle with only the theoretical cost provider
     * 
     * @tparam DMADesc The DMA descriptor type
     * @return DMACostProviderList Provider list with only theoretical provider
     */
    template <typename DMADesc>
    static DMACostProviderList<DMADesc> createTheoreticalOnlyProviders() {
        
        DMACostProviderList<DMADesc> providers;
        providers.push_back(std::make_shared<DMANNAdapter<DMADesc>>(
            std::shared_ptr<const IDMACostProvider<DMAWorkload>>(
                std::make_shared<DMATheoreticalCostProvider>()
            )
        ));
        
        return providers;
    }
};

}  // namespace VPUNN

#endif  // DMA_COST_PROVIDER_BUNDLES_H
