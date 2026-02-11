// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#pragma once

#include "dma_cost_provider_interface.h"
#include <memory>
#include <type_traits>
#include <utility>
#include "vpu/dma_workload.h"
#include "dmann_cost_provider.h"

namespace VPUNN {

/// @brief Adapter that converts workload types for DMA cost providers
/// @tparam WlT The input workload type this adapter accepts
///
/// This adapter uses type erasure to wrap any IDMACostProvider that accepts
/// a different workload type (TargetWlT) and automatically converts WlT to TargetWlT.
/// This allows providers with different workload types to be used interchangeably.
template <typename WlT>
class DMANNAdapter : public IDMACostProvider<WlT> {
public:
     /// Deleted default constructor - requires explicit provider
    DMANNAdapter() = delete;
    
    /// @brief Construct adapter with a target provider
    /// @tparam TargetWlT The workload type accepted by the target provider (deduced)
    /// @param target The underlying cost provider to wrap and adapt (shared ownership)
    ///
    /// The adapter will convert WlT workloads to TargetWlT before forwarding
    /// to the target provider.
    template<typename TargetWlT>
    explicit DMANNAdapter(std::shared_ptr<const IDMACostProvider<TargetWlT>> target)
        : provider_wrapper_(std::make_unique<ProviderWrapper<TargetWlT>>(std::move(target))) {}

    /// @brief Calculate cost by converting workload and delegating to target provider
    /// @param wl The input workload (of type WlT)
    /// @return The cost cycles from the underlying provider
    CyclesInterfaceType get_cost(const WlT& wl, std::string* cost_source = nullptr) const override {
        return provider_wrapper_->get_cost_converted(wl, cost_source);
    }

    /// @brief This method is used to propagate initialization status because types like DMANNCostProvider and DMATheoreticalCostProvider
    /// does not inherit from DMANNAdapter, so we need to forward the is_initialized call.
    /// @return true if the underlying provider is initialized, false otherwise
    bool is_initialized() const override {
        return provider_wrapper_->is_initialized();
    }
    
private:
    /// @brief Type-erased interface for storing any provider wrapper
    ///
    /// This interface hides the TargetWlT template parameter, allowing us to
    /// store ProviderWrapper<TargetWlT> for any TargetWlT in a single member variable.
    struct IProviderWrapper {
        virtual ~IProviderWrapper() = default;

        /// @brief Convert WlT workload and compute cost
        /// @param wl The input workload to convert and evaluate
        /// @return The cost from the wrapped provider
        virtual CyclesInterfaceType get_cost_converted(const WlT& wl, std::string* cost_source = nullptr) const = 0;

        /// @brief Check if the wrapped provider is initialized
        /// @return true if the wrapped provider is initialized, false otherwise
        virtual bool is_initialized() const = 0;
    };
    
    /// @brief Concrete wrapper that knows both WlT and TargetWlT types
    /// @tparam TargetWlT The workload type accepted by the wrapped provider
    ///
    /// This class performs the actual workload conversion from WlT to TargetWlT
    /// and delegates the cost calculation to the wrapped provider.
    template<typename TargetWlT>
    struct ProviderWrapper : IProviderWrapper {
        // Shared pointer to the underlying provider
        std::shared_ptr<const IDMACostProvider<TargetWlT>> provider;

        /// @brief Construct wrapper around a provider
        /// @param p The provider to wrap (shared ownership)
        explicit ProviderWrapper(std::shared_ptr<const IDMACostProvider<TargetWlT>> p) : provider(std::move(p)) {}
        
        /// @brief Convert workload and calculate cost
        /// @param wl The input workload (WlT type)
        /// @return The cost calculated by the wrapped provider
        ///
        /// Uses DMAWorkloadTransformer to convert WlT -> TargetWlT,
        /// then delegates to the wrapped provider's get_cost method.
        CyclesInterfaceType get_cost_converted(const WlT& wl, std::string* cost_source = nullptr) const override {
            TargetWlT new_wl = DMAWorkloadTransformer::create_workload<TargetWlT>(wl);
            return provider->get_cost(new_wl, cost_source);
        }
        
        bool is_initialized() const override {
            return provider->is_initialized();
        }
    };
    
    std::unique_ptr<IProviderWrapper> provider_wrapper_;
};

}  // namespace VPUNN