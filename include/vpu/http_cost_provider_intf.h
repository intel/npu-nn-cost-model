// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef HTTP_COST_PROVIDER_INTF_H_
#define HTTP_COST_PROVIDER_INTF_H_

#include <memory>
#include <string>
#include "vpu/types.h"
#include "vpu/profiling_service.h"

namespace VPUNN {

/**
 * @brief Forward declaration, full definition in http_workload_variant.h
 * @details This structure is used as variant wrapper so it can be defined in a separate internal header
 */
struct HttpWorkloadVariant;

/**
 * @brief Interface for HTTP Cost Provider used to expose the functionality of HttpCostProvider.
 */
class IHttpCostProvider {
public:
    virtual ~IHttpCostProvider() = default;

    /**
     * @brief Checks if the profiling service is available.
     * @return True if available, false otherwise.
     */
    virtual bool is_available() const = 0;

    /**
     * @brief Enable or disable debug output.
     * @param enable True to enable debug output, false to disable.
     * @return void
     */
    virtual void setDebug(bool enable) = 0;

    /**
     * @brief Retrieves the profiling backend as string from the provided enum hint.
     * @param backend The backend enum value to convert.
     * @return String representation of the backend, defaults to "ProfilingServiceBackend::SILICON" if invalid
     */
    virtual const std::string profilingBackendToString(ProfilingServiceBackend backend) const = 0;
protected:
    /**
     * @brief Internal implementation of getCost that dispatches calls based on workload type.
     *
     * This method is intended to be overridden by concrete implementations and is called by
     * the templated getCost() wrapper, which wraps the workload into a HttpWorkloadVariant for
     * type-safe dispatch via std::visit.
     * 
     * @param op The workload operation wrapped in a HttpWorkloadVariant.
     * @param info A string to store additional information.
     * @return The cost as CyclesInterfaceType, in case of error returns Cycles::ERROR_PROFILING_SERVICE.
     */
    virtual CyclesInterfaceType getCostImpl(const HttpWorkloadVariant& op, std::string& info) const = 0;

public:
    /**
     * @brief Retrieves the cost associated with a given workload operation.
     * @tparam WlT The type of the workload operation.
     * @param op The operation of type WlT for which to get the cost.
     * @param info A string to store additional information.
     * @return The cost as CyclesInterfaceType, in case of error returns Cycles::ERROR_PROFILING_SERVICE.
     */
    template <typename WlT>
    CyclesInterfaceType getCost(const WlT& op, std::string& info) const;
};

}  // namespace VPUNN

#endif // HTTP_COST_PROVIDER_INTF_H_
