// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/http_cost_provider_intf.h"
#include "vpu/http_workload_variant.h" // this header is exposed only to this .cpp file to avoid pulling heavy type dependencies into the lightweight interface header

namespace VPUNN {

// IHttpCostProvider::getCost - wraps the workload into HttpWorkloadVariant and dispatches via getCostImpl.
template <typename WlT>
CyclesInterfaceType IHttpCostProvider::getCost(const WlT& op, std::string& info) const {
    return getCostImpl(HttpWorkloadVariant(std::cref(op)), info);
}

// Explicit template instantiations for IHttpCostProvider::getCost (called through the interface pointer)
template CyclesInterfaceType IHttpCostProvider::getCost<DPUOperation>(const DPUOperation&, std::string&) const;
template CyclesInterfaceType IHttpCostProvider::getCost<DMANNWorkload_NPU27>(const DMANNWorkload_NPU27&, std::string&) const;
template CyclesInterfaceType IHttpCostProvider::getCost<DMANNWorkload_NPU40_50>(const DMANNWorkload_NPU40_50&, std::string&) const;

}  // namespace VPUNN
