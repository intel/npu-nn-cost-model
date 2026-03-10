// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/http_cost_provider_factory.h"

#ifdef VPUNN_BUILD_HTTP_CLIENT
#include "http_client/http_cost_provider.h"
#endif


namespace VPUNN {

std::unique_ptr<IHttpCostProvider> HttpCostProviderFactory::create() {
#ifdef VPUNN_BUILD_HTTP_CLIENT
    return HttpCostProvider::initFromEnvironment();
#else
    return nullptr;
#endif // VPUNN_BUILD_HTTP_CLIENT
}

} // namespace VPUNN