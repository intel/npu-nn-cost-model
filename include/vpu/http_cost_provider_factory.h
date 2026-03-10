// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef HTTP_COST_PROVIDER_FACTORY_H_
#define HTTP_COST_PROVIDER_FACTORY_H_

#include "vpu/http_cost_provider_intf.h"

namespace VPUNN {

class HttpCostProviderFactory {
public:
    /**
     * @brief Factory method to create an instance of IHttpCostProvider.
     * 
     * If the VPUNN_BUILD_HTTP_CLIENT flag is set, it creates an instance of HttpCostProvider,
     * otherwise returns nullptr. That's why after using create one should always check for nullptr.
     * 
     * @return A unique pointer to the created IHttpCostProvider instance.
     */
    static std::unique_ptr<IHttpCostProvider> create();
};

}  // namespace VPUNN

#endif  // HTTP_COST_PROVIDER_FACTORY_H_
