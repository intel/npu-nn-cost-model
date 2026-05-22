// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TILING_ALGORITHM_FACTORY_H
#define VPUNN_TILING_ALGORITHM_FACTORY_H

#include <list>
#include <memory>

#include "tiler.h"
#include "split_options.h"
#include "vpu/layer.h"

namespace VPUNN {

/**
 * @brief A list of algorithms
 */
using TilingAlgorithmsContainer = std::list<std::unique_ptr<ITilerAlgorithm>>;
    
class TilingAlgorithmFactory {
public:
    /**
     * @brief Factory functions that returns a list of algorithms from the strategies
     *
     * @param layer the DPULayer
     * @param options the split algorithm optimization options
     * @return TilingAlgorithms
     */
    static TilingAlgorithmsContainer getTilingAlgorithms(const DPULayer& layer, const SplitOptions& options);
};

} // namespace VPUNN

#endif // TILING_ALGORITHM_FACTORY_H
