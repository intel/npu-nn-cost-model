// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_POSTPROCESSING_FACTORY_H
#define DMA_POSTPROCESSING_FACTORY_H

#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <vector>

#include "dma_post_process.h"
#include "dma_postprocessing_mocks.h"

namespace VPUNN {
/**
 * @brief Provides processing related objects based on context
 *
 * The provided objects may be bounded(lifespan) to this instance
 */
template <class DMADesc>
class DMAPostProcessingFactory {
private:
    using ProcType = IPostProcessDMA<DMADesc>;
    using PostprocessingMap = std::map<int, ProcType&>;

    // a simple and not optimum (allocates all static)
    ConvertFromSizeDivCycleToDPUCyc<DMADesc> pp_SizeDivCycToDPUCyc_converter{};
    ConvertFromDirectCycleToDPUCyc<DMADesc> pp_DirectCycToDPUCyc_converter{};

    /// @brief the map of versions mapped to preprocessing concrete objects
    const PostprocessingMap pp_map{
            {(int)DMAOutputVersions::OUT_BANDWIDTH_UTILIZATION, pp_SizeDivCycToDPUCyc_converter},
            {(int)DMAOutputVersions::OUT_CYCLES_DIRECT, pp_DirectCycToDPUCyc_converter},
    };

public:
    /// @brief True if a preprocessor exists for required/interrogated version
    bool exists(int version) const noexcept {
        auto found = pp_map.find(version);
        return (found != pp_map.cend());
    }
    /** @brief provides a processor for the required interface
     * The provided processor is owned by this class.
     * For NOW multiple requests for the same version will provide the same object, the factory just shares the
     * preprocessors , does not create a new one for each request
     * @param version desired interface version
     * @return the processor object to be used (shared)
     * @throws out_of_range in case the version is not supported
     */
    const IPostProcessDMA<DMADesc>& make(int version) const {
        if (exists(version)) {
            return pp_map.at(version);
        }

        // throw
        std::stringstream buffer;
        buffer << "[ERROR]:DMA Post processing cannot be created for version:   " << version;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }
};

}  // namespace VPUNN
#endif  // guard
