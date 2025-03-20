// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef POSTPROCESSING_FACTORY_H
#define POSTPROCESSING_FACTORY_H

#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <vector>

#include "post_process.h"
#include "postprocessing_mocks.h"

namespace VPUNN {
/**
 * @brief Provides processing related objects based on context
 *
 * The provided objects may be bounded(lifespan) to this instance
 */
class PostProcessingFactory {
private:
    using ProcType = IPostProcess;
    using PostprocessingMap = std::map<int, ProcType&>;

    // a simple and not optimum (allocates all static)
    PassThroughPostP pp_default{};
    AdaptFromNPU27to40 pp_NPU27BasedAdapter{};
    AdaptFromNPU40to40 pp_NPU40Selfdapter{};

    /// @brief the map of versions mapped to preprocessing concrete objects
    const PostprocessingMap pp_map{
            {(int)NNOutputVersions::OUT_LATEST, pp_default},
            {(int)NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED, pp_default},
            {(int)NNOutputVersions::OUT_HW_OVERHEAD_UNBOUNDED, pp_default},
            {(int)NNOutputVersions::OUT_CYCLES, pp_default},
            {(int)NNOutputVersions::OUT_CYCLES_NPU27, pp_NPU27BasedAdapter},
            {(int)NNOutputVersions::OUT_CYCLES_NPU40_DEV, pp_NPU40Selfdapter},
    };

public:
    /// @brief True if a preprocessor exists for required/interrogated version
    bool exists(int version) const noexcept {
        auto found = pp_map.find(version);
        return (found != pp_map.cend());
    }
    /** @brief provides a processor for the required interface
     * The provided rocessor is owned by this class.
     * For NOW multiple requests for the same version will provide the same object, the factory just shares the
     * preprocessors , does not create a new one for each request
     * @param version desired interface version
     * @return the processor object to be used (shared)
     * @throws out_of_range in case the version is not supported
     */
    const IPostProcess& make(int version) const {
        if (exists(version)) {
            return pp_map.at(version);
        }

        // throw
        std::stringstream buffer;
        buffer << "[ERROR]:Post processing cannot be created for version:   " << version;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }
};

}  // namespace VPUNN
#endif  // guard
