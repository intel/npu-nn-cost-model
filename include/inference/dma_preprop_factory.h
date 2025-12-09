// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_PREPROC_FACTORY_H
#define DMA_PREPROC_FACTORY_H

#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <vector>

#include "dma_preprocessing.h"

#include "vpu/compatibility/dma_types_01x.h"  // detailed implementations  pp_v01

namespace VPUNN {

template <class DMADesc>
class DMAVersionsMapTypes {
public:
    using PrepropType = IPreprocessingDMA<float, DMADesc>;
    using PreprocessingMap = std::map<int, PrepropType&>;
};

template <class DMADesc>
class DMAVersionsMap : DMAVersionsMapTypes<DMADesc> {};

template <>
class DMAVersionsMap<DMANNWorkload_NPU27> : DMAVersionsMapTypes<DMANNWorkload_NPU27> {
    Preprocessing_Interface01_DMA<float> pp_v01;

public:
    /// @brief the map of versions mapped to preprocessing concrete objects
    const PreprocessingMap pp_map{
            {pp_v01.getInterfaceVersion(), pp_v01},
    };
};

template <>
class DMAVersionsMap<DMANNWorkload_NPU40_50> : DMAVersionsMapTypes<DMANNWorkload_NPU40_50> {
    Preprocessing_Interface02_DMA<float> pp_v02;
    Preprocessing_Interface03_DMA<float> pp_v03;

public:
    /// @brief the map of versions mapped to preprocessing concrete objects
    const PreprocessingMap pp_map{
            {pp_v02.getInterfaceVersion(), pp_v02},
            {pp_v03.getInterfaceVersion(), pp_v03},
    };
};

/**
 * @brief Provides processing related objects based on context
 *
 * The provided objects may be bounded(lifespan) to this instance
 */
template <class DMADesc>
class DMARuntimeProcessingFactory {
private:
    /// instances of preprocesors compatible with the selected descriptor
    DMAVersionsMap<DMADesc> descriptorVersionsMap;

public:
    /// @brief True if a preprocessor exists for required/interrogated version
    bool exists_preprocessing(int input_version) const noexcept {
        auto found = descriptorVersionsMap.pp_map.find(input_version);
        return (found != descriptorVersionsMap.pp_map.cend());
    }
    /** @brief provides a preprocessor for the required interface
     * The provided preprocessor is owned by this class.
     * For NOW multiple requests for the same version will provide the same object, the factory just shares the
     * preprocessors , does not create a new one for each request
     * @param version desired interface version
     * @return the preprocessor object to be used (shared)
     * @throws out_of_range in case the version is not supported
     */
    IPreprocessingDMA<float, DMADesc>& make_preprocessing(int version) const {
        if (exists_preprocessing(version)) {
            return descriptorVersionsMap.pp_map.at(version);
        }

        // throw
        std::stringstream buffer;
        buffer << "[ERROR]:DMA Preprocessing cannot be created for version:   " << version;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }
};

}  // namespace VPUNN
#endif  // guard
