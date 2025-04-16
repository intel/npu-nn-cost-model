// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef PREPROC_FACTORY_H
#define PREPROC_FACTORY_H

// #include <math.h>
#include <vpu/compatibility/types01.h>  // detailed implementations
#include <vpu/compatibility/types11.h>  // detailed implementations  pp_v01
#include <vpu/compatibility/types12.h>  // detailed implementations
#include <vpu/compatibility/types13.h>  // detailed implementations
#include <vpu/compatibility/types14.h>  // detailed implementations
#include <vpu/types.h>
#include <sstream>  // for error formating
#include <stdexcept>
#include <string>
#include <vector>

#include "nn_descriptor_versions.h"
#include "preprocessing.h"

namespace VPUNN {
/**
 * @brief Provides processing related objects based on context
 *
 * The provided objects may be bounded(lifespan) to this instance
 */
class RuntimeProcessingFactory {
private:
    using PrepropType = Preprocessing<float>;
    using PreprocessingMap = std::map<int, PrepropType&>;

    // a simple and not optimum (allocates all static)
    // PreprocessingLatest<float> pp_v00_latest;
    Preprocessing_Interface01<float> pp_v01_base;
    Preprocessing_Interface10<float> pp_v10;
    Preprocessing_Interface11<float> pp_v11;
    Preprocessing_Interface4011<float> pp_v4011;
    Preprocessing_Interface4111<float> pp_v4111;
    Preprocessing_Interface12<float> pp_v12;
    Preprocessing_Interface13<float> pp_v13;
    Preprocessing_Interface14<float> pp_v14;
    Preprocessing_Interface15911<float> pp_v89_11;  // special v159

    /// @brief the map of versions mapped to preprocessing concrete objects
    const PreprocessingMap pp_map{
            //{pp_v00_latest.getInterfaceVersion(), pp_v00_latest},
            {pp_v01_base.getInterfaceVersion(), pp_v01_base},  //
            {pp_v10.getInterfaceVersion(), pp_v10},            //
            {pp_v11.getInterfaceVersion(), pp_v11},            //
            {pp_v89_11.getInterfaceVersion(), pp_v89_11},      //
            {pp_v4011.getInterfaceVersion(), pp_v4011},        //
            {pp_v4111.getInterfaceVersion(), pp_v4111},        //
            {pp_v12.getInterfaceVersion(), pp_v12},            //
            {pp_v13.getInterfaceVersion(), pp_v13},            //
            {pp_v14.getInterfaceVersion(), pp_v14}             //
    };

public:
    /// @brief True if a preprocessor exists for required/interrogated version
    bool exists_preprocessing(int input_version) const noexcept {
        auto found = pp_map.find(input_version);
        return (found != pp_map.cend());
    }
    /** @brief provides a preprocessor for the required interface
     * The provided preprocessor is owned by this class.
     * For NOW multiple requests for the same version will provide the same object, the factory just shares the
     * preprocessors , does not create a new one for each request
     * @param version desired interface version
     * @return the preprocessor object to be used (shared)
     * @throws out_of_range in case the version is not supported
     */
    Preprocessing<float>& make_preprocessing(int version) const {
        if (exists_preprocessing(version)) {
            return pp_map.at(version);
        }

        // throw
        std::stringstream buffer;
        buffer << "[ERROR]:Preprocessing cannot be created for version:   " << version;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }
};

}  // namespace VPUNN
#endif  // guard
