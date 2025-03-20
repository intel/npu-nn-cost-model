// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace VPUNN {

/// @brief enum for NN output versions
enum class NNOutputVersions : int {
    OUT_LATEST = 0,                 ///< last version, expecting cycles as output
    OUT_HW_OVERHEAD_BOUNDED = 1,    ///< Output expected as hw_overhead bounded, deprecated
    OUT_CYCLES = 2,                 ///< expecting cycles as output
    OUT_HW_OVERHEAD_UNBOUNDED = 3,  ///< Output expected as hw_overhead unbounded, deprecated
    OUT_CYCLES_NPU27 = 4,           ///< expecting cycles as output, tuned for NPU27. ALl other new need mocks/adapters
    OUT_CYCLES_NPU40_DEV = 5,  ///< expecting cycles as output, tuned for NPU40. Also NPU40 can use post processing for
                               ///< untrained spaces. This is changeable during development .
};

/** @brief Configuration options concerning the interpretation and post processing of inferred values
 * This class have the goal to check if we know something about the output version of the model and if we don't know we
 * will not support the output for the CostModel. We are going to use the output version parsed by the ModelVersion and
 * use it to determine based on known output version if we support it or not. In case that we don't know the version we
 * are going to not supprt the output.
 */
class PostProcessSupport {
public:
    /// @brief The constructor for this object who sets the supported bool depending on output version
    /// @param output_version is the output version based on the ModelVersion extracted from NN raw name
    PostProcessSupport(int output_version) {
        set_output_version(output_version);
    }

    /// @brief a method to see if we support the output
    bool is_output_supported() const {
        return output_support;
    }

protected:
    /** @brief a method to set the output support based on the output version parsed as an int
     *
     * The set_output_version will determine based on the NNOutputVersions if we know something about the output
     * version. In case that we know if it is supported or not based on the known versions of output, we are going to
     * set the bool output_support on either true or false. In case that we don't know the version that is coming than,
     * we are going to not support that version.
     *
     *@param output_version the output version of the Model
     */
    void set_output_version(const int version) {
        // based on known output versions that we support:
        switch (version) {
        case (int)VPUNN::NNOutputVersions::OUT_LATEST:
        case (int)VPUNN::NNOutputVersions::OUT_CYCLES:
        case (int)VPUNN::NNOutputVersions::OUT_CYCLES_NPU27:
        case (int)VPUNN::NNOutputVersions::OUT_CYCLES_NPU40_DEV:
            output_support = true;
            break;

        case (int)VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED:
        case (int)VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_UNBOUNDED:
        // in case we don't have the version in the list we will not support the output
        default:
            output_support = false;
            break;
        }
    }

private:
    bool output_support;
};
}  // namespace VPUNN
#endif
