// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_POST_PROCESSING_H
#define DMA_POST_PROCESSING_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace VPUNN {

/// @brief enum for DMA NN output versions
enum class DMAOutputVersions : int {
    OUT_BANDWIDTH_UTILIZATION = 1,  ///< [0-1] representing 0 to devices' bytes per cycle
    OUT_CYCLES_DIRECT = 2,          ///< DPU cycles directly output by the NN
};

/** @brief Configuration options concerning the interpretation and post processing of inferred values
 * This class have the goal to check if we know something about the output version of the model and if we don't know we
 * will not support the output for the DMA. We are going to use the output version parsed by the ModelVersion and
 * use it to determine based on known output version if we support it or not. In case that we don't know the version we
 * are going to not support the output.
 */
class DMAPostProcessSupport {
public:
    /// @brief The constructor for this object who sets the supported bool depending on output version
    /// @param output_version is the output version based on the ModelVersion
    DMAPostProcessSupport(int output_version) {
        set_output_version(output_version);
    }

    /// @brief a method to see if we support the output
    bool is_output_supported() const {
        return output_support;
    }

protected:
    /** @brief a method to set the output support based on the output version parsed as an int
     *
     * The set_output_version will determine based on the DMAOutputVersions if we know something about the output
     * version. In case that we know if it is supported or not based on the known versions of output, we are going to
     * set the bool output_support on either true or false. In case that we don't know the version that is coming than,
     * we are going to not support that version.
     *
     *@param output_version the output version of the Model
     */
    void set_output_version(const int version) {
        // based on known output versions that we support:
        switch (version) {
        case (int)VPUNN::DMAOutputVersions::OUT_BANDWIDTH_UTILIZATION:
        case (int)VPUNN::DMAOutputVersions::OUT_CYCLES_DIRECT:
            output_support = true;
            break;

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
