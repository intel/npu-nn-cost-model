// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef POLY_MODELS_H
#define POLY_MODELS_H

#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <vector>

#include "shave_equations.h"
#include "shave_model_basics.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

class InterpolateWHModel_1 : public ShaveCyclesProvider<InterpolateWHModel_1> {
private:
    const DataType dtype_;  ///< the data type of the output ()

    friend std::ostream& operator<<(std::ostream& stream, const VPUNN::InterpolateWHModel_1& d);

public:
    InterpolateWHModel_1(DataType dtype, unsigned int DpuFreq, unsigned int ShvFreq)
            : ShaveCyclesProvider<InterpolateWHModel_1>{DpuFreq, ShvFreq}, dtype_(dtype) {
    }

public:
    const std::string formula{"22.83932536793509 + 1.1171697884536729 * H + 2.001050052740202 * W + "
                              "0.008051784863285609 * out_size +0.0067717638956515695 * out_size / W"};

    /**
     * gets us time, specific parameters
     *
     * @return the time in us
     */
    float getMicroSeconds(const int& wi, const int& hi, const int& out_size) const {
        const float outOverW{(float)out_size / wi};
        const float y_microseconds{22.83932536793509f + 1.1171697884536729f * hi + 2.001050052740202f * wi +
                                   0.008051784863285609f * out_size +
                                   0.0067717638956515695f * outOverW};  // all in samples
        return y_microseconds;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::InterpolateWHModel_1& d) {
    stream << "InterpolateWHModel_1: \n"
           << " dtype: \t" << (int)d.dtype_ << " : " << DataType_ToText.at(static_cast<int>(d.dtype_)) << " ;\n"  //
           << " formula: \t" << d.formula << "; \n"
           << d.converter                                 //
           << out_terminator() << "InterpolateWHModel_1"  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif /* SHAVEMODEL1to1_H*/
