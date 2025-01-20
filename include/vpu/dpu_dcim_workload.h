// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DPU_DCIM_WORKLOAD_H
#define VPUNN_DPU_DCIM_WORKLOAD_H

#include <array>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>  //
#include <string>

#include "dpu_defaults.h"
#include "dpu_halo.h"
#include "dpu_types.h"
// #include "sep_mode.h"
#include "vpu_tensor.h"

namespace VPUNN {

/// @brief The base structure that encodes a DPU DCIM workloads
/// Normally the tensors in/out that describe the operation are expected to be the compute tensors
struct DCIMWorkload {
    // Operation op;      ///< operation, like convolution, etc

    /// input0 tensors, the data/activation tensor details. This is the Compute Tensor
    /// the weights tensor is deduced.
    VPUTensor input0{};

    /// output tensor. This is the Compute Tensor
    VPUTensor output0{};

    std::array<unsigned int, 2> kernels;  ///< kernel sizes WH
    std::array<unsigned int, 2> strides;  ///< kernel strides WH
    std::array<unsigned int, 4> padding;  ///< kernel padding Top, Bottom, Left, Right.

    //// Padding and positive halo do not work together

    // ExecutionMode execution_order;  ///< execution mode

    ///// @brief broadcast policy, Split Over K situation , In the SOK tiling strategy, weights are split across
    ///// the tiles over the K dimension. The DPU in each tile compute a K-slice of the output tensors and
    ///// then broadcast the result in each CMX tile, implicitly concatenating the results and having then
    ///// all activations completely replicated.
    /////
    ///// OWT = The full Output is written to tiles specified. (does not have a direction!) Not limited to SOK
    /// situations
    /////  (0, 1 is self, 2,3,4,5,6 is to how many in total).
    ///// Individual output halo still have meaning independent of it(owt).
    unsigned int output_write_tiles{1};

    HaloWorkload halo{};  ///< halo aspects

    // operations/methods
public:
    /// equality test operator
    bool operator==(const DCIMWorkload& b) const {
        bool r{true};

        // r = r && (op == b.op);
        r = r && (input0 == b.input0);
        r = r && (output0 == b.output0);

        r = r && (kernels == b.kernels);
        r = r && (strides == b.strides);
        r = r && (padding == b.padding);

        // r = r && (execution_order == b.execution_order);

        r = r && (output_write_tiles == b.output_write_tiles);

        // halo
        r = r && (halo == b.halo);

        return false;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::DCIMWorkload& d) {
    stream << "DCIMWorkload: \n"  //
           << " input: \t{\n"
           << d.input0 << " } ;\n"  //
           << " output: \t{\n"
           << d.output0 << " } ;\n"  //

           << " kernels: [W,H]  \t{" << d.kernels[Dim::Grid::W] << "," << d.kernels[Dim::Grid::H] << "} ;\n"  //
           << " strides: [W,H]  \t{" << d.strides[Dim::Grid::W] << "," << d.strides[Dim::Grid::H] << "} ;\n"  //
           << " padding: [TBLR] \t{" << d.padding[Dim::TOP] << "," << d.padding[Dim::BOTTOM] << ","           //
           << d.padding[Dim::LEFT] << "," << d.padding[Dim::RIGHT] << "} ;\n"

           << " output_write_tiles: \t" << d.output_write_tiles << " ;\n"  //
           << d.halo                                                       //<< " ;\n"

           //
           << out_terminator() << "DCIMWorkload "  // terminator
            ;
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
