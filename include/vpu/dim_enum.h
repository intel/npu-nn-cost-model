// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_DIM_ENUM_H
#define VPUNN_DIM_ENUM_H


namespace VPUNN {

// dimensions labels for indexes
namespace Dim {
// the enums are un-scoped enums because they must be easy to convert to array indexes

enum Grid : int { W, H };
enum Act : int { X, Y, Z, B };
enum Wt : int { K, C, Ky, Kx };
enum Padding : int { TOP, BOTTOM, LEFT, RIGHT };

}  // namespace Dim

}  // namespace VPUNN

#endif
