// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPUEM_TYPES_H
#define VPUNN_VPUEM_TYPES_H

#include <array>
#include "types.h"
#include "vpu/vpuem_models_struct.h"


#include <iostream>

namespace VPUNN {


class VPUEM_Subblk_Tensor {
private:
    std::array<int, 3> shape_;
    DataType dtype_;
    Layout layout_;

public:
    VPUEM_Subblk_Tensor(const std::array<int, 3>& shape = {1, 1, 1}, DataType dtype = DataType::UINT8,
                        Layout layout = Layout::ZXY)
            : shape_(shape), dtype_(dtype), layout_(layout){};

    // get a 3 dim vector of tensor shape
    const std::array<int, 3>& get_shape() const noexcept {
        return shape_;
    }

    void set_shape(const int& m, const int& val) {
        shape_[m] = val;
    }

    DataType get_dtype_() const noexcept {
        return dtype_;
    }

    Layout get_layout_() const noexcept {
        return layout_;
    }

    int get_output_size() const {
        return static_cast<int>(shape_[0] * shape_[1] * shape_[2] * dtype_to_bytes(dtype_));
    };
};
}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
