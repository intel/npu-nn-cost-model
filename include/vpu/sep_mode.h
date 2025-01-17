// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SEP_MODE_H
#define VPUNN_SEP_MODE_H

#include <array>
#include <iostream>
#include <map>
#include <sstream>  //
#include <string>

#include "dim_enum.h"
#include "dpu_types.h"
#include "utils.h"

namespace VPUNN {

// to be moved to SEP info
using DimType = unsigned int;
class WHCBTensorShape {
protected:
    std::array<DimType, 4> shape{0, 0, 0, 0};

public:
    WHCBTensorShape(DimType w, DimType h, DimType c, DimType b): shape{w, h, c, b} {};

    /// @brief Get the x dimension
    DimType x() const noexcept {
        return shape[Dim::Act::X];
    };

    /// @brief Get the y dimension
    DimType y() const noexcept {
        return shape[Dim::Act::Y];
    };

    /// @brief Get the z dimension
    DimType z() const noexcept {
        return shape[Dim::Act::Z];
    };

    /// @brief Get the batch dimension
    DimType b() const noexcept {
        return shape[Dim::Act::B];
    };
    /// @brief Get the height
    DimType height() const noexcept {
        return y();
    };

    /// @brief Get the width
    DimType width() const noexcept {
        return x();
    };

    /// @brief Get the channels
    DimType channels() const noexcept {
        return z();
    };

    /// @brief Get the batches dimension
    DimType batches() const noexcept {
        return b();
    };

    /// @brief Set the width dimension
    void set_width(DimType w) {
		shape[Dim::Act::X] = w;
	}

    /// @brief Set the height dimension
    void set_height(DimType h) {
        shape[Dim::Act::Y] = h;
    }

    /// @brief Set the channels dimension
    void set_channels(DimType c) {
		shape[Dim::Act::Z] = c;
	}

    /// @brief Set the batches dimension
    void set_batches(DimType b) {
        shape[Dim::Act::B] = b;
    }

    /// @brief Get the size in samples
    /// @return how many elements are in this tensor shape
    DimType numberOfElements() const {
        return multiply_vector(shape);
    }

    /// equality test operator
    bool operator==(const WHCBTensorShape& b) const {
        return (shape == b.shape);
    }
};

class SEPModeInfo {
public:
    bool sep_activators{false};                             ///< activators using Storage elements table with pointers
    WHCBTensorShape storage_elements_pointers{0, 0, 0, 0};  ///< SEP pointer table, 32 bits pointers assumed
    WHCBTensorShape actual_activators_input{
            0, 0, 0, 0};        ///< actual tensor shape for activators. Datatype is the same as for input tensor
    bool no_sparse_map{false};  ///< if true the sparse map is ignored/non existent

public:
    bool isEnabled() const {
        return sep_activators;
    }
    bool operator==(const SEPModeInfo& b) const {
        bool r{true};
        r = r && (sep_activators == b.sep_activators);
        r = r && (storage_elements_pointers == b.storage_elements_pointers);
        r = r && (actual_activators_input == b.actual_activators_input);
        r = r && (no_sparse_map == b.no_sparse_map);
        return r;
    }
};

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::WHCBTensorShape& d) {
    stream << "[WHCB] :  \t{" << d.x() << "," << d.y() << "," << d.z() << "," << d.b() << "} ;";
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const VPUNN::SEPModeInfo& d) {
    stream << "SEPModeInfo: \n"                                                      //
           << " active: \t" << (d.isEnabled() ? "true" : "false") << " ;\n"          //
           << " SEP: " << d.storage_elements_pointers << " \n"                       //
           << " Input memory: " << d.actual_activators_input << " \n"                //
           << " No SparseMap: \t" << (d.no_sparse_map ? "true" : "false") << " ;\n"  //
            ;
    return stream;
}

}  // namespace VPUNN

#endif  // VPUNN_TYPES_H
