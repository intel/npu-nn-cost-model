// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SERIALIZABLE_TENSOR_INFO_H
#define SERIALIZABLE_TENSOR_INFO_H

#include "vpu/dpu_types.h"
#include "vpu/dpu_types_info.h"
#include "vpu/vpu_tensor.h"
#include "vpu/dpu_defaults.h"

#include <array>
#include <iostream>

namespace VPUNN {

/// @brief holds info for a tensor.
struct TensorInfo {
    long long height{0};
    long long width{0};
    long long channels{0};
    long long batch{1};
    DataType datatype{DataType::UINT8};
    Layout layout{Layout::ZXY};  // same as ZMAJOR
    float sparsity{0.0F};
    bool sparsity_enabled{false};
    Swizzling swizzling{default_init_swizzling()};

    /// constructor based on DPUworkload related VPUTensor structure
    explicit TensorInfo(const VPUTensor& t)
            : height{static_cast<int>(t.height())},  // Y
              width{static_cast<int>(t.width())},    // X
              channels{static_cast<int>(t.z())},
              batch{static_cast<int>(t.b())},
              datatype{t.get_dtype()},
              layout{t.get_layout()},
              sparsity_enabled{t.get_sparsity()} {
    }

    TensorInfo() = default;

    /// @brief Get the size in samples
    /// @return how many elements are in this tensor shape
    long long numberOfElements() const {
        return height * width * channels * batch;
    }

    /// @brief Get the size in bytes based on packmode
    /// @return size in bytes
    unsigned int tensor_size_B() const {
        const std::array<unsigned int, 4> shape{static_cast<unsigned int>(width), static_cast<unsigned int>(height),
                                                static_cast<unsigned int>(channels), static_cast<unsigned int>(batch)};
        VPUTensor t{shape, datatype, layout, sparsity_enabled};
        return t.size();
    }

    /// @brief This function compute the size in bytes of the innermost dimension of a given tensor by using the
    /// function tensor_size_B() which takes into account datatype size and the way memory is packed (packmode)
    /// @return size in bytes of the innermost dimension
    long long get_tensor_innermost_dim_B() const noexcept {
        const auto innermost_dim{layout_to_order(layout)[0]};  // innermost dimension

        // create a TensorInfo that contains only the innermost dimension
        TensorInfo t_inner_dim_algn{*this};
        t_inner_dim_algn.width = 1;
        t_inner_dim_algn.height = 1;
        t_inner_dim_algn.channels = 1;
        t_inner_dim_algn.batch = 1;

        switch (innermost_dim) {
        case Dim::Act::X:
            t_inner_dim_algn.width = width;
            break;
        case Dim::Act::Y:
            t_inner_dim_algn.height = height;
            break;
        case Dim::Act::Z:
            t_inner_dim_algn.channels = channels;
            break;
        case Dim::Act::B:
            t_inner_dim_algn.batch = batch;
            break;
        default:
            break;  // nothing
        }

        const auto innermost_dim_B{t_inner_dim_algn.tensor_size_B()};

        return innermost_dim_B;
    }

    /// @brief Product of all shape dimensions except the innermost one (element count, not bytes)
    long long numberOfElementsExcludingInnermost() const noexcept {
        const auto innermost_dim{layout_to_order(layout)[0]};  // innermost dimension
        switch (innermost_dim) {
        case Dim::Act::X:  // exclude width
            return height * channels * batch;
        case Dim::Act::Y:  // exclude height
            return width * channels * batch;
        case Dim::Act::Z:  // exclude channels
            return width * height * batch;
        case Dim::Act::B:  // exclude batch
            return width * height * channels;
        default:
            return width * height * channels * batch;  // fallback: all dims
        }
    }

    /// @brief Convert to VPUTensor
    VPUTensor toVPUTensor() const {
        return VPUTensor({static_cast<unsigned int>(width), static_cast<unsigned int>(height),
                          static_cast<unsigned int>(channels), static_cast<unsigned int>(batch)},
                         datatype, layout, sparsity_enabled);
    }

    /// Check if this slot has been populated (non-zero shape)
    bool isNonZero() const {
        return !(channels == 0 && height == 0 && width == 0);
    }
};

inline std::ostream& operator<<(std::ostream& stream, const TensorInfo& d) {
    stream << "TensorInfo: \n"                                                                                        //
           << " shape: \t{" << d.width << "," << d.height << ","                                                      //
           << d.channels << "," << d.batch << "} ;\n"                                                                 //
           << " dtype: \t" << (int)d.datatype << " : " << DataType_ToText.at(static_cast<int>(d.datatype)) << " ;\n"  //
           << " layout: \t" << (int)d.layout << " : " << Layout_ToText.at(static_cast<int>(d.layout)) << " ;\n"       //
           << " sparsity enabled: \t" << (d.sparsity_enabled ? "true" : "false") << " ;\n"                            //
           << " sparsity value: \t" << d.sparsity << " ;\n"                                                           //
           << " swizzling: \t{" << (int)d.swizzling << "}"
           << " :  {" << Swizzling_ToText.at(static_cast<int>(d.swizzling)) << "} ;\n"  //
            ;
    return stream;
}

} // namespace VPUNN

#endif // SERIALIZABLE_TENSOR_INFO_H
