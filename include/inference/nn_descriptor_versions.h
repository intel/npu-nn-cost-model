// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef NN_DESCRIPTOR_VERSIONS_H
#define NN_DESCRIPTOR_VERSIONS_H

namespace VPUNN {

/// @brief enum for NN descriptor versions (input versions)
enum class NNVersions : int {
    VERSION_00_LATEST_NONE = 0,  ///< no version OR last version
    VERSION_01_BASE = 1,         ///< base version, the unnamed one
    VERSION_10_ENUMS_SAME = 10,  ///< evo of v01, with correct size. est November 2022 VPU2.7 alpha release
    VERSION_11_VPU27_BETA = 11,  ///< input 1 generated, isi strategy, layouts. est Jan 2023 VPU2.7 beta release
    VERSION_12_HALO = 12,        ///< +halo, -isi_startegy, -device. est Jan 2024 prepared for NPU4

    VERSION_11_V89_COMPTBL = 89,  ///< v11 for NPU27 NNs where no swizzling or trainingspace limitations are active.
                                  ///< Compatibility mode for intial release (NOv 2024 vs Oct 2023)
    VERSION_11_NPU40 = 4011,         ///< version 11 used for NPU40 trained NNs
    VERSION_11_NPU41 = 4111,     ///< version 11 used for NPU40 trained NNs with swizz0,5 support.
    VERSION_12_NPU_RESERVED = 5112,  ///< version 12 used for NPU_RESERVED trained NNs
};

}  // namespace VPUNN
#endif
