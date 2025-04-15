// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

//#include "vpu/shave/shave_devices.h"
//
//namespace VPUNN {
//// this file gets  active when we have a shave lib. Intention is to generate the populate method!
//
//void ShaveCache::populate() {
//    // add concrete instances, generated or by hand
//    const SHAVEWorkload wl3{
//            "sigmoid",
//            VPUDevice::VPU_4_0,
//            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
//            {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
//            {{7}},
//    };
//
//    this->shaveCacheRaw.add(wl3, 1);
//    this->shaveCacheRaw.add({"sigmoid",
//                             VPUDevice::VPU_4_0,
//                             {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
//                             {VPUTensor(10, 100, 5, 1, DataType::FLOAT16)},
//                             {{10}}},
//                            2);
//
//    // clang-format off
//
//    // clang-format on
//}
//
//}  // namespace VPUNN