// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/shave/shave_collection.h"

#include "vpu/shave/layers.h"
namespace VPUNN {
// this file gets  active when we have a shave lib. Intention is to generate the populate method!

void ShaveInstanceHolder_VPU27::populate() {
    // adauga 1 instanta concreta
    // clang-format off
         //Add<DataType::FLOAT16, 8, 16, 1300,975>("sigmoid", 8.040653645736843e-05F, 2.3877565530828724F,0.052000000000000046F, 0.6203295622495082F);
         //Add("Mocking1");
        // Add<SHVHardSigmoid, int(0.547F * 1000),4956>("HardSigmoid");//activation
        // Add<SHVTranspose, int(0.1F * 1000),1000>("Transpose");//data movement
        // Add<SHVMinimum, int(0.015F * 1000),11047>("Minimum");//element wise

         #include "vpu/shave/SHAVE_V27.inl"

    // clang-format on
}

void ShaveInstanceHolder_VPU27CLassic::populate() {
    // adauga 1 instanta concreta
    // clang-format off
         //Add<SHVHardSigmoid, int(0.547F * 1000),4956>("HardSigmoid");//activation
        // Add<SHVTranspose, int(0.1F * 1000),1000>("Transpose");//data movement
         //Add<SHVMinimum, int(0.015F * 1000),11047>("Minimum");//element wise
        
        #include "vpu/shave/SHAVE_V27_Linear.inl"

    // clang-format on
}

}  // namespace VPUNN