// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

const VPUNN = require("vpunn/vpunn_bind");


VPUNN.onRuntimeInitialized = async _ => {
    console.log("===========================");
    await main("vpu_2_0");
    console.log("===========================");
    await main("vpu_2_7");
    console.log("===========================");
};


const main = async (device = "vpu_2_0") => {
    const inT = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);
    const outT = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);

    const VPUNN_device = device.toLowerCase() == "vpu_2_0" ? VPUNN.VPUDevice.VPU_2_0 : VPUNN.VPUDevice.VPU_2_7;
    const path = `models/${device.toLowerCase()}.vpunn`

    const mode = device.toLowerCase() == "vpu_2_0" ? VPUNN.ExecutionMode.MATRIX : VPUNN.ExecutionMode.CUBOID_16x16;

    const wl = VPUNN.createWorkload(
        VPUNN_device,
        VPUNN.Operation.CONVOLUTION,
        inT, outT, mode,
        3, 3, 1, 1, 0, 0, 0, 0)

    const model = VPUNN.createVPUCostModel(path)
    // Load the model from filesystem
    if (!model.initialized()) {
        console.log(`Model is NOT initialized!`)
    }
    else {
        console.log(`Model correctly initialized with ${device}`)
    }

    const dma_cost = model.DMA(VPUNN_device, inT, outT, 1);
    console.log(`DMA cost: ${dma_cost}`)

    const dpu_cost = model.DPU(wl)
    console.log(`DPU cost: ${dpu_cost}`)

}

