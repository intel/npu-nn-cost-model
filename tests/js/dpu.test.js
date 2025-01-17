// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

const test_setup = require("./setup");

beforeAll(async () => await test_setup.initialize())

const VPUNN = test_setup.VPUNN;


test.each(["vpu_2_0", "vpu_2_7"])('test DPU cost', (device) => {

    const inT = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);
    //const inT_1 = VPUNN.createTensor(0, 0, 0, 0, VPUNN.DataType.UINT8); // a dummy input 1
    const outT = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);

    const VPUNN_device = device.toLowerCase() == "vpu_2_0" ? VPUNN.VPUDevice.VPU_2_0 : VPUNN.VPUDevice.VPU_2_7;
    const path = `models/${device.toLowerCase()}.vpunn`

    const mode = device.toLowerCase() == "vpu_2_0" ? VPUNN.ExecutionMode.MATRIX : VPUNN.ExecutionMode.CUBOID_16x16;

    const wl = VPUNN.createWorkload(
        VPUNN_device,
        VPUNN.Operation.CONVOLUTION,
        inT, /*inT_1,*/ outT, mode,
        3, 3, 1, 1, 0, 0, 0, 0)

    expect(wl).toBeTruthy();

    const model = VPUNN.createVPUCostModel(path);

    expect(model).toBeTruthy();
    expect(model.initialized()).toBeTruthy();

    const dpu_cost = model.DPU(wl)

    expect(dpu_cost).toBeGreaterThan(0);

})