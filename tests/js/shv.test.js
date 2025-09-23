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


test.each(['Sigmoid', 'Swish', 'HardSwish'])('test SHV cost', (op) => {

    const inT = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);
    const outT = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);

    const VPUNN_device = VPUNN.VPUDevice.VPU_2_7;
    const path = `models/vpu_2_7.vpunn`

    const wl = VPUNN.createSHV(
        op,
        VPUNN_device,
        inT, outT)

    expect(wl).toBeTruthy();

    const model = VPUNN.createVPUCostModel(path);

    expect(model).toBeTruthy();
    expect(model.initialized()).toBeTruthy();

    const shv_cost = model.SHAVE(wl)
    expect(shv_cost).toBeGreaterThan(0);

})