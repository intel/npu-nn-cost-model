// Copyright © 2023 Intel Corporation
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

test('libary load', () => {
    expect(VPUNN).not.toBeUndefined()
})


test.each(["vpu_2_0", "vpu_2_7", undefined])('model load', (device) => {
    const path = `models/${device}.vpunn`
    const model = VPUNN.createVPUCostModel(path);
    expect(model).toBeTruthy();
    if (device) {
        expect(model.initialized()).toBeTruthy();
    }
    else {
        expect(model.initialized()).toBeFalsy();
    }
})


test('wrong model', () => {
    const path = `./js/package.json`
    const model = VPUNN.createVPUCostModel(path);
    expect(model).toBeTruthy();
    expect(model.initialized()).toBeFalsy();
})


