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

test('test uint8 tensor creation', () => {
  const tensor = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.UINT8);
  expect(tensor).not.toBeUndefined()
});

test('test fp16 tensor creation', () => {
  const tensor = VPUNN.createTensor(56, 56, 64, 1, VPUNN.DataType.FLOAT16);
  expect(tensor).not.toBeUndefined()
});

