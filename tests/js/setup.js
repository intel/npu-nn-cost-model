// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

const VPUNN = require("vpunn/vpunn_bind");

const sleep = (ms) => {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const initialize = async () => {
    let isInitialized = false;

    VPUNN.onRuntimeInitialized = async _ => {
        isInitialized = true;
    };

    while (!isInitialized) {
        await sleep(100);
    }
}

exports.initialize = initialize;
exports.VPUNN = VPUNN
