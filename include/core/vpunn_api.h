// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_API_H
#define VPUNN_API_H
#if defined(_WIN32)
#define VPUNN_API(...) __VA_ARGS__
#else
#define VPUNN_API(...) __attribute__((visibility("default"))) __VA_ARGS__
#endif

#endif
