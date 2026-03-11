# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

from ._VPUNN import *
import inspect

__all__ = []

for name, obj in list(globals().items()):
    if not name.startswith("_") and not inspect.ismodule(obj):
        __all__.append(name)
        if hasattr(obj, "__module__"):
            obj.__module__ = "vpunn"
