# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

import cppyy
import os

# Import CPPYY class
file_path = os.path.dirname(os.path.abspath(__file__))
cppyy.add_include_path(os.path.join(file_path, "include"))
cppyy.include("vpu_network_cost_model.h")
cppyy.load_library(os.path.join(file_path, "lib/libinference.so"))
cppyy.load_library(os.path.join(file_path, "lib/libvpunn_optimization.so"))


from cppyy.gbl import VPUNN  # noqa
