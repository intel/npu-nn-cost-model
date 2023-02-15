# Copyright © 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the ?Software Package?)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the ?third-party-programs.txt? or other similarly-named text file included with the
# Software Package for additional details.

import test_wl_selection

from vpunn.cost import VPUCostModel
from vpunn.cost import getInputDim
from test_wl_selection import (
    ceildiv,
    is_valid_combination,
    common_h_w_size,
    common_channel_size,
)
from allpairspy import AllPairs
import pytest
import json
import os

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

this_file_path = os.path.dirname(os.path.abspath(__file__))
test_config = os.path.join(this_file_path, "dma_list.json")

with open(test_config) as fp:
    confis = json.load(fp)


@pytest.fixture(scope="module")
def model():
    return VPUCostModel("models/vpu_2_0.vpunn")


def buildDMAWl(
    op,
    device,
    batch,
    input_channels,
    output_channels,
    height,
    width,
    kernel,
    strides,
    padding,
    input_dtype,
    output_dtype,
):

    # Kernel for eltwise
    kernel = kernel if op.upper() != "ELTWISE" else 1

    # Check same channels for non conv
    if (op != "CONVOLUTION") and input_channels != output_channels:
        input_channels = output_channels

    padding = 0 if padding == "VALID" else kernel // 2

    input_width, input_height = getInputDim(
        [width, height], 2 * [kernel], 2 * [padding], 2 * [strides]
    )

    input_dimension = [input_width, input_height, input_channels, batch]
    output_dimension = [width, height, output_channels, batch]
    input_datatype = f"DataType.{input_dtype}"
    output_datatype = f"DataType.{output_dtype}"

    return {
        "device": f"VPUDevice.VPU_2_0",
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "input_datatype": input_datatype,
        "output_datatype": output_datatype,
        "input_location": f"MemoryLocation.DRAM",
        "output_location": f"MemoryLocation.CMX",
        "output_write_tiles": 1,
    }


@pytest.mark.parametrize(
    [
        "op",
        "device",
        "batch",
        "input_channels",
        "output_channels",
        "height",
        "width",
        "kernel",
        "strides",
        "padding",
        "dtype",
    ],
    confis,
)
def testDMA(
    model,
    op,
    device,
    batch,
    input_channels,
    output_channels,
    height,
    width,
    kernel,
    strides,
    padding,
    dtype,
):

    # ============================= BASELINE =============================

    dma_wl = buildDMAWl(
        op,
        device,
        batch,
        input_channels,
        output_channels,
        height,
        width,
        kernel,
        strides,
        padding,
        dtype,
        dtype,
    )

    single_cycles = model.DMA(**dma_wl)

    # ============================= SPLIT WORKLOAD =============================
    dma_wl = buildDMAWl(
        op,
        device,
        batch,
        input_channels,
        output_channels,
        ceildiv(height, 5),
        width,
        kernel,
        strides,
        padding,
        dtype,
        dtype,
    )

    # Check that smaller workloads always return smaller cycles
    assert model.DMA(**dma_wl) <= single_cycles

    # ============================= MORE OUTPUT CHANNELS =============================

    # More channels
    dma_wl = buildDMAWl(
        op,
        device,
        batch,
        input_channels,
        4 * output_channels,
        height,
        width,
        kernel,
        strides,
        padding,
        dtype,
        dtype,
    )

    # Check that larger workloads always return more cycles
    assert model.DMA(**dma_wl) >= single_cycles

    # ============================= MORE INPUT CHANNELS =============================

    # More channels
    dma_wl = buildDMAWl(
        op,
        device,
        batch,
        4 * input_channels,
        output_channels,
        height,
        width,
        kernel,
        strides,
        padding,
        dtype,
        dtype,
    )

    # Check that larger workloads always return more cycles
    assert model.DMA(**dma_wl) >= single_cycles

    # ============================= LARGE KERNELS =============================

    dma_wl = buildDMAWl(
        op,
        device,
        batch,
        input_channels,
        output_channels,
        height,
        width,
        2 * kernel + 1,
        strides,
        padding,
        dtype,
        dtype,
    )

    # Check that larger workloads always return more cycles
    assert model.DMA(**dma_wl) >= single_cycles


if __name__ == "__main__":

    print("Generating the list of tests")

    lst = [
        value_list
        for value_list in AllPairs(
            [
                [
                    "DW_CONVOLUTION",
                    "ELTWISE",
                    "MAXPOOL",
                    "CM_CONVOLUTION",
                    "AVEPOOL",
                    "CONVOLUTION",
                ],
                ["VPU_2_0"],
                [1],
                common_channel_size,
                common_channel_size,
                common_h_w_size,
                common_h_w_size,
                [1, 3, 5],
                [1, 2],
                ["VALID", "SAME"],
                ["UINT8", "FLOAT16"],
            ],
            filter_func=is_valid_combination,
        )
    ]

    with open(test_config, "w") as fp:
        json.dump(lst, fp, indent=4)
