# Copyright © 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.


from vpunn.cost import VPUCostModel
from allpairspy import AllPairs
import pytest
import json
import os

this_file_path = os.path.dirname(os.path.abspath(__file__))
test_config = os.path.join(this_file_path, "wl_selection_list.json")

with open(test_config) as fp:
    confis = json.load(fp)


def ceildiv(a, b):
    return -(a // -b)


@pytest.fixture(scope="module")
def model():
    return VPUCostModel("models/vpu_2_0.vpunn")


def buildWl(
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
    execution_mode,
):

    activation = "NONE"
    act_sparsity = 0
    param_sparsity = 0
    swizzling = 0
    output_write_tiles = 1

    # MPE mode
    execution_mode = execution_mode if input_dtype == "UINT8" else "VECTOR_FP16"
    # Kernel for eltwise
    kernel = kernel if op.upper() != "ELTWISE" else 1

    # Check same channels for non conv
    if (op != "CONVOLUTION") and input_channels != output_channels:
        input_channels = output_channels

    padding = 0 if padding == "VALID" else kernel // 2

    return {
        "device": f"VPUDevice.{device.upper()}",
        "operation": f"Operation.{op.upper()}",
        "input_0_width": width,
        "input_0_height": height,
        "input_0_channels": input_channels,
        "input_0_batch": batch,
        "output_0_channels": output_channels,
        "input_0_datatype": f"DataType.{input_dtype}",
        "output_0_datatype": f"DataType.{output_dtype}",
        "activation_function": f"ActivationFunction.{activation.upper()}",
        "execution_order": f"ExecutionMode.{execution_mode}",
        "kernel_height": kernel,
        "kernel_width": kernel,
        "kernel_stride_height": strides,
        "kernel_stride_width": strides,
        "kernel_pad_top": padding,
        "kernel_pad_bottom": padding,
        "kernel_pad_left": padding,
        "kernel_pad_right": padding,
        "input_sparsity_enabled": act_sparsity>0.0,
        "output_sparsity_enabled": False,
        "weight_sparsity_enabled": param_sparsity>0.0,
        "input_sparsity_rate": act_sparsity,
        "weight_sparsity_rate": param_sparsity,
        "input_0_swizzling": f"Swizzling.KEY_{swizzling}",
        "input_1_swizzling": f"Swizzling.KEY_{swizzling}",
        "output_0_swizzling": f"Swizzling.KEY_{swizzling}",
        "output_write_tiles": output_write_tiles,
        "input_0_layout": f"Layout.ZXY",
        "input_1_layout": f"Layout.ZXY",
        "output_0_layout": f"Layout.ZXY",
        "isi_strategy": f"ISIStrategy.CLUSTERING",
    }



# Check if it is a valid combination
def is_valid_combination(row):
    # row passing here can be incomplete
    if len(row) > 9:
        op = row[0]
        input_channels = row[3]
        output_channels = row[4]
        kernel = row[7]
        strides = row[8]
        # Remove invalid workloads
        if ("POOL" in op) and (
            (kernel == 1 and strides == 1) or input_channels != output_channels
        ):
            return False
        if op == "ELTWISE" and (
            kernel > 1 or strides > 1 or input_channels != output_channels
        ):
            return False

    return True


# Common dimension sizes
range16 = list(range(1, 16 + 1))

common_h_w_size = list(
    set.union(
        {idx for idx in range16},
        {448 // (2**idx) for idx in range(5)},
        # {416//(2**idx) for idx in range(5)}
    )
)
common_channel_size = list(
    set.union(
        {16 * idx for idx in range16},
        # {64*idx for idx in range16}
    )
)


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
        "execution_mode",
    ],
    confis,
)
def testWlSplitsVPU_2_0(
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
    execution_mode,
):

    # ============================= BASELINE =============================

    wl = buildWl(
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
        execution_mode,
    )

    try:
        single_wl_cycles = model.DPU(**wl)
    except ValueError as e:
        if "workload doesn't fit in cmx" in str(e):
            return
        else:
            raise

    if op in ["DW_CONVOLUTION", "ELTWISE", "MAXPOOL"]:
        return

    # ============================= SPLIT WORKLOAD =============================
    split_wl = buildWl(
        op,
        device,
        batch,
        input_channels,
        output_channels,
        ceildiv(height, 5),
        width,
        ceildiv(kernel, 5),
        strides,
        padding,
        dtype,
        dtype,
        execution_mode,
    )

    try:
        # Check that smaller workloads always return smaller cycles
        assert model.DPU(**split_wl) <= single_wl_cycles
    except ValueError as e:
        if "workload doesn't fit in cmx" in str(e):
            return
        else:
            raise
    # ============================= MORE OUTPUT CHANNELS =============================

    if op != "CM_CONVOLUTION":
        # More channels
        larger_wl = buildWl(
            op,
            device,
            batch,
            4 * input_channels,
            4 * output_channels,
            height,
            width,
            kernel,
            strides,
            padding,
            dtype,
            dtype,
            execution_mode,
        )

        try:
            # Check that larger workloads always return more cycles
            assert model.DPU(**larger_wl) >= single_wl_cycles
        except ValueError as e:
            if "workload doesn't fit in cmx" in str(e):
                return
            else:
                raise
    # ============================= LARGE KERNELS =============================

    more_kernel_wl = buildWl(
        op,
        device,
        batch,
        input_channels,
        output_channels,
        2 * height + 1,
        2 * width + 1,
        2 * kernel + 1,
        strides,
        padding,
        dtype,
        dtype,
        execution_mode,
    )

    try:
        # Check that larger workloads always return more cycles
        assert model.DPU(**more_kernel_wl) >= single_wl_cycles
    except ValueError as e:
        if "workload doesn't fit in cmx" in str(e):
            return
        else:
            raise


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
                ["MATRIX", "VECTOR"],
            ],
            filter_func=is_valid_combination,
        )
    ]

    with open(test_config, "w") as fp:
        json.dump(lst, fp, indent=4)
