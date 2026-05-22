// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_TILING_PRIMITIVES_H
#define VPUNN_TILING_PRIMITIVES_H

#include <array>
#include <vector>

#include "vpu/dpu_workload.h"
#include "vpu/layer.h"
#include "vpu/optimization/split_options.h"

namespace VPUNN {

class TilingPrimitives {
public:
    /**
     * @brief Return true if it is possible to tile over H and/or W dimension.
     * If number of DPU's (inside tile) is 1 and ZTiling is in list, than nothing else except ZTiling is permitted
     * If the operation requires a max CHannel size (like 16) than HW tiling is not allowed
     *
     * @param layer the DPULayer object to optimize
     * @param options that control the tiling
     * @returns true if the layer support HWTiling
     */
    static bool isHWTilingAllowed(const DPULayer& layer, const SplitOptions& options);

    /**
     * @brief Return true if the layer require a maximum size in Z
     * ALso NOT suitable for HW tiling!
     * CM_CONV is not allowed for split, (Hw due to this presence)??
     *
     * @param layer the DPULayer object to optimize
     */
    static bool requireMaxZTile(const DPULayer& layer);

    /**
     * @brief Infers input shape from a DPU workload and its original layer. It modify the DPUWorkload
     *
     * @param wl [in, out] DPUWorkload that has to have the input computed based on output and kernel. This is a
     * workload part of the originalLayer but split on  one or more DPUs
     * @param originalLayer the DPULayer that is the source of the workloads split on one or more DPU's
     */
    static void inferInputTensorShape(DPUWorkload& wl, const DPULayer& originalLayer);

    /// @brief creates a Workload starting from layer. modifies the output tensor and shape only, input tensor remains
    /// the same, kernels remain the same. This stage does not contain a valid Input output correlation.
    static DPUWorkload createIncompleteTile(const DPULayer& layer, const std::array<unsigned int, 4>& new_shape,
                                            const std::array<unsigned int, 4>& offsets = {0, 0, 0, 0});

    /**
     * @brief Generate splits values from a range
     *
     * @param maxSplitRange the maximum split range as a power of two
     * @param maxLimit the absolute maximum split value
     * @return std::vector<unsigned int>
     */
    static std::vector<unsigned int> getSplitsFromRange(const unsigned int maxSplitRange, const unsigned int maxLimit);

    /**
     * @brief Create a Z-dimension tile workload with proper halo handling
     *
     * Creates a workload for a specific channel range within a layer, properly computing
     * the halo front/back based on the offset and number of channels produced.
     *
     * @param layer Source layer to create tile from
     * @param channels Number of channels for this tile
     * @param offset_channels Starting channel offset (channels produced by previous tiles)
     * @return DPUWorkload The created Z-dimension tile
     */
    static DPUWorkload createTileZ(const DPULayer& layer, unsigned int channels, unsigned int offset_channels);

    /**
     * @brief Check if operation should not be split over Z dimension
     *
     * These operations are typically not split in Z for various architectural reasons:
     * - CONVOLUTION, CM_CONVOLUTION: Benefit from full channel processing
     * - ELTWISE, ELTWISE_MUL: Element-wise operations don't benefit from Z splits
     * - LAYER_NORM: Requires full channel context
     *
     * @param op The operation type to check
     * @return true if operation should not be split, false otherwise
     */
    static bool isNoSplitOperation(Operation op);
};

}  // namespace VPUNN

#endif  // TILING_PRIMITIVES_H
