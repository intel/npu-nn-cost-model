// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_H
#define VPUNN_LAYER_H

#include "core/logger.h"
#include "vpu/performance.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpunn.h"

namespace VPUNN {

/**
 * @brief VPU tiling strategy
 *
 */
enum class VPUTilingStrategy { NONE, SOH, SOK, SOW, SOHW, SOHK, __size };

/**
 * @brief DPULayer class
 *
 */
struct DPULayer : public DPUWorkload {
    /**
     * @brief Using DPUWorkload constructor
     *
     */
    using DPUWorkload::DPUWorkload;

    /**
     * @brief Implements the Clustering tiling strategy
     * @details In the clustering tiling strategy, both activations and weights are fully replicated in all tiles
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> clustering(unsigned int nTiles) {
        return std::vector<DPULayer>(nTiles, *this);
    }

    /**
     * @brief Implements the SplitOverH (SOH) tiling strategy
     * @details In the SOH tiling strategy, activations are splitted across the tiles over the H dimension
     * The weights are fully replicated in all tiles
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> SOH(unsigned int nTiles) {
        std::vector<DPULayer> tiles(nTiles, *this);
        unsigned int height = outputs[0].height();
        unsigned int max_tile_height = ceil_division(height, nTiles);
        auto output_shape = outputs[0].get_shape();
        auto input_shape = inputs[0].get_shape();
        for (unsigned int idx = 0; idx < tiles.size(); idx++) {
            auto output_tile_height = height > max_tile_height ? max_tile_height : height;
            auto input_tile_height =
                    helper_input_dim(output_tile_height, kernels[1], padding[2] + padding[3], strides[1]);
            // Set input and output shape
            tiles[idx].outputs[0].set_shape({output_shape[0], output_tile_height, output_shape[2], output_shape[3]});
            tiles[idx].inputs[0].set_shape({input_shape[0], input_tile_height, input_shape[2], input_shape[3]});
            height -= output_tile_height;
        }
        // Remove tiles that are of zero size
        tiles.erase(std::remove_if(std::begin(tiles), std::end(tiles),
                                   [](DPULayer layer) {
                                       return layer.outputs[0].size() == 0;
                                   }),
                    std::end(tiles));
        return tiles;
    }

    /**
     * @brief Implements the SplitOverK (SOK) tiling strategy
     * @details In the SOK tiling strategy, weights are splitted across the tiles over the K dimension.
     * The DPU in each tile compute a K-slice of the output tensors and then broadcast the result in each
     * CMX tile, implicitly concatenating the results and havign then all activations completely replicated
     *
     * @param nTiles number of CMX tiles
     * @param rounding the channel alignment
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> SOK(unsigned int nTiles, unsigned int rounding = 16) {
        std::vector<DPULayer> tiles(nTiles, *this);
        unsigned int channels = outputs[0].channels();
        unsigned int max_tile_channels = ceil_division(channels, nTiles);
        // Round up to a multiple of 16 channels
        max_tile_channels = round_up(max_tile_channels, rounding);
        auto shape = outputs[0].get_shape();
        for (unsigned int idx = 0; idx < tiles.size(); idx++) {
            auto tile_channels = channels > max_tile_channels ? max_tile_channels : channels;
            tiles[idx].outputs[0].set_shape({shape[0], shape[1], tile_channels, shape[3]});
            channels -= tile_channels;
        }
        // Remove tiles that are of zero size
        tiles.erase(std::remove_if(std::begin(tiles), std::end(tiles),
                                   [](DPULayer layer) {
                                       return layer.outputs[0].size() == 0;
                                   }),
                    std::end(tiles));
        return tiles;
    }

public:
    /**
     * @brief Construct a new DPULayer object
     *
     * @param device VPUDevice
     * @param op DPULayer Operation
     * @param inputs input tensors (activations)
     * @param inputs_1 input tensors (weights, TODO)
     * @param outputs output tensor
     * @param kernels kernel sizes
     * @param strides kernel strides
     * @param padding operation padding
     */
    DPULayer(VPUDevice device, Operation op, std::array<VPUTensor, 1> inputs,
             /*std::array<VPUTensor, 1> inputs_1,*/ std::array<VPUTensor, 1> outputs,
             std::array<unsigned int, 2> kernels, std::array<unsigned int, 2> strides,
             std::array<unsigned int, 4> padding) {
        this->device = device;
        this->op = op;
        this->inputs = inputs;
        // this->inputs_1 = inputs_1;
        this->outputs = outputs;
        this->kernels = kernels;
        this->strides = strides;
        this->padding = padding;

        switch (device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
            this->execution_order = inputs[0].is_float() ? ExecutionMode::VECTOR_FP16 : ExecutionMode::MATRIX;
            break;
        case VPUDevice::VPU_2_7:
        case VPUDevice::VPU_4_0:
            this->execution_order = ExecutionMode::CUBOID_16x16;
            break;
        default:
            Logger::error() << "Invalid VPU device type";
        }
    }

    /**
     * @brief Split a DPULayer across N CMX tiles
     *
     * @param strategy the VPUTilingStrategy to implement
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> splitAcrossTiles(VPUTilingStrategy strategy, unsigned int nTiles = 1) {
        switch (strategy) {
        case VPUTilingStrategy::SOH:
            return SOH(nTiles);
        case VPUTilingStrategy::SOK:
            return SOK(nTiles);
        case VPUTilingStrategy::SOHK:
        case VPUTilingStrategy::SOW:
        case VPUTilingStrategy::SOHW:
        default:
            Logger::error() << "Unsupported VPU4 strategy!";
            return clustering(nTiles);
        }
    }

    /**
     * @brief The memory footprint of the input tensors
     *
     * @return unsigned int
     */
    unsigned int input_footprint() {
        unsigned long size = 0;
        for (auto& in : inputs) {
            size += in.size();
        }
        return size;
    }

    /**
     * @brief The memory footprint of the output tensors
     *
     * @return unsigned int
     */
    unsigned int output_footprint() {
        unsigned long size = 0;
        for (auto& out : outputs) {
            size += out.size();
        }
        return size;
    }

    /**
     * @brief The memory footprint of the weights
     *
     * @return unsigned int
     */
    unsigned int weight_footprint() {
        unsigned wt_size = dtype_to_bytes(outputs[0].get_dtype()) * multiply_vector(kernels);
        if (op == Operation::CONVOLUTION || op == Operation::CM_CONVOLUTION) {
            // Ceil division between input channels and the DPU mac
            wt_size *= (unsigned int)inputs[0].get_shape()[2];
        }

        unsigned table_size = outputs[0].get_shape()[2] * 16;
        return wt_size + table_size;
    }

    /**
     * @brief Layer total memory footprint
     *
     * @return unsigned int
     */
    unsigned int footprint() {
        return input_footprint() + output_footprint() + weight_footprint();
    }
};

/**
 * @brief Get the valid ExecutionMode for VPU_2_0
 *
 * @param wl the DPULayer
 * @return std::vector<ExecutionMode>
 */
inline std::vector<ExecutionMode> getValidExecutionMode_2_0(const DPULayer& wl) {
    // Float input or output -> ExecutionMode::VECTOR_FP16
    if (wl.inputs[0].is_float() || wl.outputs[0].is_float())
        return {ExecutionMode::VECTOR_FP16};
    // Find the optimal Execution Mode given output tensor layout
    auto shape = wl.outputs[0].get_shape();
    const double W = static_cast<double>(shape[Dim::Act::X]);
    const double H = static_cast<double>(shape[Dim::Act::Y]);
    // ExecutionMode::MATRIX process tensor using a W=4 H=4 grid, calculate grid cells count for it
    const double matrixPartsCount = std::ceil(W / 4.0) * std::ceil(H / 4.0);
    // ExecutionMode::VECTOR process tensor using a W=16 H=1 grid, calculate grid cells count for it
    const double vectorPartsCount = std::ceil(W / 16.0) * H;
    // Cells count is in direct ratio with work size, so choose smaller one
    if (vectorPartsCount <= matrixPartsCount) {
        return {ExecutionMode::VECTOR};
    }
    return {ExecutionMode::MATRIX};
}

/**
 * @brief Get the valid ExecutionMode for VPU_2_7
 *
 * @param wl the DPULayer
 * @return std::vector<ExecutionMode>
 */
inline std::vector<ExecutionMode> getValidExecutionMode_2_7(const DPULayer& wl) {
    // The available mode choice is based on the OP type
    switch (wl.op) {
    case Operation::CM_CONVOLUTION:
    case Operation::DW_CONVOLUTION:
    case Operation::AVEPOOL:
    case Operation::MAXPOOL:
        return {ExecutionMode::CUBOID_16x16};
    case Operation::ELTWISE:
        return {ExecutionMode::CUBOID_8x16};
    default:
        return {ExecutionMode::CUBOID_16x16, ExecutionMode::CUBOID_8x16, ExecutionMode::CUBOID_4x16};
    }
}

/**
 * @brief Get the valid ExecutionMode for the DPULayer
 *
 * @param wl the DPULayer
 * @return std::vector<ExecutionMode>
 */
inline std::vector<ExecutionMode> getValidExecutionMode(const DPULayer& wl) {
    switch (wl.device) {
    case VPUDevice::VPU_2_0:
    case VPUDevice::VPU_2_1:
        return getValidExecutionMode_2_0(wl);
    case VPUDevice::VPU_2_7:
    case VPUDevice::VPU_4_0:
        return getValidExecutionMode_2_7(wl);
    default:
        return {};
    }
}

}  // namespace VPUNN

#endif  // VPUNN_LAYER_H
