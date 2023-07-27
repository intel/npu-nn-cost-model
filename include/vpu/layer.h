// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_H
#define VPUNN_LAYER_H

#include <cmath>
#include "core/logger.h"
#include "vpu/types.h"
#include "vpu/utils.h"

namespace VPUNN {

/// @brief VPU tiling strategy. How to split a Layer on multiple tiles
enum class VPUTilingStrategy { NONE, SOH, SOK, SOW, SOHW, SOHK, __size };
static const EnumMap VPUTilingStrategy_ToText{
        link(VPUTilingStrategy::NONE, "NONE"), link(VPUTilingStrategy::SOH, "SOH"),
        link(VPUTilingStrategy::SOK, "SOK"),   link(VPUTilingStrategy::SOW, "SOW"),
        link(VPUTilingStrategy::SOHW, "SOHW"), link(VPUTilingStrategy::SOHK, "SOHK"),
};
template <>
inline const EnumMap& mapToText<VPUTilingStrategy>() {
    return VPUTilingStrategy_ToText;
}

/// @brief DPULayer class. no data  only methods on top of DPUWorkload
struct DPULayer : public DPUWorkload {
    using DPUWorkload::DPUWorkload;  ///< Using DPUWorkload constructor

    /**
     * @brief Implements the Clustering tiling strategy (inter tile)
     * @details In the clustering tiling strategy, both activations and weights are fully replicated in all tiles
     * isi_strategy set to clustering
     * and output_write_tiles are propagated from input
     *
     * @param nTiles number of dpu tiles
     * @return std::vector<DPULayer> the list of layers
     */
    std::vector<DPULayer> clustering(unsigned int nTiles) const {
        std::vector<DPULayer> tiles(nTiles, *this);  // initial split

        // ensure that  the Layer/workloads in each tile is marked as clustering and output tiles =1
        for (auto& tile : tiles) {
            tile.isi_strategy = ISIStrategy::CLUSTERING;         // in order to propagate to workloads
            tile.output_write_tiles = this->output_write_tiles;  // in order to propagate to workloads. CLUSTERING  does
                                                                 // not force to any particular value
        }
        return tiles;
    }

    /**
     * @brief Implements the SplitOverH (SOH) tiling strategy (inter tile)
     * @details In the SOH tiling strategy, activations are split across the tiles over the H dimension
     * todo: Cut lines must be without padding
     * todo: compute tensors have a halo where the cut is to indicate how many lines are taken from the other tile
     * The weights are fully replicated in all tiles
     * Populates also ISI strategy with SOH
     * output_write_tiles is propagated from inputLayer  to the tiles Layers
     *
     *if output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>  the list of split layers. can be smaller than nTiles
     */
    std::vector<DPULayer> SOH(unsigned int nTiles) const {
        std::vector<DPULayer> tiles(nTiles, *this);  // initial split
        unsigned int height_remaining_to_split = outputs[0].height();
        const unsigned int max_tile_height = ceil_division(height_remaining_to_split, nTiles);
        const auto& output_shape = outputs[0].get_shape();
        const auto& input_shape = inputs[0].get_shape();

        for (auto& tile : tiles) {
            const auto output_tile_height{height_remaining_to_split >= max_tile_height ? max_tile_height
                                                                                       : height_remaining_to_split};
            height_remaining_to_split -= output_tile_height;  // no underflow possible

            const auto input_tile_height{helper_input_dim(output_tile_height, kernels[Dim::Grid::H],
                                                          padding[Dim::Padding::TOP] + padding[Dim::Padding::BOTTOM],
                                                          strides[Dim::Grid::H])};

            // Set input and output shape
            tile.outputs[0].set_shape({output_shape[0], output_tile_height, output_shape[2], output_shape[3]});
            tile.inputs[0].set_shape({input_shape[0], input_tile_height, input_shape[2], input_shape[3]});

            tile.isi_strategy = ISIStrategy::SPLIT_OVER_H;       // in order to propagate to workloads
            tile.output_write_tiles = this->output_write_tiles;  // in order to propagate to workloads. SOH does not
                                                                 // force to any particular value
        }
        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);

        return tiles;
    }
    /**
     * @brief Implements the SplitOverH_OVERLAPPED (SOH) tiling strategy (inter tile)
     * @details In the SOHOVERLAPPED tiling strategy, activations are split across the tiles over the H dimension
     * Compute tensors are the same as memory tensors (no halo at cut lines)
     * no padding at cut lines
     * Internal slices (tiles>2), have 2 borders and may produce more than outside slices.
     * Populates also ISI strategy with CLUSTERING,
     * output_write_tiles is propagated from inputLayer  to the tiles Layers
     *
     * if output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>  the list of split layers. can be smaller than nTiles
     */
    std::vector<DPULayer> SOH_overlapped(unsigned int nTiles) const {
        const unsigned int output_size = outputs[0].height();
        const unsigned int input_size = inputs[0].height();
        const auto dimKernel{Dim::Grid::H};
        const auto pad_begin{Dim::TOP};
        const auto pad_end{Dim::BOTTOM};

        const auto k = kernels[dimKernel];
        const auto s = strides[dimKernel];
        const auto layer_pad_begin = padding[pad_begin];
        const auto layer_pad_end = padding[pad_end];

        const unsigned int desired_tile_dim = ceil_division(output_size, nTiles);  // non uniform

        std::vector<DPULayer> tiles(nTiles, *this);  // initial split , just copy to nTiles

        // first tile will keep the external padding from parent
        // last tile will keep the external padding from parent
        // internal tiles (internal cuts) must have padding = 0

        int available_in = input_size + (int)(padding[pad_begin]);  // positive or 0

        unsigned int remaining_to_split = output_size;
        for (size_t i = 0; i < tiles.size(); ++i) {
            const bool first_pad_inherited{(i == 0) ? true : false};  // is top tile
            const auto output_tile_dim{(remaining_to_split >= desired_tile_dim) ? desired_tile_dim
                                                                                : remaining_to_split};
            remaining_to_split -= output_tile_dim;                                      // no underflow possible
            const bool second_pad_inherited{(remaining_to_split == 0) ? true : false};  // is last/bottom tile

            auto& tile = tiles[i];
            // set correct padding of this tile based on position vs borders
            tile.padding[pad_begin] = first_pad_inherited ? layer_pad_begin : 0;
            tile.padding[pad_end] = second_pad_inherited ? layer_pad_end : 0;

            // ho much input this output needs!
            const int input_max_required_size = helper_input_dim(output_tile_dim, k, 0 /*pad=0*/, s);

            const int need_more_size =
                    input_max_required_size -
                    available_in;  // if positive we need more space, we have to take from second padding

            auto input_tile = input_max_required_size - (int)tile.padding[pad_begin];  // padding start is used first;
            if (need_more_size > 0) {  // will rely also on a part of second padding (but maybe not all)
                input_tile = input_tile - std::min(need_more_size /*take from padding*/,
                                                   (int)tile.padding[pad_end] /*take all padding*/);
            }

            if (input_tile < 0) {
                input_tile = 0;
            }

            const unsigned int input_tile_dim = input_tile;

            {
                const auto& output_shape = outputs[0].get_shape();
                const auto& input_shape = inputs[0].get_shape();
                // Set input and output shape, specific for H
                tile.outputs[0].set_shape(
                        {output_shape[Dim::X], output_tile_dim, output_shape[Dim::Z], output_shape[Dim::B]});
                tile.inputs[0].set_shape(
                        {input_shape[Dim::X], input_tile_dim, input_shape[Dim::Z], input_shape[Dim::B]});
            }
            tile.isi_strategy = ISIStrategy::CLUSTERING;         // in order to propagate to workloads
            tile.output_write_tiles = this->output_write_tiles;  // in order to propagate to workloads. SOH overlapped
                                                                 // does not force to any particular value

            available_in = available_in - output_tile_dim * s;  // consumed at input
        }

        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);
        return tiles;
    }

    /**
     * @brief Implements the SplitOverK (SOK) tiling strategy
     * @details In the SOK tiling strategy, weights are split across the tiles over the K dimension.
     * The DPU in each tile compute a K-slice of the output tensors and then broadcast the result in each
     * CMX tile, implicitly concatenating the results and having then all activations completely replicated
     *
     * Populates also ISI strategy with SOK
     * output_write_tiles is set to actual nTiles
     * if output tiles are less than nTiles output_write_tiles are adjusted to actual output tiles
     *
     * @param nTiles number of CMX tiles
     * @param rounding the channel alignment
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> SOK(unsigned int nTiles, unsigned int rounding = 16) const {
        std::vector<DPULayer> tiles(nTiles, *this);
        unsigned int channels_remaining_to_split = outputs[0].channels();
        // Round up to a multiple of 16 channels
        const unsigned int max_tile_channels{round_up(ceil_division(channels_remaining_to_split, nTiles), rounding)};

        const auto& shape = outputs[0].get_shape();
        for (auto& tile : tiles) {
            const auto tile_channels{channels_remaining_to_split >= max_tile_channels ? max_tile_channels
                                                                                      : channels_remaining_to_split};
            channels_remaining_to_split -= tile_channels;  // no underflow possible

            tile.outputs[0].set_shape({shape[0], shape[1], tile_channels, shape[3]});

            // what is input shape based on the new output?
            tile.recomputeInputTensorShape();

            tile.isi_strategy = ISIStrategy::SPLIT_OVER_K;  // in order to propagate to workloads
            tile.output_write_tiles = nTiles;  // in order to propagate to workloads. Might mbe less in practice
        }
        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);
        // in case the actual number of tiles <nTiles, limit the propagated value to number of
        // actual tiles
        if (tiles.size() < nTiles) {
            const unsigned int n_broadcast{static_cast<unsigned int>(tiles.size())};
            std::for_each(tiles.begin(), tiles.end(), [n_broadcast](auto& t) {
                t.output_write_tiles = n_broadcast;
            });
        }
        return tiles;
    }

protected:
    /// @brief remove tiles that are of zero size (output size is checked )
    void remove_empty_tiles(std::vector<DPULayer>& tiles_list) const {
        tiles_list.erase(std::remove_if(std::begin(tiles_list), std::end(tiles_list),
                                        [](const DPULayer& layer) {
                                            return layer.outputs[0].size() == 0;
                                        }),
                         std::end(tiles_list));
    };

public:
    /**
     * @brief Construct a new DPULayer object
     *
     * @param device VPUDevice
     * @param op DPULayer Operation
     * @param inputs input tensors (activations)
     * @param outputs output tensor
     * @param kernels kernel sizes
     * @param strides kernel strides
     * @param padding operation padding
     */
    DPULayer(VPUDevice device, Operation op, std::array<VPUTensor, 1> inputs, std::array<VPUTensor, 1> outputs,
             std::array<unsigned int, 2> kernels, std::array<unsigned int, 2> strides,
             std::array<unsigned int, 4> padding) {
        this->device = device;
        this->op = op;
        this->inputs = inputs;
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

        // rest of fields are the workload default (CLUSTERING, output write tiles  =1)
    }

    /// @brief ctor from base class
    explicit DPULayer(const DPUWorkload& wl): DPUWorkload(wl) {
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

        // rest of fields are the workload default (CLUSTERING, output write tiles  =1)
    }

    /**
     * @brief Split a DPULayer across N CMX tiles
     *
     * @param strategy the VPUTilingStrategy to implement
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> splitAcrossTiles(VPUTilingStrategy strategy, unsigned int nTiles = 1) const {
        switch (strategy) {
        case VPUTilingStrategy::SOH:
            return SOH(nTiles);
            // return SOH_overlapped(nTiles);  // make it  SOH Overlapped (also for VPU 2.0?)
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

    static ISIStrategy mapTilingStrategiesToWorkload(VPUTilingStrategy strategy) {
        switch (strategy) {
        case VPUTilingStrategy::NONE:
            return ISIStrategy::CLUSTERING;
        case VPUTilingStrategy::SOH:
            return ISIStrategy::SPLIT_OVER_H;
        case VPUTilingStrategy::SOK:
            return ISIStrategy::SPLIT_OVER_K;

        case VPUTilingStrategy::SOHK:
        case VPUTilingStrategy::SOW:
        case VPUTilingStrategy::SOHW:
        default:
            Logger::error() << "Unsupported VPU4 strategy!";
            return ISIStrategy::CLUSTERING;
        }
    }

    /**
     * @brief The memory footprint of the input tensors
     *
     * @return unsigned int
     */
    unsigned int input_footprint() const {
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
    unsigned int output_footprint() const {
        unsigned long size = 0;
        for (auto& out : outputs) {
            size += out.size();
        }
        return size;
    }

    /**
     * @brief The memory footprint of the weights
     * @todo: this might be wrong, review
     *
     * @return unsigned int
     */
    unsigned int weight_footprint() const {
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
    unsigned int footprint() const {
        return input_footprint() + output_footprint() + weight_footprint();
    }

    /**
     * @brief recomputes the input based on operation and a changed output size
     * affects only WHC dimensions based on Outputs kernels, stride, padding
     */
    void recomputeInputTensorShape() {
        // const auto layout = inputs[0].get_layout();  // unchanged
        // const auto dtype = inputs[0].get_dtype();    // unchanged

        const auto input_width = (outputs[0].width() - 1) * strides[Dim::Grid::W] + kernels[Dim::Grid::W] -
                                 padding[Dim::Padding::LEFT] - padding[Dim::Padding::RIGHT];
        const auto input_height = (outputs[0].height() - 1) * strides[Dim::Grid::H] + kernels[Dim::Grid::H] -
                                  padding[Dim::Padding::TOP] - padding[Dim::Padding::BOTTOM];

        // the workload can be a result of HW split or Z split
        // for HW split the input channels remain unaffected (operation irrelevant)
        //
        // for Z split: the output (of 1 DPU workload) has only a fraction of the original z. 
        // For the operations that use the full input channels in their kernel (e.g. CONV, CM_CONV)  
        // - the input's Z should be again the full original input channels, there was no split of input Z
        // For the operation that are not having a kernel with depth (e.g. ELEMENTWISE)
        // - the input's Z should be equal to output's Z  (as a more general rule)
        const auto input_channel =
                ((op == Operation::CONVOLUTION) || (op == Operation::CM_CONVOLUTION))  // kernels need all input Z
                        ? inputs[0].z()    // use what the split left here by default (maybe original z, maybe a Z
                                           // split of inputs(future?))
                        : outputs[0].z();  // for elementwise operations the in-out channels should match

        const auto input_batch = inputs[0].batches();  // unchanged

        // create a new tensor only with shape changed!
        const auto inputTensor = VPUTensor({input_width, input_height, input_channel, input_batch}, inputs[0]);
        inputs[0] = inputTensor;
    }

    /// @brief enables /disables the w sparsity and sets the value. Only combinations that are allowed
    /// when enabling sparsity_value is limited to [0.0, 1.0]
    /// when disabling , sparsity will be set to zero
    ///
    /// @param enabled true/false
    /// @param sparsity_value , limited to [0.0, 1.0] for enabled and to 0.0 for disabled
    void set_weight_sparsity(bool enabled, float sparsity_value) {
        weight_sparsity = sparsity_value;
        weight_sparsity_enabled = enabled;
        if (weight_sparsity_enabled) {
            weight_sparsity = std::min(std::max(0.0f, sparsity_value), 1.0f);
        } else {
            weight_sparsity = 0.0F;
        }
    }
};

/// provides differentiated information for a layer based on its content
class DPULayerModes {
private:
    /// @brief Get the valid ExecutionMode for VPU_2_0
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode_2_0(const DPULayer& wl) {
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

    /// @brief Get the valid ExecutionMode for VPU_2_7
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode_2_7(const DPULayer& wl) {
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
    DPULayerModes() = default;  // no instance possible

public:
    /// @brief Get the valid ExecutionMode for the DPULayer
    ///
    /// @param wl the DPULayer
    /// @return std::vector<ExecutionMode>
    static std::vector<ExecutionMode> getValidExecutionMode(const DPULayer& wl) {
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
};

}  // namespace VPUNN

#endif  // VPUNN_LAYER_H
