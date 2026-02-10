// Copyright © 2024 Intel Corporation
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
#include "vpu/validation/interface_valid_values.h"

#include "vpu/vpu_tiling_strategy.h"

namespace VPUNN {

/// @brief DPULayer class. no data  only methods on top of DPUWorkload
struct DPULayer : public DPUWorkload {
    using DPUWorkload::DPUWorkload;  ///< Using DPUWorkload constructor
public:
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
     * THIS IS OLD and is neither SOH or SOHO
     * /deprecated
     */
    std::vector<DPULayer> SOH_deprecated(unsigned int nTiles) const {
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
     * Algorithhm or principle is: split the output in (almost)equal slices, for each output slice compute(chose the
     * elements from original in tensor) the input tensor that you need in order to produce that output. The result will
     * be a list of input tensors that overlap (except if kernel =1).
     *
     * Internal slices (tiles>2), have 2 borders and may produce more than outside slices.
     * Populates also ISI strategy with CLUSTERING,
     * output_write_tiles is propagated from inputLayer  to the tiles Layers
     *
     * if output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted
     *
     * Halo info are adjusted(set to zero) only for the vertical dimension/direction. REst are just propagated.
     * OWT>1 will influence output_inbound halo foe vertical direction: like saying broadcast this tile to all tiles,
     * the output memory tensor will get big (= original full height)
     *
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>  the list of split layers. can be smaller than nTiles
     */
    std::vector<DPULayer> SOH_overlapped_inputs(unsigned int nTiles, bool force_broadcast) const {
        const unsigned int output_size = outputs[0].height();
        const unsigned int input_size = inputs[0].height();
        const auto dimKernel{Dim::Grid::H};
        const auto dim_pad_begin{Dim::TOP};
        const auto dim_pad_end{Dim::BOTTOM};

        const auto k = kernels[dimKernel];
        const auto s = strides[dimKernel];
        const auto layer_pad_begin = padding[dim_pad_begin];
        const auto layer_pad_end = padding[dim_pad_end];

        const unsigned int desired_tile_dim = ceil_division(output_size, nTiles);  // non uniform

        std::vector<DPULayer> tiles(nTiles, *this);  // initial split , just copy to nTiles

        // first tile and last tile will keep the external padding from parent
        // internal tiles (internal cuts) must have padding = 0, except when using entire input(till external edge) and
        // need more=> will take some padding from Layer extremity

        {  // this block is for variables used in for as pseudo counters
            int available_in{static_cast<int>(
                    input_size + layer_pad_begin)};  ///< positive or 0, includes padding start(but not end).
                                                     ///< Represents how much input values are still available
            int remnant_pad_begin = static_cast<int>(
                    layer_pad_begin);  ///< Represents how much padding is contained in the available_in inputs.
                                       ///< Initially all pad begin is to be used (at least for first tile). It will
                                       ///< remain positive also for second tile in the special case that first tile has
                                       ///< not entirely used the padding

            unsigned int remaining_output_to_split = output_size;
            for (unsigned int i{0}; i < tiles.size(); ++i) {
                const auto output_tile_dim{(remaining_output_to_split >= desired_tile_dim) ? desired_tile_dim
                                                                                           : remaining_output_to_split};
                const bool is_last_usefull_tile{
                        (remaining_output_to_split == output_tile_dim) ? true : false};  // is last/bottom tile

                const auto current_pad_begin{
                        remnant_pad_begin};  // take it from the one that goes towards zero after every tile

                int current_pad_end{is_last_usefull_tile ? (int)layer_pad_end : 0};

                // how much input this output needs!
                const int input_max_required_size = helper_input_dim(output_tile_dim, k, 0 /*pad=0*/, s);

                const int need_more_size =
                        input_max_required_size - available_in;  // if positive we need more space(reached end of
                                                                 // layer), we have to take from end padding of Layer

                auto input_tile = input_max_required_size -
                                  current_pad_begin;  // padding start is always used (completely), first ;

                if (need_more_size > 0) {  // will rely also on a part of end padding (but maybe not all)
                    const auto use_from_end_padding =
                            std::min(need_more_size /*take from padding*/, (int)layer_pad_end /*take all padding*/);
                    input_tile = input_tile - use_from_end_padding;

                    // end padding of tile must be adjusted to what we have taken, but only  in case is not the last one
                    if (!is_last_usefull_tile) {
                        current_pad_end = use_from_end_padding;  // not inherited but computed
                    }
                }

                if (input_tile < 0) {  // sanity
                    input_tile = 0;    // invalid
                }

                {  // change/set tile attributes/dimensions
                    auto& tile = tiles[i];
                    const unsigned int input_tile_dim = input_tile;
                    redimension_tile_H(tile, input_tile_dim, output_tile_dim);  // changes the tile
                    tile.padding[dim_pad_begin] = current_pad_begin;
                    tile.padding[dim_pad_end] = current_pad_end;

                    tile.isi_strategy = ISIStrategy::CLUSTERING;  // in order to propagate to workloads
                    if (!force_broadcast) {
                        tile.output_write_tiles = this->output_write_tiles;  // propagate from parent, does not force to
                                                                             // any particular value
                    } else {
                        tile.output_write_tiles =
                                nTiles;  // in order to propagate to workloads. Might be less in practice
                    }

                    HaloWorkload& tHalo{tile.halo};
                    // By default halo info propagates from upper layer, except  what we create here
                    // out inbound =0, no other tile writes to us, except if OWT is >1
                    tHalo.setVerticalNoHalo();
                    if (tile.output_write_tiles > 1)  // we assume here that we want broadcast for all tiles
                    {
                        tHalo.setInboundHaloVerticalForBroadcastAll(output_size, remaining_output_to_split,
                                                                    output_tile_dim);
                    }

                    {  // set SEP for this tile, reduce according to tile dim vs layer dim
                        auto& sep{tile.sep_activators};
                        if (sep.isEnabled()) {
                            const auto part{input_tile_dim};  // this tile Height
                            const auto whole{input_size};     // layer Height
                            rescale_tensor_height(sep.storage_elements_pointers, part, whole);
                            rescale_tensor_height(sep.actual_activators_input, part, whole);
                        }
                    }
                }

                {  // increment context for next tile (like i++ stage) ; last operation in for
                    const int consumed_input = static_cast<int>(output_tile_dim * s);
                    available_in = available_in - consumed_input;  // consumed at input

                    if (remnant_pad_begin > consumed_input) {  // consume also remnant padding
                        remnant_pad_begin -= consumed_input;   // assures stays >=0
                    } else {
                        remnant_pad_begin = 0;
                    }
                    remaining_output_to_split -= output_tile_dim;  // no underflow possible
                }
            }  // for each tile
        }  // variables used in for

        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);

        // in case the actual number of tiles <nTiles, limit the propagated value to number of
        // actual tiles (only in case forced broadcast is ON)
        if ((force_broadcast) && (tiles.size() < nTiles)) {
            const unsigned int n_broadcast{static_cast<unsigned int>(tiles.size())};
            std::for_each(tiles.begin(), tiles.end(), [n_broadcast](auto& t) {
                t.output_write_tiles = n_broadcast;
            });
        }

        return tiles;
    }

    /**
     * @brief Implements the SplitOverWidth (SOW) tiling strategy
     * @details In the SOW tiling strategy, activations are split across the tiles over the W dimension
     * Algorithm or principle is: split the output in (almost)equal slices, for each output slice compute(chose the
     * elements from original in tensor) the input tensor that you need in order to produce that output. The result will
     * be a list of input tensors that overlap (except if kernel =1).
     *
     * Internal slices (tiles>2), have 2 borders and may produce more than outside slices.
     * Populates also ISI strategy with CLUSTERING,
     * output_write_tiles is propagated from inputLayer  to the tiles Layers
     *
     * if output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted
     *
     * Halo info are adjusted(set to zero) only for the horizontal dimension/direction. REst are just propagated.
     * OWT>1 will influence output_inbound halo for horizontal direction: like saying broadcast this tile to all tiles,
     * the output memory tensor will get big (= original full height)
     *
     *
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>  the list of split layers. can be smaller than nTiles
     */
    std::vector<DPULayer> SOW_overlapped_inputs(unsigned int nTiles, bool force_broadcast) const {
        const unsigned int output_size = outputs[0].width();
        const unsigned int input_size = inputs[0].width();
        const auto dimKernel{Dim::Grid::W};
        const auto dim_pad_begin{Dim::LEFT};
        const auto dim_pad_end{Dim::RIGHT};

        const auto k = kernels[dimKernel];
        const auto s = strides[dimKernel];
        const auto layer_pad_begin = padding[dim_pad_begin];
        const auto layer_pad_end = padding[dim_pad_end];

        const unsigned int desired_tile_dim = ceil_division(output_size, nTiles);  // non uniform

        std::vector<DPULayer> tiles(nTiles, *this);  // initial split , just copy to nTiles

        // first tile and last tile will keep the external padding from parent
        // internal tiles (internal cuts) must have padding = 0, except when using entire input(till external edge) and
        // need more=> will take some padding from Layer extremity

        {  // this block is for variables used in for as pseudo counters
            int available_in{static_cast<int>(
                    input_size + layer_pad_begin)};  ///< positive or 0, includes padding left(but not right).
                                                     ///< Represents how much input values are still available
            int remnant_pad_begin = static_cast<int>(
                    layer_pad_begin);  ///< Represents how much padding is contained in the available_in inputs.
                                       ///< Initially all pad left is to be used (at least for first tile). It will
                                       ///< remain positive also for second tile in the special case that first tile has
                                       ///< not entirely used the padding

            unsigned int remaining_output_to_split = output_size;
            for (unsigned int i{0}; i < tiles.size(); ++i) {
                const auto output_tile_dim{(remaining_output_to_split >= desired_tile_dim) ? desired_tile_dim
                                                                                           : remaining_output_to_split};
                const bool is_last_usefull_tile{
                        (remaining_output_to_split == output_tile_dim) ? true : false};  // is last/bottom tile

                const auto current_pad_begin{
                        remnant_pad_begin};  // take it from the one that goes towards zero after every tile

                int current_pad_end{is_last_usefull_tile ? (int)layer_pad_end : 0};

                // how much input this output needs!
                const int input_max_required_size = helper_input_dim(output_tile_dim, k, 0 /*pad=0*/, s);

                const int need_more_size =
                        input_max_required_size - available_in;  // if positive we need more space(reached end of
                                                                 // layer), we have to take from right padding of Layer

                auto input_tile = input_max_required_size -
                                  current_pad_begin;  // padding start is always used (completely), first ;

                if (need_more_size > 0) {  // will rely also on a part of right padding (but maybe not all)
                    const auto use_from_right_padding =
                            std::min(need_more_size /*take from padding*/, (int)layer_pad_end /*take all padding*/);
                    input_tile = input_tile - use_from_right_padding;

                    // right padding of tile must be adjusted to what we have taken, but only  in case is not the last
                    // one
                    if (!is_last_usefull_tile) {
                        current_pad_end = use_from_right_padding;  // not inherited but computed
                    }
                }

                if (input_tile < 0) {  // sanity
                    input_tile = 0;    // invalid
                }

                {  // change/set tile attributes/dimensions
                    auto& tile = tiles[i];
                    const unsigned int input_tile_dim = input_tile;
                    redimension_tile_W(tile, input_tile_dim, output_tile_dim);  // changes the tile
                    tile.padding[dim_pad_begin] = current_pad_begin;
                    tile.padding[dim_pad_end] = current_pad_end;

                    tile.isi_strategy = ISIStrategy::CLUSTERING;  // in order to propagate to workloads

                    if (!force_broadcast) {
                        tile.output_write_tiles = this->output_write_tiles;  // propagate from parent, does not force to
                                                                             // any particular value
                    } else {
                        tile.output_write_tiles =
                                nTiles;  // in order to propagate to workloads. Might be less in practice
                    }

                    HaloWorkload& tHalo{tile.halo};
                    // By default halo info propagates from upper layer, except  what we create here
                    // out inbound =0, no other tile writes to us, except if OWT is >1
                    tHalo.setHorizontalNoHalo();
                    if (tile.output_write_tiles > 1)  // we assume here that we want broadcast for all tiles
                    {
                        tHalo.setInboundHaloHorizontalForBroadcastAll(output_size, remaining_output_to_split,
                                                                    output_tile_dim);
                    }

                    {  // set SEP for this tile, reduce according to tile dim vs layer dim
                        auto& sep{tile.sep_activators};
                        if (sep.isEnabled()) {
                            const auto part{input_tile_dim};  // this tile Height
                            const auto whole{input_size};     // layer Height
                            rescale_tensor_width(sep.storage_elements_pointers, part, whole);
                            rescale_tensor_width(sep.actual_activators_input, part, whole);
                        }
                    }
                }

                {  // increment context for next tile (like i++ stage) ; last operation in for
                    const int consumed_input = static_cast<int>(output_tile_dim * s);
                    available_in = available_in - consumed_input;  // consumed at input

                    if (remnant_pad_begin > consumed_input) {  // consume also remnant padding
                        remnant_pad_begin -= consumed_input;   // assures stays >=0
                    } else {
                        remnant_pad_begin = 0;
                    }
                    remaining_output_to_split -= output_tile_dim;  // no underflow possible
                }
            }  // for each tile
        }  // variables used in for

        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);

        // in case the actual number of tiles <nTiles, limit the propagated value to number of
        // actual tiles (only in case forced broadcast is ON)
        if ((force_broadcast) && (tiles.size() < nTiles)) {
            const unsigned int n_broadcast{static_cast<unsigned int>(tiles.size())};
            std::for_each(tiles.begin(), tiles.end(), [n_broadcast](auto& t) {
                t.output_write_tiles = n_broadcast;
            });
        }

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
     * Halo Aspects: SOK makes no changes in input halo or output halo for HW dimensions.
     * Output_inbound halo is affected by OWT (full tile broadcast), all tiles having full output tensor.
     *
     * @param nTiles number of CMX tiles
     * @param rounding the channel alignment
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> SOK(unsigned int nTiles, unsigned int rounding = 16) const {
        return SOK_Variants(nTiles, true, rounding);
    }

    std::vector<DPULayer> SOK_no_broadcast(unsigned int nTiles, unsigned int rounding = 16) const {
        return SOK_Variants(nTiles, false, rounding);  // not forcing classic SOK with broadcast
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
     * Halo Aspects: SOK makes no changes in input halo or output halo for HW dimensions.
     * Output_inbound halo is affected by OWT (in case of full tile broadcast), all tiles having full output tensor.
     *
     * @param nTiles number of CMX tiles
     * @param force_broadcast The split will be done  forcing broadcast, the output memory tensor will be larger (SOK
     * classic). OWT will be propagated from parent in case of no force broadcast, and if the parent specifies OWT it
     * will generate halo(broadcast is received).
     * @param rounding the channel alignment
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> SOK_Variants(unsigned int nTiles, bool force_broadcast,
                                       unsigned int rounding /*= 16*/) const {
        const unsigned int output_size = outputs[0].channels();

        // Round up to a multiple of 16(rounding) channels
        const unsigned int max_tile_channels{round_up(ceil_division(output_size, nTiles), rounding)};

        const auto& shape = outputs[0].get_shape();

        std::vector<DPULayer> tiles(nTiles, *this);             // initial split , just copy to nTiles
        unsigned int channels_remaining_to_split{output_size};  // counter like
        for (auto& tile : tiles) {
            const auto tile_channels{channels_remaining_to_split >= max_tile_channels ? max_tile_channels
                                                                                      : channels_remaining_to_split};
            tile.outputs[0].set_shape({shape[Dim::X], shape[Dim::Y], tile_channels, shape[Dim::B]});

            // what is input shape based on the new output?
            tile.recomputeInputTensorShape();
            if (!force_broadcast) {
                tile.isi_strategy = ISIStrategy::CLUSTERING;         // in order to propagate to workloads
                tile.output_write_tiles = this->output_write_tiles;  // propagate from parent

            } else {
                tile.isi_strategy = ISIStrategy::SPLIT_OVER_K;  // in order to propagate to workloads
                tile.output_write_tiles = nTiles;  // in order to propagate to workloads. Might be less in practice
            }

            HaloWorkload& tHalo{tile.halo};
            // By default halo info propagates from upper layer,
            // out inbound =0, no other tile writes to us, except if OWT is >1
            if (tile.output_write_tiles > 1)  // we assume here that we want broadcast for all tiles
            {
                // if OWT is >1 , it means that we want to broadcast the split to all other tiles
                //   we need to populate output_inbound halo , so that the memory tensor is equal to all output
                //   tensor of the full layer
                //  the front/back value depend on the position of the tile in the list.

                const auto prev_tiles_output_processed{output_size - channels_remaining_to_split};  // sum of prev tiles
                const auto next_tiles_output_to_process{channels_remaining_to_split -
                                                        tile_channels};  // sum of next tiles

                tHalo.output_0_inbound_halo.front = prev_tiles_output_processed;
                tHalo.output_0_inbound_halo.back = next_tiles_output_to_process;
                // memory output tensor = output_size , constant for all tiles
            } else {
                tHalo.output_0_inbound_halo.front = 0;
                tHalo.output_0_inbound_halo.back = 0;
            }

            // decrement/increment
            channels_remaining_to_split -= tile_channels;  // no underflow possible
        }  // for
        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);
        // in case the actual number of tiles <nTiles, limit the propagated value to number of
        // actual tiles (only in case forced broadcast SOK is ON)
        if ((force_broadcast) && (tiles.size() < nTiles)) {
            const unsigned int n_broadcast{static_cast<unsigned int>(tiles.size())};
            std::for_each(tiles.begin(), tiles.end(), [n_broadcast](auto& t) {
                t.output_write_tiles = n_broadcast;
            });
        }
        return tiles;
    }

    /**
     * @brief Implements the SplitOverH_with HALO inputs (ISI for NPU2.7) tiling strategy (inter tile)
     * @details In the SOH with input HALO tiling strategy, activations are split across the tiles over the H dimension
     * Compute tensors arelarger than memory tensors ( halo present at cut lines)
     * no padding at cut lines
     * Algorithm or principle is: split the output in (almost)equal slices, for each output slice compute( chose the
     * elements from original in tensor) the input tensor that you need in order to produce that output. For 2
     * consecutive split compute tensors calculate the overlap region and distribute this overlap to memory and halo
     * regions of the adjacent tensors.
     * E.g.: if overlap is 4, tensors will take 2 in their memory, teh remaining 2 will treat as halo regions T1: takes
     * 2 in memory + 2 as halo (from T2)
     *
     * Internal slices (tiles>2), have 2 borders and may produce more than outside slices.
     * Populates also ISI strategy with SOH,
     * output_write_tiles is propagated from inputLayer  to the tiles Layers
     *
     * @Todo: rework:  if output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted
     *
     * @Todo: rework:
     * Halo info are adjusted(set to zero) only for the vertical dimension/direction. REst are just propagated.
     * OWT>1 will influence output_inbound halo foe vertical direction: like saying broadcast this tile to all tiles,
     * the output memory tensor will get big (= original full height)
     *
     *
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>  the list of split layers. can be smaller than nTiles
     */
    std::vector<DPULayer> SOH_HALO_inputs(unsigned int nTiles) const {
        const unsigned int output_size = outputs[0].height();
        const int input_size{static_cast<int>(inputs[0].height())};
        const auto dimKernel{Dim::Grid::H};
        const auto dim_pad_begin{Dim::TOP};
        const auto dim_pad_end{Dim::BOTTOM};

        const auto k = kernels[dimKernel];
        const auto s = strides[dimKernel];
        const auto layer_pad_begin = padding[dim_pad_begin];
        const auto layer_pad_end = padding[dim_pad_end];

        const unsigned int desired_tile_dim = ceil_division(output_size, nTiles);  // non uniform

        std::vector<DPULayer> tiles(nTiles, *this);  // initial split , just copy to nTiles

        // first tile and last tile will keep the external padding from parent
        // internal tiles (internal cuts) must have padding = 0, except when using entire input(till external edge) and
        // need more=> will take some padding from Layer extremity

        {  // this block is for variables used in for as pseudo counters
            int available_in{static_cast<int>(
                    input_size + layer_pad_begin)};  ///< positive or 0, includes padding start(but not end).
                                                     ///< Represents how much input values are still available

            int prevTile_memoryEnd{-input_size - 1};  //< where the prev  layer ends its memory, rest is halo for it.

            int remnant_pad_begin = static_cast<int>(
                    layer_pad_begin);  ///< Represents how much padding is contained in the available_in inputs.
                                       ///< Initially all pad begin is to be used (at least for first tile). It will
                                       ///< remain positive also for second tile in the special case that first tile has
                                       ///< not entirely used the padding

            unsigned int remaining_output_to_split = output_size;
            for (unsigned int i{0}; i < tiles.size(); ++i) {
                const auto output_tile_dim{(remaining_output_to_split >= desired_tile_dim) ? desired_tile_dim
                                                                                           : remaining_output_to_split};
                const int consumed_input = static_cast<int>(output_tile_dim * s);  // by this tile

                const bool is_last_usefull_tile{
                        (remaining_output_to_split == output_tile_dim) ? true : false};  // is last/bottom tile

                const auto current_pad_begin{
                        remnant_pad_begin};  // take it from the one that goes towards zero after every tile

                int current_pad_end{is_last_usefull_tile ? (int)layer_pad_end : 0};

                // how much input this output needs!
                const int input_max_required_size = helper_input_dim(output_tile_dim, k, 0 /*pad=0*/, s);

                const int need_more_size =
                        input_max_required_size - available_in;  // if positive we need more space(reached end of
                                                                 // layer), we have to take from end padding of Layer

                auto input_tile =
                        input_max_required_size - current_pad_begin;  // padding start is always used (completely),
                                                                      // first ;(negative not allowed/treated)

                const int compute_start = -available_in;
                const int compute_end = compute_start + input_max_required_size - 1;  //

                const int memory_start{
                        std::max(prevTile_memoryEnd + 1,
                                 compute_start)};  // no need to have memory before compute(), maybe due to big stride
                const auto input_halo_top{(current_pad_begin == 0) ? (memory_start - compute_start)
                                                                   : 0};  // should not be negative

                if (need_more_size > 0) {  // will rely also on a part of end padding (but maybe not all)
                    const auto use_from_end_padding =
                            std::min(need_more_size /*take from padding*/, (int)layer_pad_end /*take all padding*/);
                    input_tile = input_tile - use_from_end_padding;

                    // end padding of tile must be adjusted to what we have taken, but only  in case is not the last one
                    if (!is_last_usefull_tile) {
                        current_pad_end = use_from_end_padding;  // not inherited but computed
                    }
                }

                int input_halo_bottom = 0;

                const int next_tile_compute_start = compute_start + consumed_input;
                if (current_pad_end == 0 && !is_last_usefull_tile) {  // not last tile
                    const int overlap = compute_end - next_tile_compute_start + 1;
                    if (overlap > 0) {
                        const int take_as_memory = ceil_division(overlap, 2);
                        input_halo_bottom = overlap - take_as_memory;  // rest is halo
                    }
                }
                const int thisTile_memoryEnd = compute_end - input_halo_bottom - current_pad_end;

                if (input_tile < 0) {  // sanity
                    input_tile = 0;    // invalid
                }

                {  // change/set tile attributes/dimensions
                    auto& tile = tiles[i];
                    const unsigned int input_tile_dim = input_tile;
                    redimension_tile_H(tile, input_tile_dim, output_tile_dim);  // changes the tile
                    tile.padding[dim_pad_begin] = current_pad_begin;
                    tile.padding[dim_pad_end] = current_pad_end;
                    tile.isi_strategy = ISIStrategy::SPLIT_OVER_H;       // in order to propagate to workloads
                    tile.output_write_tiles = this->output_write_tiles;  // in order to propagate to workloads.

                    HaloWorkload& tHalo{tile.halo};
                    // By default halo info propagates from upper layer, except  what we create here
                    // out inbound =0, no other tile writes to us, except if OWT is >1
                    tHalo.setVerticalNoHalo();
                    tHalo.input_0_halo.top = input_halo_top;
                    tHalo.input_0_halo.bottom = input_halo_bottom;
                    if (tile.output_write_tiles > 1)  // we assume here that we want broadcast for all tiles
                    {
                        tHalo.setInboundHaloVerticalForBroadcastAll(output_size, remaining_output_to_split,
                                                                    output_tile_dim);
                    }
                }
                {  // todo: handle SEP aspects like in SOHO
                }

                {  // increment context for next tile (like i++ stage)

                    available_in = available_in - consumed_input;  // consumed at input

                    if (remnant_pad_begin > consumed_input) {  // consume also remnant padding
                        remnant_pad_begin -= consumed_input;   // assures stays >=0
                    } else {
                        remnant_pad_begin = 0;
                    }
                    remaining_output_to_split -= output_tile_dim;  // no underflow possible
                    prevTile_memoryEnd = thisTile_memoryEnd;
                }
            }  // for each tile
        }  // variables used in for

        // Remove tiles that are of zero size
        remove_empty_tiles(tiles);
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

    void redimension_tile_H(DPULayer& tile, const unsigned int input_tile_dim,
                            const unsigned int output_tile_dim) const {
        const auto& output_shape = outputs[0].get_shape();
        const auto& input_shape = inputs[0].get_shape();
        // Set input and output shape, specific for H
        tile.outputs[0].set_shape({output_shape[Dim::X], output_tile_dim, output_shape[Dim::Z], output_shape[Dim::B]});
        tile.inputs[0].set_shape({input_shape[Dim::X], input_tile_dim, input_shape[Dim::Z], input_shape[Dim::B]});
    }

    void redimension_tile_W(DPULayer& tile, const unsigned int input_tile_dim,
                            const unsigned int output_tile_dim) const {
        const auto& output_shape = outputs[0].get_shape();
        const auto& input_shape = inputs[0].get_shape();
        // Set input and output shape, specific for W
        tile.outputs[0].set_shape({output_tile_dim, output_shape[Dim::Y], output_shape[Dim::Z], output_shape[Dim::B]});
        tile.inputs[0].set_shape({input_tile_dim, input_shape[Dim::Y], input_shape[Dim::Z], input_shape[Dim::B]});
    }

    void rescale_tensor_height(WHCBTensorShape& tensor, const DimType part, const DimType whole) const {
        const DimType newRawHeight{ceil_division(tensor.height() * part, whole)};  // h*part/whole
        const DimType newHeight{newRawHeight > 0 ? newRawHeight : 1};              // at least 1

        const WHCBTensorShape newSep{tensor.width(), newHeight, tensor.channels(), tensor.batches()};
        tensor = newSep;
    }

    void rescale_tensor_width(WHCBTensorShape& tensor, const DimType part, const DimType whole) const {
        const DimType newRawWidth{ceil_division(tensor.width() * part, whole)};  // w*part/whole
        const DimType newWidth{newRawWidth > 0 ? newRawWidth : 1};               // at least 1

        const WHCBTensorShape newSep{newWidth, tensor.height(), tensor.channels(), tensor.batches()};
        tensor = newSep;
    }

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
             std::array<unsigned int, 4> padding);

    /// @brief ctor from base class
    explicit DPULayer(const DPUWorkload& wl);

    /**
     * @brief Split a DPULayer across N CMX tiles
     *
     * @param strategy the VPUTilingStrategy to implement
     * @param nTiles number of CMX tiles
     * @return std::vector<DPULayer>
     */
    std::vector<DPULayer> splitAcrossTiles(VPUTilingStrategy strategy, unsigned int nTiles = 1) const {
        //@todo: all strategies should fill in also the HALO info field!
        switch (strategy) {
        case VPUTilingStrategy::NONE: {  // clustering
            return clustering(nTiles);
        }
        case VPUTilingStrategy::SOH_Overlapped: {  // old VPUTilingStrategy::SOH
            return SOH_overlapped_inputs(nTiles, false);
        }
        case VPUTilingStrategy::SOHO_K_SWITCH: {         // known as HKSwitch = SOHO with broadcast
            return SOH_overlapped_inputs(nTiles, true);  //
        }
        case VPUTilingStrategy::SOK: {
            return SOK(nTiles);
        }
        case VPUTilingStrategy::SOK_NO_BROADCAST: {
            return SOK_no_broadcast(nTiles);
        }
        case VPUTilingStrategy::SOH_HaloRead: {  // split by dividing by 2 , no other alignment
            return SOH_HALO_inputs(nTiles);
        }
        case VPUTilingStrategy::SOW: {  // same as SOH_Overlapped, but for W dimension
            return SOW_overlapped_inputs(nTiles, false);
        }
        case VPUTilingStrategy::SOK_H_SWITCH:
        case VPUTilingStrategy::SOK_W_SWITCH:
        case VPUTilingStrategy::SOHW:
        case VPUTilingStrategy::SOHK:
        // case VPUTilingStrategy::SOHO_K_SWITCH:  // not handled for now, SOH_Overlapped + Output full broadcast @TODO
        case VPUTilingStrategy::SOH_K_SWITCH:  // not handled for now SOH_HaloRead + Output full broadcast
        default: {
            Logger::error() << "Unsupported Layer split strategy!" << static_cast<int>(strategy) << " : "
                            << VPUTilingStrategy_ToText.at(static_cast<int>(strategy)) << ". No split performed";
            std::vector<DPULayer> no_split;
            return no_split;
            // return clustering(nTiles);
        }
        }
    }

    /*@brief Only for assuming Layer strategy for Sanity Checks. Do not use to actually populate created splits.
     * Should become DEPRECATED soon
     */
    static ISIStrategy mapTilingStrategiesToWorkload(VPUTilingStrategy strategy) {
        switch (strategy) {
        case VPUTilingStrategy::NONE:
        case VPUTilingStrategy::SOW:
            return ISIStrategy::CLUSTERING;

        case VPUTilingStrategy::SOH_Overlapped:  // Checks it has the H to be split
        case VPUTilingStrategy::SOHO_K_SWITCH:
            return ISIStrategy::CLUSTERING;

        case VPUTilingStrategy::SOH_HaloRead:  // this is still a H split, check the rules
            return ISIStrategy::SPLIT_OVER_H;  // still does a split so layer checks make sense (no CLustering) , halo
                                               // in

        case VPUTilingStrategy::SOK:
        case VPUTilingStrategy::SOK_NO_BROADCAST:
            return ISIStrategy::SPLIT_OVER_K;

        case VPUTilingStrategy::SOHK:
        case VPUTilingStrategy::SOHW:
        case VPUTilingStrategy::SOH_K_SWITCH:
        default:
            Logger::error() << "Unsupported NPU strategy!";
            return ISIStrategy::CLUSTERING;
        }
    }

    /**
     * @brief The memory footprint of the input tensors
     *
     * @return unsigned int
     */
    unsigned int input_footprint(const IDeviceValidValues& config) const {
        const DPUOperation w(*this);
        const auto& operation_behaviour = config.get_specific_behaviour(op);
        const auto movable_size{operation_behaviour.input_0_contiguous_size_bytes(config, w)};
        return static_cast<unsigned int>(movable_size);
    }

    /**
     * @brief The memory footprint of the output tensors
     *
     * @return unsigned int
     */
    unsigned int output_footprint(const IDeviceValidValues& config) const {
        const DPUOperation w(*this);
        const auto& operation_behaviour = config.get_specific_behaviour(op);
        const auto movable_size{operation_behaviour.output_0_contiguous_size_bytes(config, w)};
        return static_cast<unsigned int>(movable_size);
    }

    /**
     * @brief The memory footprint of the weights, without end cmx alignment(16Kb normally)
     * @todo: this might be wrong, review
     *
     * @param config a good configuration(rules & behaviors) according to the device of the layer.
     * @return unsigned int bytes
     */
    unsigned int weight_footprint(const IDeviceValidValues& config) const {
        DPUOperation w(*this, config);
        const auto& operation_behaviour = config.get_specific_behaviour(op);
        const auto in_1_movable_size{operation_behaviour.input_1_contiguous_size_bytes(config, w)};

        return static_cast<unsigned int>(in_1_movable_size);
    }

    /**
     * @brief Layer total memory footprint
     *
     * @return unsigned int
     */
    unsigned int footprint(const IDeviceValidValues& config) const {
        return input_footprint(config) + output_footprint(config) + weight_footprint(config);
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
        const VPUTensor inputTensor{{input_width, input_height, input_channel, input_batch}, inputs[0]};
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

    // layer information section
    using DPUWorkload::get_layer_info;  // make visible at this level
    using DPUWorkload::set_layer_info;  // make visible at this level

    std::string get_layer_name() const {  // alias
        return DPUWorkload::get_layer_info();
    }

    // use the accessors for accessing this information
    std::string get_compiler_pass() const {
        return compiler_pass;
    }
    void set_compiler_pass(const std::string& compiler_pass_info) {
        compiler_pass = compiler_pass_info;
    }
    std::string compiler_pass{""};  ///< The name of the compiler pass that generated this layer

};  // DPULAyer end

}  // namespace VPUNN

#endif  // VPUNN_LAYER_H
