// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_COST_MODEL_H
#define VPUNN_LAYER_COST_MODEL_H

#include <exception>

#include "core/logger.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/layer.h"
#include "vpu/optimization/workload_optimization.h"
#include "vpu/performance.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu/validation/layer_sanitizer.h"
#include "vpu_cost_model.h"

namespace VPUNN {

/// @brief A VPU layer strategy
struct VPULayerStrategy {
    unsigned int nDPUs{1};   ///< Number of DPUs per tile
    unsigned int nSHVs{1};   ///< Number of Shaves per tile
    unsigned int nTiles{1};  ///< Number of tiles

    VPUTilingStrategy tiling_strategy{VPUTilingStrategy::NONE};  ///< tiling strategy

    bool input_fetching{false};   ///< true if the layer input is in DDR
    bool output_spilling{false};  ///< true if the layer output is in DDR

    bool prefetching{true};  ///< If layer parameters are prefetched with previous layers. If true it considers the
                             ///< weights are prefetched, if false will fetch the weights considering also sparsity
};

inline std::ostream& operator<<(std::ostream& stream, const VPULayerStrategy& d) {
    stream << "\nVPU Layer Strategy : \n"
           << " n DPUs: \t" << d.nDPUs << "\n"
           << " n SHVs: \t" << d.nSHVs << "\n"
           << " n Tiles: \t" << d.nTiles << "\n"
           << " Tiling Strategy: \t" << (int)d.tiling_strategy << " : "
           << VPUTilingStrategy_ToText.at(static_cast<int>(d.tiling_strategy)) << " ;\n"
           << " input_fetching : \t" << (int)d.input_fetching << " : " << (d.input_fetching ? "true" : "false")
           << " ;\n"  //
           << " output_spilling: \t" << (int)d.output_spilling << " : " << (d.output_spilling ? "true" : "false")
           << " ;\n"                                                                                                 //
           << " prefetching    : \t" << (int)d.prefetching << " : " << (d.prefetching ? "true" : "false") << " ;\n"  //
            ;
    return stream;
}

/// @brief The VPUNN layer cost model (also called VPUNN Level2 API)
class VPUNN_API(VPULayerCostModel): public VPUCostModel {
private:
    const LayersValidation the_layer_validator{};     ///< used for validating the un-split layers and split layers
    unsigned int maxWorkloadsPerIntraTileSplit{50U};  ///< max splits for a tile

public:
    using VPUCostModel::VPUCostModel;  ///< exposing/Using the same VPUCostModel constructor (base class)

    /// @brief limits the split of a tile (intra-tile split) to this number of individual workloads
    void set_maxWorkloadsPerIntraTileSplit(unsigned int new_value) noexcept {
        maxWorkloadsPerIntraTileSplit = new_value;
    }
    auto get_maxWorkloadsPerIntraTileSplit() const noexcept {
        return maxWorkloadsPerIntraTileSplit;
    }

    /**
     * @brief Compute the optimal cost of a DPULayer given a strategy and context
     *
     * @param layer the DPULayer
     * @param strategy the layer strategy, shaves do not matter
     * @return  measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType Layer(DPULayer& layer, VPULayerStrategy strategy) {
        return layer_cycles(layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles, strategy.input_fetching,
                            strategy.output_spilling, strategy.prefetching);
    }

    /**
     * @brief Compute the optimal cost of a DPULayer using a specific strategy and context
     *
     * It splits on tiles(between tiles, using the strategy), then, for each tile , makes the intra-tile split on
     * workloads and choses the best one
     *
     * @param layer the DPULayer
     * @param strategy the inter-tile tiling strategy to use
     * @param nDPU the number of DPU (for each tile)
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @param prefetching If true it considers the weights are prefetched, if false
     * will fetch the weights considering also sparsity
     * takes in consideration the sparsity(enabled and value)
     * @return measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType Layer(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU = 1,
                              unsigned int nTiles = 1, bool input_in_ddr = false, bool output_in_ddr = false,
                              bool prefetching = true) {
        return layer_cycles(layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr, prefetching, nullptr);
    }
    /**
     * @brief Compute the optimal cost of a DPULayer using a specific strategy and execution mode
     *
     * It splits on tiles(between tiles, using the strategy), then, for each tile , makes the intra-tile split on
     * workloads and choses the best one
     *
     * @param layer the DPULayer
     * @param strategy the inter-tile tiling strategy to use
     * @param nDPU the number of DPU (for each tile)
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX). Data fetch time is
     * computed considering the full layer input tensor not he split ones
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX). Data fetch time is
     * computed considering the full layer output tensor not he split ones
     * @param prefetching  If true it considers the weights are prefetched, if false
     * will fetch the weights considering also sparsity. Data fetch time is computed considering the split layers
     * weights tensors, that are pipelined on all available DMA channels.
     * @param detailed_split [out] gives as output the information on how was split this layer and what is the best
     * split on workloads
     * @return measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType Layer(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU, unsigned int nTiles,
                              bool input_in_ddr, bool output_in_ddr, bool prefetching, LayerSplitInfo& detailed_split) {
        return layer_cycles(layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr, prefetching, &detailed_split);
    }

    /**
     * @brief Compute the optimal cost of a pre split layer. Layer is already split on tiles, only the intratile split
     * si performed.
     *
     * For each tile , makes the intra-tile split on workloads and choses the best one
     *
     * @param layers_pre_split the list of layers split on tiles, their number indicates the tiles. Full info has to be
     * specified, as it is for a DPUWorkload
     * @param nDPU the number of DPU (for each tile)
     *
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX). Data fetch time is
     * computed considering the split layers input tensors, that are summed up.
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX). Data fetch time is
     * computed considering the split layers output tensors, that are summed up.
     * @param prefetching  If true it considers the weights are prefetched, if false
     * will fetch the weights considering also sparsity. Data fetch time is computed considering the split layers
     * weights tensors, that are pipelined on all available DMA channels.
     *
     * @param detailed_split [out] gives as output the information on how was split this layer and what is the best
     * split on workloads
     *
     * @return measured best cycles for the overall vector of layers or error code . \see Cycles for error codes
     */
    CyclesInterfaceType LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                       bool input_in_ddr, bool output_in_ddr, bool prefetching,
                                       LayerSplitInfo& detailed_split) {
        return layer_pre_split_cycles(layers_pre_split, nDPU, input_in_ddr, output_in_ddr, prefetching,
                                      &detailed_split);
    }

    /// version without detailed split output parameter.
    CyclesInterfaceType LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                       bool input_in_ddr, bool output_in_ddr, bool prefetching) {
        return layer_pre_split_cycles(layers_pre_split, nDPU, input_in_ddr, output_in_ddr, prefetching, nullptr);
    }

protected:
    /**
     * @brief Compute the optimal cost of a DPULayer using a specific strategy and execution mode
     *
     * It splits on tiles(between tiles, using the strategy), then, for each tile , makes the intra-tile split on
     * workloads and choses the best one
     *
     * @param layer the DPULayer
     * @param strategy the inter-tile tiling strategy to use
     * @param nDPU the number of DPU (for each tile)
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX). Data fetch time is
     * computed considering the full layer input tensor not he split ones
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX). Data fetch time is
     * computed considering the full layer output tensor not he split ones
     * @param prefetching  If true it considers the weights are prefetched, if false
     * will fetch the weights considering also sparsity. Data fetch time is computed considering the split layers
     * weights tensors, that are pipelined on all available DMA channels.
     * @param detailed_split [out] gives as output the information on how was split this layer and what is the best
     * split on workloads. ignored if null
     * @return measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType layer_cycles(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU = 1,
                                     unsigned int nTiles = 1, bool input_in_ddr = false, bool output_in_ddr = false,
                                     bool prefetching = true, LayerSplitInfo* detailed_split = nullptr) {
        std::vector<CyclesInterfaceType> tiles_cost;  // cost of each tile
        std::vector<DPULayer> tiles_layer;            //< layer list after split
        {
            operation_sanitisation(layer);  // AVEPOOL will be transformed to something equivalent
            const SplitOptions options{maxWorkloadsPerIntraTileSplit /*maxWorkloads*/, 0,
                                       nDPU};  // here always for LATENCY => cycles

            // Create a new instance of the cost model
            // std::shared_ptr<VPUCostModel> model{std::make_shared<VPUCostModel>(
            //         *this)};  // why? and how does this work properly since CostMOdel does not have a proper copy
            //         ctor

            {  // the layer must be verified to be valid
                SanityReport unsplit_result;
                the_layer_validator.sanitize_preconditions(
                        layer);  // this might change the layer. eg: siwzzlings for VPU2.0
                the_layer_validator.check_completeLayer_consistency(
                        layer, unsplit_result, DPULayer::mapTilingStrategiesToWorkload(strategy), nTiles);

                if (!unsplit_result.is_usable()) {
                    Logger::warning() << "\n Layer is NOT Valid \n *** INFO from LayerValidator:\n "
                                      << unsplit_result.info << "\n"
                                      << "\n *** LAYER: "
                                      << "\n " << layer << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU
                                      << ", nTiles: " << nTiles
                                      << "\n Result: Early termination with Error code: " << unsplit_result.value()
                                      << " : " << Cycles::toErrorText(unsplit_result.value()) << "\n";
                    return unsplit_result.value();  // EARLY RETURN
                }
            }

            // split the layer across multiple tiles
            tiles_layer = layer.splitAcrossTiles(strategy, nTiles);  // max each tile a layer

            {  // tile-layers must be verified to be valid
                SanityReport post_result;
                for (const auto& one_tile_layer : tiles_layer) {
                    the_layer_validator.check_splitLayer_consistency(one_tile_layer, post_result);
                    if (!post_result.is_usable()) {
                        Logger::warning() << "\n Split Layer is NOT Valid \n *** INFO from LayerValidator: \n"
                                          << post_result.info << "\n *** This LAYER: "
                                          << "\n " << one_tile_layer << " \n strategy: " << (int)strategy
                                          << ", nDPU: " << nDPU << ", nTiles: " << nTiles
                                          << "\nResult: Early termination with Error code:  " << post_result.value()
                                          << " : " << Cycles::toErrorText(post_result.value()) << "\n";
                        break;  // EARLY LOOP exit, otherwise it will be overwritten by next tile check
                    }
                }

                if (!post_result.is_usable()) {
                    if (detailed_split) {  // add all tile layers for info
                        for (const auto& one_tile_layer : tiles_layer) {
                            detailed_split->emplace_back(
                                    OneTileLayerInfo{one_tile_layer, {0, {}}});  // no workloads. no info if good/bad
                        }
                    }

                    return post_result.value();  // EARLY RETURN
                }
            }

            auto tiler = getDPUTiler(*this);
            for (auto& one_tile_layer : tiles_layer) {
                try {
                    // obtains the best DPU workloads split
                    const DPUWorkloadsCost cost_and_workloads = tiler->intraTileSplit(one_tile_layer, options);
                    const auto cycles = cost_and_workloads.first;
                    tiles_cost.push_back(cycles);

                    if (detailed_split) {
                        detailed_split->emplace_back(OneTileLayerInfo{one_tile_layer, cost_and_workloads});
                    }
                } catch (const std::exception& e) {
                    Logger::warning() << "\n Exception thrown while performing intra tile split "
                                      << "\n Exception: " << e.what() << "\n " << layer
                                      << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU
                                      << ", nTiles: " << nTiles
                                      << "\nResult: this tile will have error result ERROR_TILE_SPLIT_EXCEPTION: "
                                      << (CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION << " \n";

                    // add the error result
                    tiles_cost.push_back((CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION);  // big value
                    if (detailed_split) {
                        detailed_split->emplace_back(
                                OneTileLayerInfo{one_tile_layer, {tiles_cost.back(), {}}});  // no workloads info
                    }
                }
            }
        }

        // The cost of the worst case in the tiles (since error codes are large the largest error code will dominate
        // any regular value)
        CyclesInterfaceType cost = extractLargestTime(tiles_cost);

        if (!Cycles::isErrorCode(cost)) {
            if (!prefetching) {
                // in case of non-overlappable prefetching the cost is the sum of DPU + weights DMA
                // each tile-layer has its own transfer (eg for SOK the w size might be  half or less).
                // we have multiple DMA channels, so we can have pipelining of tile memory
                //  still the naive/simple assumption is that DPU will start after all tile memory is copies (no overlap
                //  tx with DPU)
                std::vector<CyclesInterfaceType> w_costs;
                for (auto& one_tile_layer : tiles_layer) {
                    auto one_tile_w_cost = OneTileWeightsPrefetching(one_tile_layer);  // contains latency
                    w_costs.push_back(one_tile_w_cost);
                }
                const auto pipelined_cost =
                        dpu_schedule(get_dma_ports(layer.device), w_costs);  // pipelines on dma channels
                const auto prefetching_cost = pipelined_cost;
                cost = Cycles::cost_adder(cost, prefetching_cost);

                // add details on layers DMA info
                if (detailed_split) {
                    const auto tile_count{detailed_split->size()};
                    if (tile_count == w_costs.size()) {
                        for (size_t i = 0; i < tile_count; ++i) {
                            (*detailed_split)[i].DMA_info.w_tensor.cycles = w_costs[i];
                        }
                    }
                }
            }

            if (input_in_ddr) {
                // Add cost of loading input activation from DDR to CMX
                for (auto& inT : layer.inputs) {
                    auto in_ddr_dma = DMA(layer.device, inT, inT, MemoryLocation::DRAM, MemoryLocation::CMX);
                    cost = Cycles::cost_adder(cost, in_ddr_dma);
                }
            }

            if (output_in_ddr) {
                // Add cost of spilling output activation from CMX to DDR
                for (auto& outT : layer.outputs) {
                    auto out_ddr_dma = DMA(layer.device, outT, outT, MemoryLocation::CMX, MemoryLocation::DRAM);
                    cost = Cycles::cost_adder(cost, out_ddr_dma);
                }
            }

        } else {
            Logger::error() << "\n Layer cost has an error, skipping DMA/memory time computation: "
                            << "ERROR code: " << cost << " : " << Cycles::toErrorText(cost) << "\n"
                            << layer << "\n"
                            << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU << ", nTiles: " << nTiles
                            << "\n";
        }

        return cost;
    }

    CyclesInterfaceType layer_pre_split_cycles(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU = 1,
                                               bool input_in_ddr = false, bool output_in_ddr = false,
                                               bool prefetching = true, LayerSplitInfo* detailed_split = nullptr) {
        // add missing info by deducing it (not anymore received by params)
        unsigned int nTiles{(unsigned int)layers_pre_split.size()};
        VPUTilingStrategy strategy{VPUTilingStrategy::__size};  //< unknown
        VPUDevice device{nTiles ? layers_pre_split[0].device : VPUDevice::__size};
        // end of missing info

        std::vector<CyclesInterfaceType> tiles_cost;          // cost of each tile
        std::vector<DPULayer> tiles_layer{layers_pre_split};  //< layer list after split
        {
            // operation_sanitisation(layer);  // AVEPOOL will be transformed to something equivalent
            const SplitOptions options{maxWorkloadsPerIntraTileSplit /*maxWorkloads*/, 0,
                                       nDPU};  // here always for LATENCY => cycles

            // Create a new instance of the cost model
            // std::shared_ptr<VPUCostModel> model = std::make_shared<VPUCostModel>(
            //        *this);  // why? and how does this work properly since CostMOdel does not have a proper copy ctor

            //{  // the layer must be verified to be valid
            //    SanityReport unsplit_result;
            //    the_layer_validator.sanitize_preconditions(
            //            layer);  // this might change the layer. eg: siwzzlings for VPU2.0
            //    the_layer_validator.check_completeLayer_consistency(
            //            layer, unsplit_result, DPULayer::mapTilingStrategiesToWorkload(strategy), nTiles);

            //    if (!unsplit_result.is_usable()) {
            //        Logger::warning() << "\n Layer is NOT Valid \n *** INFO from LayerValidator:\n "
            //                          << unsplit_result.info << "\n"
            //                          << "\n *** LAYER: "
            //                          << "\n " << layer << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU
            //                          << ", nTiles: " << nTiles
            //                          << "\n Result: Early termination with Error code: " << unsplit_result.value()
            //                          << " : " << Cycles::toErrorText(unsplit_result.value()) << "\n";
            //        return unsplit_result.value();  // EARLY RETURN
            //    }
            //}

            // split the layer across multiple tiles
            // tiles_layer = layer.splitAcrossTiles(strategy, nTiles);  // max each tile a layer

            {  // tile-layers must be verified to be valid
                SanityReport post_result;
                for (auto& one_tile_layer : tiles_layer) {
                    operation_sanitisation(one_tile_layer);  // AVEPOOL will be transformed to something equivalent
                    the_layer_validator.sanitize_preconditions(
                            one_tile_layer);  // this might change the layer. eg: siwzzlings for VPU2.0

                    the_layer_validator.check_splitLayer_consistency(one_tile_layer, post_result);
                    if (!post_result.is_usable()) {
                        Logger::warning() << "\n Split Layer is NOT Valid \n *** INFO from LayerValidator: \n"
                                          << post_result.info << "\n *** This LAYER: "
                                          << "\n " << one_tile_layer << " \n strategy: " << (int)strategy << " = "
                                          << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
                                          << ", nDPU: " << nDPU << ", nTiles: " << nTiles
                                          << "\nResult: Early termination with Error code:  " << post_result.value()
                                          << " : " << Cycles::toErrorText(post_result.value()) << "\n";
                        break;  // EARLY LOOP exit, otherwise it will be overwritten by next tile check
                    }
                }

                if (!post_result.is_usable()) {
                    if (detailed_split) {  // add all tile layers for info
                        for (const auto& one_tile_layer : tiles_layer) {
                            detailed_split->emplace_back(
                                    OneTileLayerInfo{one_tile_layer, {0, {}}});  // no workloads. no info if good/bad
                        }
                    }
                    return post_result.value();  // EARLY RETURN
                }
            }  // inter tile layers sanitized and validated

            auto tiler = getDPUTiler(*this);  // intra-tile tiler
            for (auto& one_tile_layer : tiles_layer) {
                try {
                    // obtains the best DPU workloads split
                    const DPUWorkloadsCost cost_and_workloads = tiler->intraTileSplit(one_tile_layer, options);
                    const auto cycles = cost_and_workloads.first;
                    tiles_cost.push_back(cycles);

                    if (detailed_split) {
                        detailed_split->emplace_back(OneTileLayerInfo{one_tile_layer, cost_and_workloads});
                    }
                } catch (const std::exception& e) {
                    Logger::warning() << "\n Exception thrown while performing intra tile split "
                                      << "\n Exception: " << e.what() << "\n *** This LAYER: "
                                      << "\n " << one_tile_layer << " \n strategy: " << (int)strategy << " = "
                                      << VPUTilingStrategy_ToText.at(static_cast<int>(strategy)) << ", nDPU: " << nDPU
                                      << ", nTiles: " << nTiles
                                      << "\nResult: this tile will have error result ERROR_TILE_SPLIT_EXCEPTION: "
                                      << (CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION << " \n";

                    // add the error result
                    tiles_cost.push_back((CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION);  // big value
                    if (detailed_split) {
                        detailed_split->emplace_back(
                                OneTileLayerInfo{one_tile_layer, {tiles_cost.back(), {}}});  // no workloads info
                    }
                }
            }
        }

        // The cost of the worst case in the tiles (since error codes are large the largest error code will dominate
        // any regular value)
        CyclesInterfaceType cost = extractLargestTime(tiles_cost);

        if (!Cycles::isErrorCode(cost)) {
            if (!prefetching) {
                // in case of non-overlappable prefetching the cost is the sum of DPU + weights DMA
                // each tile-layer has its own transfer (eg for SOK the w size might be  half or less).
                // we have multiple DMA channels, so we can have pipelining of tile memory
                //  still the naive/simple assumption is that DPU will start after all tile memory is copies (no overlap
                //  tx with DPU)
                std::vector<CyclesInterfaceType> w_costs;
                for (auto& one_tile_layer : tiles_layer) {
                    auto one_tile_w_cost = OneTileWeightsPrefetching(one_tile_layer);  // contains latency
                    w_costs.push_back(one_tile_w_cost);
                }
                const auto pipelined_cost = dpu_schedule(get_dma_ports(device), w_costs);  // pipelines on dma channels
                const auto prefetching_cost = pipelined_cost;
                cost = Cycles::cost_adder(cost, prefetching_cost);

                // add details on layers DMA info
                if (detailed_split) {
                    const auto tile_count{detailed_split->size()};
                    if (tile_count == w_costs.size()) {
                        for (size_t i = 0; i < tile_count; ++i) {
                            (*detailed_split)[i].DMA_info.w_tensor.cycles = w_costs[i];
                        }
                    }
                }
            }
            // Data fetch time is computed considering the split layers input tensors, that are summed up.
            if (input_in_ddr) {
                // Add cost of loading input activation from DDR to CMX
                std::vector<CyclesInterfaceType> dma_costs;
                for (auto& one_tile_layer : tiles_layer) {
                    static_assert(std::tuple_size<decltype(one_tile_layer.inputs)>::value == 1,
                                  "one input restriction");

                    const auto& inT{one_tile_layer.inputs[0]};
                    CyclesInterfaceType one_tile_cost =
                            DMA(device, inT, inT, MemoryLocation::DRAM, MemoryLocation::CMX);

                    dma_costs.push_back(one_tile_cost);
                }

                const auto sum_dma_layers = std::accumulate(dma_costs.begin(), dma_costs.end(), 0u, Cycles::cost_adder);
                cost = Cycles::cost_adder(cost, sum_dma_layers);

                // add details on layers DMA info
                if (detailed_split) {
                    const auto tile_count{detailed_split->size()};
                    if (tile_count == dma_costs.size()) {
                        for (size_t i = 0; i < tile_count; ++i) {
                            (*detailed_split)[i].DMA_info.input_tensor.cycles = dma_costs[i];
                        }
                    }
                }
            }

            // Data fetch time is computed considering the split layers output tensors, that are summed up.
            if (output_in_ddr) {
                // Add cost of spilling output activation from CMX to DDR
                std::vector<CyclesInterfaceType> dma_costs;
                for (auto& one_tile_layer : tiles_layer) {
                    static_assert(std::tuple_size<decltype(one_tile_layer.outputs)>::value == 1,
                                  "one input restriction");

                    const auto& outT{one_tile_layer.outputs[0]};
                    CyclesInterfaceType one_tile_cost =
                            DMA(device, outT, outT, MemoryLocation::CMX, MemoryLocation::DRAM);

                    dma_costs.push_back(one_tile_cost);
                }

                const auto sum_dma_layers = std::accumulate(dma_costs.begin(), dma_costs.end(), 0u, Cycles::cost_adder);
                cost = Cycles::cost_adder(cost, sum_dma_layers);

                // add details on layers DMA info
                if (detailed_split) {
                    const auto tile_count{detailed_split->size()};
                    if (tile_count == dma_costs.size()) {
                        for (size_t i = 0; i < tile_count; ++i) {
                            (*detailed_split)[i].DMA_info.output_tensor.cycles = dma_costs[i];
                        }
                    }
                }
            }

        } else {
            Logger::error() << "\n Layer cost has an error, skipping DMA/memory time computation: "
                            << "ERROR code: " << cost << " : " << Cycles::toErrorText(cost) << "\n"
                            << " \n strategy: " << (int)strategy << " = "
                            << VPUTilingStrategy_ToText.at(static_cast<int>(strategy)) << ", nDPU: " << nDPU
                            << ", nTiles: " << nTiles << "\n";
        }

        return cost;
    }

public:
    /**
     * @brief Compute the optimal cost of a DPULayer, given a context but no strategy
     *
     * Analyses all strategies and selects the time o the fastest one
     *
     * @param layer the DPULayer
     * @param nDPU the number of DPU
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @param prefetching enable/disable weight prefetching
     * @return measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType Layer(DPULayer& layer, unsigned int nDPU = 1, unsigned int nTiles = 1,
                              bool input_in_ddr = false, bool output_in_ddr = false, bool prefetching = true) {
        // Cost of a layer if executed in nTiles using nDPU/tile
        auto valid_strategies = getValidTilingStrategies(layer.device);

        // The cost of all configurations
        std::vector<CyclesInterfaceType> costs;

        for (const auto& strategy : valid_strategies) {
            auto configuration_cost = Layer(layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr, prefetching);
            costs.push_back(configuration_cost);
        }

        // Return the configuration with min cost (the optimal one). Any good value will dominate any error code
        return *std::min_element(costs.begin(), costs.end());
    }

    // Shave Operations area is next

    /**
     * @brief Compute the optimal cost of a SWOperation
     *
     * @param layer the SHV kernel
     * @param strategy the layer strategy
     * @return unsigned long int
     */
    unsigned long int Layer(SWOperation& layer, const VPULayerStrategy& strategy) {
        return Layer(layer, strategy.nSHVs, strategy.nTiles, strategy.input_fetching, strategy.output_spilling);
    }

    /**
     * @brief Compute the optimal cost of a SHV kernel
     *
     * @param layer the SHV kernel
     * @param nSHV the number of SHV/tile
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @return unsigned long int
     */
    unsigned long int Layer(SWOperation& layer, unsigned int nSHV = 1, unsigned int nTiles = 1,
                            bool input_in_ddr = false, bool output_in_ddr = false) {
        // For shave layer we use a simplistic model, as we assume the cost can be scale up to 4 tiles
        unsigned int single_shv_cost = SHAVE(layer);
        unsigned int cost = static_cast<unsigned int>((float)single_shv_cost / ((float)(nSHV * nTiles)));

        if (input_in_ddr) {
            // Add cost of loading input activation from DDR to CMX
            for (auto& inT : layer.inputs) {
                auto in_ddr_dma = DMA(layer.device, inT, inT);
                cost = Cycles::cost_adder(cost, in_ddr_dma);
                // cost += DMA(layer.device, inT, inT);
            }
        }

        if (output_in_ddr) {
            // Add cost of spilling output activation from DDR to CMX
            for (auto& outT : layer.outputs) {
                auto out_ddr_dma = DMA(layer.device, outT, outT);
                cost = Cycles::cost_adder(cost, out_ddr_dma);
                // cost += DMA(layer.device, outT, outT);
            }
        }

        return cost;
    }

protected:
    const IDeviceValidValues& getDeviceConfiguratorForTiles(VPUDevice device) const {
        return the_layer_validator.getDeviceConfiguratorForTiles(device);
    }
    /**
     * @brief The cycles it takes to prefetch the weights of one tile, not considering any pipelining
     * Takes in consideration sparsity (enabled and value) for this transfer
     *
     * @param layer the DPULayer, expected to be a layer allocated to this tile
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType OneTileWeightsPrefetching(const DPULayer& layer) {
        const auto& config = getDeviceConfiguratorForTiles(layer.device);

        const auto in_1_movable_size = layer.weight_footprint(config);

        const VPUTensor weight_plus_table_tensor{{static_cast<unsigned int>(in_1_movable_size), 1, 1, 1},
                                                 DataType::UINT8};  // weight_footprint is in bytes
        return DMA(layer.device, weight_plus_table_tensor, weight_plus_table_tensor, MemoryLocation::DRAM,
                   MemoryLocation::CMX, 1);
    }

    /**
     * @brief Overall memory footprint of a layer
     *
     * @param layer the DPULayer
     * @param strategy the tiling strategy to use
     * @param nTiles the number of CMX tiles
     * @return unsigned long int
     */
    unsigned long int MemoryFootprint(const DPULayer& layer, VPUTilingStrategy strategy, unsigned int nTiles = 1) {
        // split the layer across multiple tiles, for now it uses clustering
        std::vector<DPULayer> tile_workloads = layer.splitAcrossTiles(strategy, nTiles);
        std::vector<unsigned long int> tile_footprints;
        const auto& config = getDeviceConfiguratorForTiles(layer.device);

        for (auto& wl : tile_workloads) {
            // Get the tile footprint
            auto tile_footprint = wl.footprint(config);
            // Compute the cost of executing those workload on 1 tile
            tile_footprints.push_back(tile_footprint);
        }

        // The worse case size in all tiles
        return *std::max_element(tile_footprints.begin(), tile_footprints.end());
    }

protected:
    /// @brief gets the overall DPU cycles considering also error cases
    /// If at least one time is an error (one tile is bad) the result will be an error with code
    /// Cycles::ERROR_TILE_OUTPUT
    ///
    /// @param tiles_cost is the container with time values CyclesInterfaceType
    /// @returns the largest if all good, and error if not all are good ()
    CyclesInterfaceType extractLargestTime(const std::vector<CyclesInterfaceType>& tiles_cost) const {
        CyclesInterfaceType largest_time{0};
        for (auto tile : tiles_cost) {
            if (Cycles::isErrorCode(tile)) {
                largest_time = Cycles::ERROR_TILE_OUTPUT;
                Logger::error() << "\n While analyzing a tile output, the cycle time value was with error: "
                                << "ERROR code: " << tile << " : " << Cycles::toErrorText(tile)
                                << " Layer results will be with ERROR \n";
            } else {                                       // zero is not concerning?
                if (!Cycles::isErrorCode(largest_time)) {  // keep error time
                    largest_time = std::max(largest_time, tile);
                }
            }
        }

        if (largest_time == 0) {  // covers also empty vector
            largest_time = Cycles::ERROR_TILE_OUTPUT;
            Logger::error() << "\n All tiles output were zero! Tiles count:  " << tiles_cost.size()
                            << " Layer results will be with ERROR \n";
        }

        return largest_time;
    }

    void operation_sanitisation(DPULayer& wl) const {
        avgpool_replace_by(wl);
        compressConv_replace_by_CM_CONV_VPU27(wl);
    }

    // static members
public:
    /**
     * @brief Get the valid tiling strategy for a device
     *
     * @param device the VPUDevice
     * @return std::vector<VPUTilingStrategy>
     */
    static std::vector<VPUTilingStrategy> getValidTilingStrategies(const VPUDevice& device) {
        switch (device) {
        case VPUDevice::VPU_2_0:
        case VPUDevice::VPU_2_1:
        case VPUDevice::VPU_2_7:
            return {VPUTilingStrategy::NONE, VPUTilingStrategy::SOH, VPUTilingStrategy::SOK};
        case VPUDevice::VPU_4_0:
            return {VPUTilingStrategy::NONE, VPUTilingStrategy::SOH,  VPUTilingStrategy::SOK,
                    VPUTilingStrategy::SOW,  VPUTilingStrategy::SOHW, VPUTilingStrategy::SOHK};
        default:
            return {};
        }
    }
};

}  // namespace VPUNN

#endif  // VPUNN_LAYER_COST_MODEL_H