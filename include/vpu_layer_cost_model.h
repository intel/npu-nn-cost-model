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

    bool prefetching{true};  ///< If layer parameters are prefetched with previous layers
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

/// details about a tile split strategy
struct OneTileLayerInfo {
    DPULayer inter_tile_split_layer;  ///<  layer resulted by splitting the orginalLayer to one tile using requested
                                      ///<  strategy
    DPUWorkloadsCost best_intra_tile_split;  ///< the cost and list of workloads that were inferred to be the best
};

/// info on how were the splits on each tile
using LayerSplitInfo = std::vector<OneTileLayerInfo>;

/// @brief The VPUNN layer cost model (also called VPUNN Level2 API)
class VPUNN_API(VPULayerCostModel): public VPUCostModel {
private:
    const LayersValidation the_layer_validator{};     ///< used for validating the un-split layers and split layers
    unsigned int maxWorkloadsPerIntraTileSplit{50U};  ///< max splits for a tile

public:
    using VPUCostModel::VPUCostModel;  ///< exposing/Using the same VPUCostModel constructor (base class)

    void set_maxWorkloadsPerIntraTileSplit(unsigned int new_value) noexcept {
        maxWorkloadsPerIntraTileSplit = new_value;
    }
    auto get_maxWorkloadsPerIntraTileSplit() const noexcept {
        return maxWorkloadsPerIntraTileSplit;
    }

    /**
     * @brief Compute the optimal cost of a DPULayer
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
     * @brief Compute the optimal cost of a DPULayer using a specific strategy and execution mode
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
     * @param prefetching enable/disable weight prefetching
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
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @param prefetching enable/disable weight prefetching
     * @param detailed_split [out] gives as output the information on how was split this layer and what is the best
     * split on workloads
     * @return measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType Layer(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU, unsigned int nTiles,
                              bool input_in_ddr, bool output_in_ddr, bool prefetching, LayerSplitInfo& detailed_split) {
        return layer_cycles(layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr, prefetching, &detailed_split);
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
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @param prefetching enable/disable weight prefetching
     * @param detailed_split [out] gives as output the information on how was split this layer and what is the best
     * split on workloads. ignored if null
     * @return measured best cycles or error code . \see Cycles for error codes
     */
    CyclesInterfaceType layer_cycles(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU = 1,
                                     unsigned int nTiles = 1, bool input_in_ddr = false, bool output_in_ddr = false,
                                     bool prefetching = true, LayerSplitInfo* detailed_split = nullptr) {
        std::vector<CyclesInterfaceType> tiles_cost;  // cost of each tile
        {
            operation_sanitisation(layer);  // AVEPOOL will be transformed to something equivalent
            const SplitOptions options{maxWorkloadsPerIntraTileSplit /*maxWorkloads*/, 0,
                                       nDPU};  // here always for LATENCY => cycles

            // Create a new instance of the cost model
            std::shared_ptr<VPUCostModel> model = std::make_shared<VPUCostModel>(
                    *this);  // why? and how does this work properly since CostMOdel does not have a proper copy ctor

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
            std::vector<DPULayer> tiles_layer = layer.splitAcrossTiles(strategy, nTiles);  // max each tile a layer

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

            auto tiler = getDPUTiler(model);
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
                auto prefetching_cost = WeightsPrefetching(layer, strategy, nTiles);
                cost = cost + prefetching_cost;
            }

            if (input_in_ddr) {
                // Add cost of loading input activation from DDR to CMX
                for (auto& inT : layer.inputs) {
                    cost += DMA(layer.device, inT, inT);
                }
            }

            if (output_in_ddr) {
                // Add cost of spilling output activation from DDR to CMX
                for (auto& outT : layer.outputs) {
                    cost += DMA(layer.device, outT, outT);
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

    /**
     * @brief Compute the optimal cost of a DPULayer
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
                cost += DMA(layer.device, inT, inT);
            }
        }

        if (output_in_ddr) {
            // Add cost of spilling output activation from DDR to CMX
            for (auto& outT : layer.outputs) {
                cost += DMA(layer.device, outT, outT);
            }
        }

        return cost;
    }

    /**
     * @brief The cycles it takes to prefetch the weights
     *
     * @param layer the DPULayer
     * @param strategy the tiling strategy to use
     * @param nTiles the number of CMX tiles
     * @return unsigned long int
     */
    unsigned long int WeightsPrefetching(const DPULayer& layer, VPUTilingStrategy strategy, unsigned int nTiles = 1) {
        auto weight_plus_table_tensor =
                VPUTensor({layer.weight_footprint(), 1, 1, 1}, DataType::UINT8);  // WHY assumed int?
        auto output_write_tiles = ((strategy == VPUTilingStrategy::SOK) ? nTiles : 1);
        return DMA(layer.device, weight_plus_table_tensor, weight_plus_table_tensor, MemoryLocation::DRAM,
                   MemoryLocation::CMX, output_write_tiles);
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

        for (auto& wl : tile_workloads) {
            // Get the tile footprint
            auto tile_footprint = wl.footprint();
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