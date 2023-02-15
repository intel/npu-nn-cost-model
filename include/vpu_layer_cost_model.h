// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_LAYER_COST_MODEL_H
#define VPUNN_LAYER_COST_MODEL_H

#include "core/logger.h"
#include "vpu/layer.h"
#include "vpu/optimization/workload_optimization.h"
#include "vpu/performance.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpu_cost_model.h"
#include "vpunn.h"

namespace VPUNN {

/**
 * @brief A VPU layer strategy
 *
 */
struct VPULayerStrategy {
    /**
     * @brief Number of DPU/tile
     *
     */
    unsigned int nDPUs = 1;
    /**
     * @brief Number of Shaves/tile
     *
     */
    unsigned int nSHVs = 1;
    /**
     * @brief Number of tiles
     *
     */
    unsigned int nTiles = 1;
    /**
     * @brief Layer tiling strategy
     *
     */
    VPUTilingStrategy tiling_strategy = VPUTilingStrategy::NONE;
    /**
     * @brief true if the layer input is in DDR
     *
     */
    bool input_fetching = false;
    /**
     * @brief true if the layer output is in DDR
     *
     */
    bool output_spilling = false;
    /**
     * @brief If layer parameters are prefetched with previous layers
     *
     */
    bool prefetching = true;
};

/**
 * @brief The VPUNN layer cost model (also called VPUNN Level2 API)
 *
 */
class VPUNN_API(VPULayerCostModel): public VPUCostModel {
public:
    /**
     * @brief Using the same VPUCostModel constructor
     *
     */
    using VPUCostModel::VPUCostModel;

    /**
     * @brief Get the valid tiling strategy for a devie
     *
     * @param device the VPUDevice
     * @return std::vector<VPUTilingStrategy>
     */
    static std::vector<VPUTilingStrategy> getValidTilingStrategy(VPUDevice& device) {
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

    /**
     * @brief Compute the optimal cost of a DPULayer
     *
     * @param layer the DPULayer
     * @param strategy the layer strategy
     * @return unsigned long int
     */
    unsigned long int Layer(DPULayer& layer, VPULayerStrategy strategy) {
        return Layer(layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles, strategy.input_fetching,
                     strategy.output_spilling, strategy.prefetching);
    }

    /**
     * @brief Compute the optimal cost of a SWOperation
     *
     * @param layer the SHV kernel
     * @param strategy the layer strategy
     * @return unsigned long int
     */
    unsigned long int Layer(SWOperation& layer, VPULayerStrategy strategy) {
        return Layer(layer, strategy.nSHVs, strategy.nTiles, strategy.input_fetching, strategy.output_spilling);
    }

    /**
     * @brief Compute the optimal cost of a DPULayer
     *
     * @param layer the DPULayer
     * @param nDPU the number of DPU
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @param prefetching enable/disable weight prefetching
     * @return unsigned long int
     */
    unsigned long int Layer(DPULayer layer, unsigned int nDPU = 1, unsigned int nTiles = 1, bool input_in_ddr = false,
                            bool output_in_ddr = false, bool prefetching = true) {
        // Cost of a layer if executed in nTiles using nDPU/tile
        auto valid_strategies = getValidTilingStrategy(layer.device);

        // The cost of all configurations
        std::vector<unsigned long int> costs;

        for (auto& strategy : valid_strategies) {
            // The cost the the worse case in the tiles
            auto configuration_cost = Layer(layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr, prefetching);
            costs.push_back(configuration_cost);
        }

        // Return the configuration with min cost (the optimal one)
        return *std::min_element(costs.begin(), costs.end());
    }

    /**
     * @brief Compute the optimal cost of a DPULayer using a specific strategy and execution mode
     *
     * @param layer the DPULayer
     * @param strategy the tiling strategy to use
     * @param nDPU the number of DPU
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @param prefetching enable/disable weight prefetching
     * @return unsigned long int
     */
    unsigned long int Layer(DPULayer layer, VPUTilingStrategy strategy, unsigned int nDPU = 1, unsigned int nTiles = 1,
                            bool input_in_ddr = false, bool output_in_ddr = false, bool prefetching = true) {
        // split the layer across multiple tiles, for now it uses clustering
        std::vector<DPULayer> tile_workloads = layer.splitAcrossTiles(strategy, nTiles);
        std::vector<unsigned long int> tile_cost;
        SplitOptions options;
        options.nDPU = nDPU;
        options.maxWorkloads = 50;

        // Create a new instance of the cost model
        std::shared_ptr<VPUCostModel> model = std::make_shared<VPUCostModel>(*this);
        auto tiler = getDPUTiler(model);

        for (auto& wl : tile_workloads) {
            // tile considering the execution mode
            auto workloads = tiler->intraTileSplit(wl, options);
            auto performance = tiler->getLayerPerformance(workloads);
            // Compute the cost of executing those workload on 1 tile
            tile_cost.push_back(performance.cycles);
        }

        // The cost of the worse case in the tiles
        auto cost = *std::max_element(tile_cost.begin(), tile_cost.end());
        if (!prefetching) {
            // in case of non-overlappable prefetching
            auto prefetching_cost = WeightsPrefetching(layer, strategy, nTiles);
            cost = std::max(cost, prefetching_cost);
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

        return cost;
    }

    /**
     * @brief Compute the optimal cost of a SHV kernel using a specific strategy and execution mode
     *
     * @param layer the SHV kernel
     * @param strategy the tiling strategy to use
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
    unsigned long int WeightsPrefetching(DPULayer layer, VPUTilingStrategy strategy, unsigned int nTiles = 1) {
        auto weight_plus_table_tensor = VPUTensor({layer.weight_footprint(), 1, 1, 1}, DataType::UINT8);
        auto output_write_tiles = strategy == VPUTilingStrategy::SOK ? nTiles : 1;
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
    unsigned long int MemoryFootprint(DPULayer layer, VPUTilingStrategy strategy, unsigned int nTiles = 1) {
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
};

}  // namespace VPUNN

#endif  // VPUNN_LAYER_COST_MODEL_H