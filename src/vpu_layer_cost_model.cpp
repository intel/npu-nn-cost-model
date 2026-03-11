// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu_layer_cost_model.h"

#include <algorithm>
#include <array>  // for std::array
#include <exception>
#include <memory>  // for std::make_shared (if used)
#include <numeric>
#include <optional>  // for std::optional
#include <string>    // for std::string
#include <tuple>     // for std::tuple_size
#include <type_traits>
#include <variant>  // for std::visit, std::is_same_v
#include <vector>   // for std::vector
#include "core/logger.h"
#include "vpu/device_layer_properties/device_layer_properties_holder.h"
#include "vpu/dpu_defaults.h"
#include "vpu/optimization/workload_optimization.h"
#include "vpu/performance.h"
#include "vpu/serialization/l2_cost_serialization_wrapper.h"
#include "vpu/utils.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu/validation/layer_sanitizer.h"
#include "vpu_dma_cost_model.h"

namespace VPUNN {

void VPULayerCostModel::initialize_serializers() {
    // TODO: better move the serializers in a subobject that makes the init in its constructor
    serializer.initialize("l2_dpu_workloads", FileMode::READ_WRITE, get_names_for_serializer());
    presplit_serializer.initialize("l2_dpu_workloads_presplit", FileMode::READ_WRITE, get_names_for_serializer());
}

CyclesInterfaceType VPULayerCostModel::Layer(DPULayer& layer, VPULayerStrategy strategy) {
    // VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
    return layer_cycles(internal_dpu_cost_provider, layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                        strategy.input_fetching, strategy.output_spilling, strategy.prefetching);
}

CyclesInterfaceType VPULayerCostModel::Layer(DPULayer& layer, VPULayerStrategy strategy,
                                             LayerSplitInfo& detailed_split) {
    // VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
    return layer_cycles(internal_dpu_cost_provider, layer, strategy.tiling_strategy, strategy.nDPUs, strategy.nTiles,
                        strategy.input_fetching, strategy.output_spilling, strategy.prefetching, &detailed_split);
}

CyclesInterfaceType VPULayerCostModel::Layer_dCiM(DPULayer& layer, VPULayerStrategy strategy) {
    Logger::info() << "DCIM LAYER:" << layer.get_layer_info();
    Logger::info() << strategy;
    return Cycles::ERROR_TILE_SPLIT_EXCEPTION;
}

CyclesInterfaceType VPULayerCostModel::Layer_dCiM(DPULayer& layer, VPULayerStrategy strategy,
                                                  LayerSplitInfo& detailed_split) {
    Logger::info() << "DCIM LAYER:" << layer.get_layer_info();
    Logger::info() << strategy;
    detailed_split.clear();
    return Cycles::ERROR_TILE_SPLIT_EXCEPTION;
}

CyclesInterfaceType VPULayerCostModel::Layer(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU,
                                             unsigned int nTiles, bool input_in_ddr, bool output_in_ddr,
                                             bool prefetching) {
    // VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
    return layer_cycles(internal_dpu_cost_provider, layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr,
                        prefetching, nullptr);
}

CyclesInterfaceType VPULayerCostModel::Layer(DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU,
                                             unsigned int nTiles, bool input_in_ddr, bool output_in_ddr,
                                             bool prefetching, LayerSplitInfo& detailed_split) {
    //        VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
    return layer_cycles(internal_dpu_cost_provider, layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr,
                        prefetching, &detailed_split);
}

CyclesInterfaceType VPULayerCostModel::LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                                      bool input_in_ddr, bool output_in_ddr, bool prefetching,
                                                      LayerSplitInfo& detailed_split, const size_t fullLayerHash,
                                                      const std::optional<VPUTilingStrategy> strategyOfSplit) {
    //        VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
    return layer_pre_split_cycles(internal_dpu_cost_provider, layers_pre_split, nDPU, input_in_ddr, output_in_ddr,
                                  prefetching, &detailed_split, fullLayerHash, strategyOfSplit);
}

CyclesInterfaceType VPULayerCostModel::LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                                      bool input_in_ddr, bool output_in_ddr, bool prefetching) {
    //        VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
    return layer_pre_split_cycles(internal_dpu_cost_provider, layers_pre_split, nDPU, input_in_ddr, output_in_ddr,
                                  prefetching, nullptr);
}

CyclesInterfaceType VPULayerCostModel::LayersPreSplit(const std::vector<SHAVEWorkload>& layers_pre_split,
                                                      unsigned int nSHV, bool input_in_ddr, bool output_in_ddr) {
    return layer_pre_split_cycles(internal_dpu_cost_provider, layers_pre_split, nSHV, input_in_ddr, output_in_ddr);
}

bool VPULayerCostModel::is_dma_model_variant() const {
    return std::visit(
            [](const auto& dma_model) {
                return dma_model && dma_model->nn_initialized();
            },
            the_dma_cost_model);
}

CyclesInterfaceType VPULayerCostModel::compute_dma_cycles(const DMATransfer1D& dwl) const {
    CyclesInterfaceType cost = Cycles::NO_ERROR;

    if (!is_dma_model_variant()) {
        Logger::warning() << "\n No DmaCostModel is initialized, fallback to theoretical model. \n";
        cost = get_TheoreticalDMA_cost_model().DMA(convert_dma1d_2_dmawl(dwl));
    } else {
        cost = std::visit(
                [dwl](const auto& dma_model) -> CyclesInterfaceType {
                    // Deduce workload type from DMACostModel type
                    using WLType = typename std::remove_pointer<typename std::remove_reference<
                            typename std::decay_t<decltype(dma_model)>>::type>::type::DescType;

                    // Check if the device corresponds to the initialized DMACostModel
                    bool device_match = false;
                    switch (dwl.device) {
                    case VPUDevice::VPU_2_0:
                    case VPUDevice::VPU_2_1:
                    case VPUDevice::VPU_2_7:
                        if constexpr (std::is_same_v<WLType, DMANNWorkload_NPU27>)
                            device_match = true;
                        break;
                    case VPUDevice::VPU_4_0:
                        if constexpr (std::is_same_v<WLType, DMANNWorkload_NPU40>)
                            device_match = true;
                        break;
                    case VPUDevice::NPU_5_0:
                    case VPUDevice::NPU_RESERVED:
                        if constexpr (std::is_same_v<WLType, DMANNWorkload_NPU50>)
                            device_match = true;
                        break;
                    default:
                        break;
                    }
                    return device_match ? dma_model->computeCycles(DMANNWorkloadCreator<WLType>::create_workload(dwl))
                                        : Cycles::ERROR_INVALID_INPUT_DEVICE;
                },
                the_dma_cost_model);
    }

    if (Cycles::isErrorCode(cost))
        Logger::error() << "\n While analyzing a DMA workload, the cycle time value was with error: "
                        << "ERROR code: " << cost << " : " << Cycles::toErrorText(cost)
                        << " Layer results will be with ERROR \n";

    return cost;
}

const std::vector<std::string> VPULayerCostModel::get_names_for_serializer() {
    auto fields = NNCostProvider::get_names_for_serializer();
    fields.emplace_back("n_requested_tiles");
    fields.emplace_back("n_computed_tiles");
    fields.emplace_back("n_dpu");
    fields.emplace_back("tiling_strategy");
    fields.emplace_back("name");
    fields.emplace_back("level");
    fields.emplace_back("layer_uid");
    fields.emplace_back("workload_uid");
    fields.emplace_back("intra_tile_seq_id");

    return fields;
}

CyclesInterfaceType VPULayerCostModel::layer_cycles(VPUCostModel& dpu_cost_provider, DPULayer& layer,
                                                    VPUTilingStrategy strategy, unsigned int nDPU, unsigned int nTiles,
                                                    bool input_in_ddr, bool output_in_ddr, bool prefetching,
                                                    LayerSplitInfo* detailed_split) const {
    dpu_cost_provider.swizzling_turn_OFF(layer);

    if (nTiles == 0) {
        Logger::warning() << "Number of tiles can't be zero, should be at least one!";
        return Cycles::ERROR_L2_INVALID_PARAMETERS;
    }

    VPUDevice device{nTiles ? layer.device : VPUDevice::__size};

    // serialization init logic
    const L2CostSerializationWrap::LayerSerializationContext layer_cyc_ctxt{nDPU, nTiles, strategy, device};
    L2CostSerializationWrap serialization_handler(serializer, layer_cyc_ctxt, the_layer_validator, dpu_cost_provider,
                                                  L2CostSerializationWrap::layer_info_content, detailed_split);

    if (detailed_split) {  // always
        detailed_split->clear();
    }

    serialization_handler.serializeLayerInformation_header_and_compute_layer_uid(layer);  // must keep the csv line
                                                                                          // open!

    std::vector<CyclesInterfaceType> tiles_cost;  // cost of each tile
    std::vector<DPULayer> tiles_layer;            //< layer list after split

    // split the layer section, all splits
    {
        operation_sanitisation(layer);  // AVEPOOL will be transformed to something equivalent
        const SplitOptions options{maxWorkloadsPerIntraTileSplit, 0, nDPU};  // here always for LATENCY => cycles

        {  // the layer must be verified to be valid
            SanityReport unsplit_result;
            the_layer_validator.sanitize_preconditions(
                    layer);  // this might change the layer. eg: siwzzlings for VPU2.0
            the_layer_validator.check_completeLayer_consistency(
                    layer, unsplit_result, DPULayer::mapTilingStrategiesToWorkload(strategy), nTiles, strategy);

            if (!unsplit_result.is_usable()) {
                Logger::warning() << "\n Layer is NOT Valid \n *** INFO from LayerValidator:\n " << unsplit_result.info
                                  << "\n"
                                  << "\n *** LAYER: "
                                  << "\n " << layer << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU
                                  << ", nTiles: " << nTiles
                                  << "\n Result: Early termination with Error code: " << unsplit_result.value() << " : "
                                  << Cycles::toErrorText(unsplit_result.value()) << "\n";

                serialization_handler.serializeCycles_closeLine(unsplit_result.value());

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
            if (0 >= tiles_layer.size()) {  // no split present
                post_result.mark_split_error();
            }

            if (!post_result.is_usable()) {
                if (detailed_split) {  // add all tile layers for info
                    for (const auto& one_tile_layer : tiles_layer) {
                        detailed_split->emplace_back(
                                OneTileLayerInfo{one_tile_layer});  // no workloads. no info if good/bad
                    }
                }

                serialization_handler.serializeCycles_closeLine(post_result.value());

                return post_result.value();  // EARLY RETURN
            }
        }  // inter tile layers sanitized and validated

        // VPUCostModel& dpu_cost_provider(*this);       // this is the cost provider for the DPU workloads
        auto tiler = getDPUTiler(dpu_cost_provider);  // intra-tile tiler
        for (auto& one_tile_layer : tiles_layer) {
            try {
                // obtains the best DPU workloads split
                std::vector<DPUWorkloadsWithCyclesSplit> splits;
                const DPUWorkloadsCost cost_and_workloads = tiler->intraTileSplit(one_tile_layer, options, &splits);
                const auto cycles = cost_and_workloads.first;
                tiles_cost.push_back(cycles);

                if (detailed_split) {
                    detailed_split->emplace_back(
                            OneTileLayerInfo{one_tile_layer, std::move(cost_and_workloads), std::move(splits)});
                }
            } catch (const std::exception& e) {
                Logger::warning() << "\n Exception thrown while performing intra tile split "
                                  << "\n Exception: " << e.what() << "\n " << layer << " \n strategy: " << (int)strategy
                                  << ", nDPU: " << nDPU << ", nTiles: " << nTiles
                                  << "\nResult: this tile will have error result ERROR_TILE_SPLIT_EXCEPTION: "
                                  << (CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION << " \n";

                // add the error result
                (void)e;
                tiles_cost.push_back((CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION);  // big value
                if (detailed_split) {
                    detailed_split->emplace_back(
                            OneTileLayerInfo{one_tile_layer, {tiles_cost.back(), {}}});  // no workloads info
                }
            }
        }
    }  // split the layer section, all splits

    // The cost of the worst case in the tiles (since error codes are large the largest error code will dominate
    // any regular value)
    CyclesInterfaceType cost = extractLargestTime(tiles_cost);

    // serialize remaining line with cycles
    serialization_handler.serializeCyclesAndTilesCnt_closeLine(cost, tiles_layer.size());

    // serialize the complete detailed splits
    // should DO ONLY if no previous serialization error! Think about how to handler these situations
    serialization_handler.serializeLayerSplitInfo(tiles_layer.size(),
                                                  *detailed_split);  // info with cluster_ not with /#

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
            const auto pipelined_cost = dpu_schedule(
                    get_HWPerformance().get_hw_info(layer.device).get_dma_ports(),  // get the number of DMA channels
                    // GlobalHarwdwareCharacteristics::get_dma_ports(layer.device),
                    w_costs);  // pipelines on dma channels
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
                auto in_ddr_dma =
                        compute_dma_cycles({layer.device, static_cast<int>(inT.size()), MemoryDirection::DDR2CMX});
                cost = Cycles::cost_adder(cost, in_ddr_dma);
            }
        }

        if (output_in_ddr) {
            // Add cost of spilling output activation from CMX to DDR
            for (auto& outT : layer.outputs) {
                auto out_ddr_dma =
                        compute_dma_cycles({layer.device, static_cast<int>(outT.size()), MemoryDirection::CMX2DDR});
                cost = Cycles::cost_adder(cost, out_ddr_dma);
            }
        }

    } else {
        Logger::error() << "\n Layer cost has an error, skipping DMA/memory time computation: "
                        << "ERROR code: " << cost << " : " << Cycles::toErrorText(cost) << "\n"
                        << layer << "\n"
                        << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU << ", nTiles: " << nTiles << "\n";
    }

    // serialization_handler.cleanBuffers();

    return cost;
}

CyclesInterfaceType VPULayerCostModel::layer_pre_split_cycles(
        VPUCostModel& dpu_cost_provider, const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
        bool input_in_ddr, bool output_in_ddr, bool prefetching, LayerSplitInfo* detailed_split,
        const size_t fullLayerHash, const std::optional<VPUTilingStrategy> strategyOfSplit) const {
    // add missing info by deducing it (not anymore received by params)
    VPUTilingStrategy strategy{strategyOfSplit.value_or(VPUTilingStrategy::UNKNOWN)};  //< unknown
    unsigned int nTiles{(unsigned int)layers_pre_split.size()};
    VPUDevice device{nTiles ? layers_pre_split[0].device : VPUDevice::__size};
    //  end of missing info

    // serialization init logic
    const bool is_serialization_inhibited{(!strategyOfSplit.has_value()) ||
                                          (fullLayerHash == 0)};  // no value no serialization
    const L2CostSerializationWrap::LayerSerializationContext pre_split_ctxt{nDPU, nTiles, strategy, device};
    L2CostSerializationWrap serialization_handler(presplit_serializer, pre_split_ctxt, the_layer_validator,
                                                  dpu_cost_provider, L2CostSerializationWrap::pre_split_info_content,
                                                  detailed_split, is_serialization_inhibited, fullLayerHash);

    // there is no layer to be serialized! only the split layers
    if (detailed_split) {  // always
        detailed_split->clear();
    }

    // sanitize
    if (nTiles == 0) {
        Logger::warning() << "Number of tiles can't be zero, should be at least one!";
        return Cycles::ERROR_L2_INVALID_PARAMETERS;
    }

    std::vector<CyclesInterfaceType> tiles_cost;          // cost of each tile
    std::vector<DPULayer> tiles_layer{layers_pre_split};  //< layers are already split, make a copy

    // split the layer section, all splits
    {
        // operation_sanitisation(layer);  // AVEPOOL will be transformed to something equivalent
        const SplitOptions options{maxWorkloadsPerIntraTileSplit, 0, nDPU};  // here always for LATENCY => cycles

        // split the layer across multiple tiles
        // tiles_layer = layer.splitAcrossTiles(strategy, nTiles);  // max each tile a layer

        {  // tile-layers must be verified to be valid
            SanityReport post_result;
            for (auto& one_tile_layer : tiles_layer) {
                dpu_cost_provider.swizzling_turn_OFF(one_tile_layer);
                operation_sanitisation(one_tile_layer);  // AVEPOOL will be transformed to something equivalent
                the_layer_validator.sanitize_preconditions(
                        one_tile_layer);  // this might change the layer. eg: siwzzlings for VPU2.0

                the_layer_validator.check_splitLayer_consistency(one_tile_layer, post_result);

                if (!post_result.is_usable()) {
                    Logger::warning() << "\n Split Layer is NOT Valid \n *** INFO from LayerValidator: \n"
                                      << post_result.info << "\n *** This LAYER: "
                                      << "\n " << one_tile_layer << " \n strategy: " << (int)strategy << " = "
                                      << VPUTilingStrategy_ToText.at(static_cast<int>(strategy)) << ", nDPU: " << nDPU
                                      << ", nTiles: " << nTiles
                                      << "\nResult: Early termination with Error code:  " << post_result.value()
                                      << " : " << Cycles::toErrorText(post_result.value()) << "\n";
                    break;  // EARLY LOOP exit, otherwise it will be overwritten by next tile check
                }
            }
            if (0 >= tiles_layer.size()) {  // no split present
                post_result.mark_split_error();
            }

            if (!post_result.is_usable()) {
                if (detailed_split) {  // add all tile layers for info
                    for (const auto& one_tile_layer : tiles_layer) {
                        detailed_split->emplace_back(OneTileLayerInfo{
                                one_tile_layer,
                        });  // no workloads. no info if good/bad
                    }
                }

                return post_result.value();  // EARLY RETURN
            }
        }  // inter tile layers sanitized and validated

        // VPUCostModel& dpu_cost_provider(*this);       // this is the cost provider for the DPU workloads
        auto tiler = getDPUTiler(dpu_cost_provider);  // intra-tile tiler
        for (auto& one_tile_layer : tiles_layer) {
            try {
                // obtains the best DPU workloads split
                std::vector<DPUWorkloadsWithCyclesSplit>
                        all_intra_tile_splits{};  ///< all intra tile splits generated. one pair() is a split
                const DPUWorkloadsCost cost_and_workloads = tiler->intraTileSplit(
                        one_tile_layer, options, detailed_split ? &all_intra_tile_splits : nullptr);
                const auto cycles = cost_and_workloads.first;
                tiles_cost.push_back(cycles);

                if (detailed_split) {
                    detailed_split->emplace_back(OneTileLayerInfo{one_tile_layer, std::move(cost_and_workloads),
                                                                  std::move(all_intra_tile_splits)});  // only
                                                                                                       // best
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
    }  // split the layer section, all splits

    // The cost of the worst case in the tiles (since error codes are large the largest error code will dominate
    // any regular value)
    CyclesInterfaceType cost = extractLargestTime(tiles_cost);

    // serialize the complete detailed splits (needs layer info)
    serialization_handler.serializeCyclesAndLayerTilesInfo_closeLine(cost, tiles_layer.size());

    // serialize the complete detailed splits
    // should DO ONLY if no previous serialization error! Think about how to handler these situations
    serialization_handler.serializeLayerSplitInfo(tiles_layer.size(),
                                                  *detailed_split);  // info with cluster_ not with /#, but it should
                                                                     // have been with # fro presplit layers part

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
                    dpu_schedule(get_HWPerformance().get_hw_info(device).get_dma_ports(),  // get the number of DMA
                                                                                           // channels
                                 // GlobalHarwdwareCharacteristics::get_dma_ports(device),
                                 w_costs);  // pipelines on dma channels
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
                static_assert(std::tuple_size<decltype(one_tile_layer.inputs)>::value == 1, "one input restriction");

                const auto& inT{one_tile_layer.inputs[0]};
                CyclesInterfaceType one_tile_cost =
                        compute_dma_cycles({device, static_cast<int>(inT.size()), MemoryDirection::DDR2CMX});
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
                static_assert(std::tuple_size<decltype(one_tile_layer.outputs)>::value == 1, "one input restriction");

                const auto& outT{one_tile_layer.outputs[0]};
                CyclesInterfaceType one_tile_cost =
                        compute_dma_cycles({device, static_cast<int>(outT.size()), MemoryDirection::CMX2DDR});
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

    // serialization_handler.cleanBuffers();

    return cost;
}

CyclesInterfaceType VPULayerCostModel::layer_pre_split_cycles(VPUCostModel& shave_cost_provider,
                                                              const std::vector<SHAVEWorkload>& layers_pre_split,
                                                              unsigned int nSHV, bool input_in_ddr,
                                                              bool output_in_ddr) const {
    // add missing info by deducing it (not anymore received by params)
    unsigned int nTiles{(unsigned int)layers_pre_split.size()};
    VPUDevice device{nTiles ? layers_pre_split[0].get_device() : VPUDevice::__size};
    //  end of missing info

    // sanitize
    if (nTiles == 0) {
        Logger::warning() << "Number of tiles can't be zero, should be at least one!";
        return Cycles::ERROR_L2_INVALID_PARAMETERS;
    }

    std::vector<CyclesInterfaceType> tiles_cost;               // cost of each tile
    std::vector<SHAVEWorkload> tiles_layer{layers_pre_split};  //< layers are already split, make a copy

    // split the layer section, all splits
    {
        {  // tile-layers must be verified to be valid
            SanityReport post_result;

            if (0 >= tiles_layer.size()) {  // no split present
                post_result.mark_split_error();
            }

            if (!post_result.is_usable()) {
                return post_result.value();  // EARLY RETURN
            }
        }  // inter tile layers sanitized and validated

        for (auto& one_tile_layer : tiles_layer) {
            try {
                std::string infoOut{};
                const auto cycles = static_cast<CyclesInterfaceType>(
                        (float)shave_cost_provider.SHAVE(one_tile_layer, infoOut) / (float)nSHV);
                tiles_cost.push_back(cycles);

            } catch (const std::exception& e) {
                Logger::warning() << "\n Exception thrown while performing intra tile split "
                                  << "\n Exception: " << e.what() << "\n *** This LAYER: "
                                  << "\n " << one_tile_layer << ", nSHV: " << nSHV << ", nTiles: " << nTiles
                                  << "\nResult: this tile will have error result ERROR_TILE_SPLIT_EXCEPTION: "
                                  << (CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION << " \n";

                // add the error result
                tiles_cost.push_back((CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION);  // big value
            }
        }
    }  // split the layer section, all splits

    // The cost of the worst case in the tiles (since error codes are large the largest error code will dominate
    // any regular value)
    CyclesInterfaceType cost = extractLargestTime(tiles_cost);

    if (!Cycles::isErrorCode(cost)) {
        // Data fetch time is computed considering the split layers input tensors, that are summed up.
        if (input_in_ddr) {
            // Add cost of loading input activation from DDR to CMX
            std::vector<CyclesInterfaceType> dma_costs;
            for (auto& one_tile_layer : tiles_layer) {
                const auto& inTs{one_tile_layer.get_inputs()};  // Multiple inputs possible
                CyclesInterfaceType one_tile_cost{};
                for (const auto& inT : inTs)
                    one_tile_cost +=
                            compute_dma_cycles({device, static_cast<int>(inT.size()), MemoryDirection::DDR2CMX});
                dma_costs.push_back(one_tile_cost);
            }

            const auto sum_dma_layers = std::accumulate(dma_costs.begin(), dma_costs.end(), 0u, Cycles::cost_adder);
            cost = Cycles::cost_adder(cost, sum_dma_layers);
        }

        // Data fetch time is computed considering the split layers output tensors, that are summed up.
        if (output_in_ddr) {
            // Add cost of spilling output activation from CMX to DDR
            std::vector<CyclesInterfaceType> dma_costs;
            for (auto& one_tile_layer : tiles_layer) {
                const auto& outTs{one_tile_layer.get_outputs()};
                CyclesInterfaceType one_tile_cost{};
                for (const auto& outT : outTs)
                    one_tile_cost +=
                            compute_dma_cycles({device, static_cast<int>(outT.size()), MemoryDirection::CMX2DDR});
                dma_costs.push_back(one_tile_cost);
            }

            const auto sum_dma_layers = std::accumulate(dma_costs.begin(), dma_costs.end(), 0u, Cycles::cost_adder);
            cost = Cycles::cost_adder(cost, sum_dma_layers);
        }

    } else {
        Logger::error() << "\n Layer cost has an error, skipping DMA/memory time computation: "
                        << "ERROR code: " << cost << " : " << Cycles::toErrorText(cost) << "\n"
                        << ", nSHV: " << nSHV << ", nTiles: " << nTiles << "\n";
    }

    return cost;
}

CyclesInterfaceType VPULayerCostModel::Layer(DPULayer& layer, unsigned int nDPU, unsigned int nTiles, bool input_in_ddr,
                                             bool output_in_ddr, bool prefetching) {
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

CyclesInterfaceType VPULayerCostModel::Layer(const SHAVEWorkload& layer, const VPULayerStrategy& strategy) const {
    return Layer(layer, strategy.nSHVs, strategy.nTiles, strategy.input_fetching, strategy.output_spilling);
}

CyclesInterfaceType VPULayerCostModel::Layer(const SHAVEWorkload& layer, const unsigned int nSHV,
                                             const unsigned int nTiles, const bool input_in_ddr,
                                             const bool output_in_ddr) const {
    // For shave layer we use a simplistic model, as we assume the cost can be scale up to 4 tiles
    std::string infoOut;
    const CyclesInterfaceType single_shv_cost = get_SHV_cost_model().SHAVE(layer, infoOut);
    if (Cycles::isErrorCode(single_shv_cost)) {
        Logger::error() << "\n While analyzing a SHAVEWorkload, the cycle time value was with error: "
                        << "ERROR code: " << single_shv_cost << " : " << Cycles::toErrorText(single_shv_cost)
                        << " Layer results will be with ERROR \n"
                        << "INFO: " << infoOut << "\n";

        return single_shv_cost;
    }

    CyclesInterfaceType cost = static_cast<CyclesInterfaceType>((float)single_shv_cost / ((float)(nSHV * nTiles)));

    if (input_in_ddr) {
        // Add cost of loading input activation from DDR to CMX
        for (auto& inT : layer.get_inputs()) {
            auto in_ddr_dma =
                    compute_dma_cycles({layer.get_device(), static_cast<int>(inT.size()), MemoryDirection::DDR2CMX});
            cost = Cycles::cost_adder(cost, in_ddr_dma);
        }
    }

    if (output_in_ddr) {
        // Add cost of spilling output activation from DDR to CMX
        for (auto& outT : layer.get_outputs()) {
            auto out_ddr_dma =
                    compute_dma_cycles({layer.get_device(), static_cast<int>(outT.size()), MemoryDirection::CMX2DDR});
            cost = Cycles::cost_adder(cost, out_ddr_dma);
        }
    }

    return cost;
}

const IDeviceValidValues& VPULayerCostModel::getDeviceConfiguratorForTiles(VPUDevice device) const {
    return the_layer_validator.getDeviceConfiguratorForTiles(device);
}

CyclesInterfaceType VPULayerCostModel::OneTileWeightsPrefetching(const DPULayer& layer) const {
    const auto& config = getDeviceConfiguratorForTiles(layer.device);

    const auto in_1_movable_size = layer.weight_footprint(config);

    const VPUTensor weight_plus_table_tensor{{static_cast<unsigned int>(in_1_movable_size), 1, 1, 1},
                                             DataType::UINT8};  // weight_footprint is in bytes
    auto cost = compute_dma_cycles(
            {layer.device, static_cast<int>(weight_plus_table_tensor.size()), MemoryDirection::DDR2CMX});
    return cost;
}

CyclesInterfaceType VPULayerCostModel::extractLargestTime(const std::vector<CyclesInterfaceType>& tiles_cost) const {
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

void VPULayerCostModel::operation_sanitisation(DPULayer& wl) const {
    get_cost_model().avgpool_replace_by(wl);
    get_cost_model().compressConv_replace_by_CM_CONV_VPU27(wl);
}

std::vector<VPUTilingStrategy> VPULayerCostModel::getValidTilingStrategies(const VPUDevice& device) {
    return LayerPropertiesHolder::get_properties(device).getValidTilingStrategies();
}

}  // namespace VPUNN
