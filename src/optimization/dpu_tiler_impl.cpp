// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/dpu_tiler_impl.h"
#include "vpu/device_layer_properties/device_layer_properties_holder.h"

#include <algorithm>
#include <exception>
#include <list>
#include <string>
#include <vector>

#include "core/logger.h"
#include "vpu/optimization/efficiency_based_splitter.h"
#include "vpu/optimization/tiling_algorithm_factory.h"
#include "vpu/optimization/workload_cost_evaluator.h"
#include "vpu/optimization/zefftiling.h"
#include "vpu/utils.h"

namespace VPUNN {

ZEffTiling* DPUTilerImplementation::findZEffTilingAlgorithm(const TilingAlgorithmsContainer& algorithms) const {
    for (auto& algo : algorithms) {
        if (auto* zeff_algo = dynamic_cast<ZEffTiling*>(algo.get())) {
            return zeff_algo;
        }
    }
    return nullptr;
}

bool DPUTilerImplementation::hasZEffTilingAlgorithm(const TilingAlgorithmsContainer& algorithms) const {
    return findZEffTilingAlgorithm(algorithms) != nullptr;
}

std::list<DPUWorkloadsWithCycleCost> DPUTilerImplementation::generateSplits_processZEffTilingSplits(
        const TilingAlgorithmsContainer& algorithms, const DPULayer& layer,
        const std::vector<ExecutionMode>& valid_execution_modes, const SplitOptions& options) const {
    std::list<DPUWorkloadsWithCycleCost> splits_costs;

    // Find ZEffTiling if present — it follows its own simple path
    ZEffTiling* const zeff_algo = findZEffTilingAlgorithm(algorithms);

    if (zeff_algo != nullptr) {
        // ZEff path: measure candidate block costs, hand them to the algo, collect result
        const unsigned int output_channels = layer.outputs[0].channels();
        const unsigned int max_block_size = output_channels < 32 ? 16 : (output_channels / 32) * 32;
        // for dense-activation 1x1 DW convolutions with output spatial dimensions <= 16 in both X and Y,
        // workload_size_z must not exceed 32. Limit candidate block sizes to {16, 32} when this condition is satisfied.
        // Limit candidate block sizes for DW/Pool ops with 1x1 kernel and small output spatial dims
        const bool isDWLimitedCase = layer.act_sparsity == 0.0f && layer.kernels[0] == 1u && layer.kernels[1] == 1u &&
                                     layer.outputs[0].width() <= 16u && layer.outputs[0].height() <= 16u &&
                                     (layer.op == Operation::DW_CONVOLUTION  //
                                      || layer.op == Operation::MAXPOOL      //
                                      || layer.op == Operation::AVEPOOL);

        unsigned int effective_max_block_size = max_block_size;
        if (isDWLimitedCase)
            effective_max_block_size = std::min(32u, max_block_size);

        SmartRanges candidate_range(16, std::min(256u, effective_max_block_size), 16, 32);
        const auto candidate_channels_sizes = candidate_range.transformSmartRangetoVector<unsigned int>();

        std::vector<unsigned int> remainder_block_sizes;
        if (output_channels % 16 != 0 && layer.is_output_autopad()) {
            // Directly enumerate all valid remainder sizes with the correct residue.
            // Each candidate R satisfies R % 16 == (output_channels % 16) and 0 < R <= output_channels.
            // This avoids the gaps that arise from the SmartRanges(16, X, 16, 32) -> adjust approach,
            // which skips raw sizes not divisible by 32 (e.g. raw=48 never appears, so adjusted=33 was missing).
            const unsigned int residue = output_channels % 16;
            for (unsigned int r = residue; r <= output_channels; r += 16) {
                remainder_block_sizes.push_back(r);
            }

            // For depthwise-style ops (DW_CONVOLUTION, AVEPOOL, MAXPOOL), input and output channels
            // are tightly coupled: each output channel slice requires a matching input channel slice.
            // The prefix tiles (aligned, multiples of 16) each consume exactly their output channel
            // count from the layer's input channels. The remainder tile may require more input channels
            // than its output count because createTestLayer applies alignment:
            //   - remainder < 16, no autopad  => padded up to 16 input channels
            //   - remainder < 16, autopad      => exactly r input channels (no padding needed)
            //   - remainder >= 16              => rounded up to the next multiple of 32 being 
            //                                    the only valid alignment for dwconv family over 16 channels
            // Total input channels needed = (output_channels - r)   [prefix tiles]
            //                             + in_ch_for_remainder      [remainder tile]
            // Filter out any remainder size r where this total exceeds the layer's actual input channels.
            if (is_dwconv_family_operation(layer.op)) {
                const unsigned int layer_in_ch = layer.inputs[0].channels();

                // Mirrors the input-channel alignment applied in createTestLayer.
                auto input_channels_for_remainder = [&](unsigned int r) -> unsigned int {
                    if (r < 16u && layer.is_input_autopad()) {
                        // Sub-16 remainder with autopad: keep exact channel count
                        return r;
                    }
                    // Apply DW-style alignment: <=16 → 16, >16 → round_up(r, 32)
                    return dw_channel_align(r);
                };

                // Remove any remainder block size r where the total input channels needed exceeds the 
                // layer's input channels. This check ensures we only consider valid tiling configurations 
                // that respect the tight coupling of input and output channels for these ops.
                remainder_block_sizes.erase(
                        std::remove_if(remainder_block_sizes.begin(), remainder_block_sizes.end(),
                                       [&](unsigned int r) {
                                           const unsigned int prefix_in_ch = output_channels - r;
                                           const unsigned int remainder_in_ch = input_channels_for_remainder(r);
                                           return (prefix_in_ch + remainder_in_ch) > layer_in_ch;
                                       }),
                        remainder_block_sizes.end());
            }
        }
        for (const auto& mode : valid_execution_modes) {
            // Generate splits will make sure that we avoid cases of autopad and invalid ops that should not be runned
            const auto nWorkloads = *(zeff_algo->generateSplitPool(options.nDPU, mode)).begin();

            //Skip block generation and splitting when no tiling is to be applied (nWorkloads == 1) to avoid unnecessary overhead and 
            // potential issues with zero block sizes
            if (nWorkloads == 0) {
                const auto blocks_of_channels =
                        evaluator_.measureCandidateBlocks(layer, mode, candidate_channels_sizes);
                if (blocks_of_channels.empty())
                    continue;

                zeff_algo->setBlocks(blocks_of_channels);

                if(output_channels % 16 != 0 && layer.is_output_autopad()) {
                    const auto remainder_blocks = evaluator_.measureCandidateBlocks(layer, mode, remainder_block_sizes);
                    if (remainder_blocks.empty())
                        continue;

                    zeff_algo->setRemainderBlocks(remainder_blocks);
                }
            }

            // If the nWls is 1 it means it should simply infer and skip,otherwise if it is on 0 it will search for best split.
            for (auto& workloads : zeff_algo->split_tile_in_workloads(mode, nWorkloads)) {
                const auto cycles = computeSplitCycles(workloads, options.runtimeOverhead);
                splits_costs.push_back({cycles <= 0 ? Cycles::ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT : cycles, workloads});
            }
        }
    }

    return splits_costs;
}

std::list<DPUWorkloadsWithCycleCost> DPUTilerImplementation::generateSplits(
        const TilingAlgorithmsContainer& algorithms, const std::vector<ExecutionMode>& valid_execution_modes,
        const SplitOptions& options) const {
    std::list<DPUWorkloadsWithCycleCost> splits_costs;
    // Loop algorithms, splits, modes and populate the DPUWorkloadsCost list
    auto timeout = SyncStopWatch<std::micro>();
    if (options.maxLatencyUs > 0)
        timeout.start();

    for (auto& algo : algorithms) {
        for (auto& mode : valid_execution_modes) {
            // in how many pieces to be tried to be split
            const auto split_count_variants = algo->generateSplitPool(options.nDPU, mode);
            for (auto nWorkloads : split_count_variants) {
                // Return if the max time has elapsed
                if (options.maxLatencyUs > 0 && timeout.interval() > options.maxLatencyUs) {
                    return splits_costs;
                }

                // populates splitVariants with 0, 1 or more workloads vectors
                std::list<DPUWorkloadsWithCyclesSplit> splitVariants{algo->split_tile_in_workloads(mode, nWorkloads)};
                for (auto& workloads : splitVariants) {
                    // measure  this variant. try catch , and check its output for errors
                    try {
                        const auto cycles = computeSplitCycles(workloads, options.runtimeOverhead);  // may throw

                        const CyclesInterfaceType wl_cost{
                                cycles <= 0 ? Cycles::ERROR_TILE_SPLIT_ZERO_CYC_OUTPUT  // no zero allowed
                                            : cycles};

                        if (Cycles::isErrorCode(wl_cost)) {
                            Logger::warning() << "\n Error result (or zero cycles) while computing the performance "
                                              << "of workloads split variants! "
                                              << "ERROR code: " << wl_cost << " : " << Cycles::toErrorText(wl_cost)
                                              << "\n Execution mode: " << (int)mode << " : "
                                              << ExecutionMode_ToText.at(static_cast<int>(mode))
                                              << "\n nWorkloads: " << nWorkloads << "\n Algo : " << algo->name()
                                              << " \n Result: ignoring the cost of this workloads split \n";
                        }

                        // good or bad we keep the result
                        splits_costs.push_back({wl_cost, workloads});

                    } catch (const std::exception& e) {
                        Logger::warning()
                                << "\n Exception thrown while computing the performance of workloads "
                                << "split variants! "
                                << "\n Execution mode: " << (int)mode << " : "
                                << ExecutionMode_ToText.at(static_cast<int>(mode)) << "\n nWorkloads: " << nWorkloads
                                << "\n Algo : " << algo->name() << "\n Exception: " << e.what() << "\n "
                                << "\nResult: ignoring the cost of this workloads split \n";

                        // add the error result
                        splits_costs.push_back({(CyclesInterfaceType)Cycles::ERROR_TILE_SPLIT_EXCEPTION, workloads});
                    }

                }  // cost of workloads
            }
        }
    }
    return splits_costs;
}

DPUWorkloadsCost DPUTilerImplementation::intraTileSplit(
        const DPULayer& layer, const SplitOptions& options,
        std::vector<DPUWorkloadsWithCyclesSplit>* complete_output_splits) const {
    // Get execution modes accepted  (e.g.: ExecutionMode::CUBOID_16x16,.....)
    const auto valid_execution_modes = LayerPropertiesHolder::get_properties(layer.device)
                                               .getValidTilingExecutionMode(layer);  // based on operation

    // get all in-tile tiling algorithms. Each algo has a copy of Layer.
    TilingAlgorithmsContainer algorithms{TilingAlgorithmFactory::getTilingAlgorithms(layer, options)};

    // Compute the cost of each split type.
    // compute splits(one is a vector of DPUWorkload)  and cost for each split.
    std::list<DPUWorkloadsWithCycleCost> splits_costs;

    if (hasZEffTilingAlgorithm(algorithms)) {
        // Super special handling if at least one of the algorithms is ZEffTiling,
        // expects that this is the only one algorithm in the container, otherwise the behavior is not good
        splits_costs = generateSplits_processZEffTilingSplits(algorithms, layer, valid_execution_modes, options);
    } else {
        splits_costs = generateSplits(algorithms, valid_execution_modes, options);
    }

    if (splits_costs.empty()) {
        throw_error<std::runtime_error>("intraTileSplit: no valid workload generated");
    }

    if (complete_output_splits != nullptr) {
        for (const auto& split : splits_costs) {
            complete_output_splits->push_back(split.second);  // second is the DPUWorkloads
        }
    }

    // lambda comparator for obtaining the minimum one that has no errors and is not zero!
    auto comp = [](const DPUWorkloadsWithCycleCost& a, const DPUWorkloadsWithCycleCost& b) {
        // zero is not a min candidate
        // error is not a min candidate
        // .first is the  cycle time of the workloads split
        if (Cycles::isErrorCode(a.first) || a.first <= 0) {
            return false;  // a not < b, b might be good or not. If both bad they are equal
        }
        // a is valid here
        if (Cycles::isErrorCode(b.first) || b.first <= 0) {
            return true;  // keep a<b if b is invalid value, and "a" valid
        }
        // both valid
        return (a.first) < (b.first);
    };

    // Return the split with min cost (the optimal one). or the first error code (or zero)
    // iterator to min
    const auto& minimum_split{std::min_element(splits_costs.begin(), splits_costs.end(), comp)};

    return {minimum_split->first, minimum_split->second.workloads};  // DPUWorkloadsCost pair
}

CyclesInterfaceType DPUTilerImplementation::computeSplitCycles(DPUWorkloadsWithCyclesSplit& workloads_split,
                                                               const unsigned int runtimeOverhead) const {
    // Delegate all cycle computation logic to the evaluator
    return evaluator_.computeSplitCycles(workloads_split, runtimeOverhead);
}

}  // namespace VPUNN