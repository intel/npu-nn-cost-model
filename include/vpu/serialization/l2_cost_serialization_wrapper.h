// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef L2_COST_SERIALIZATION_WRAPPER_H
#define L2_COST_SERIALIZATION_WRAPPER_H
#include "vpu/serialization/serialization_wrapper.h"
#include "vpu_cost_model.h"

namespace VPUNN {

/// @todo: make it a single class with more specialised functions (that will be reused as much as possible)
///        no sense to make a template class that will be inherited by each implementaiton
/* coverity[rule_of_three_violation:FALSE] */
class L2CostSerializationWrap : public CostSerializationWrap {
public:
    struct LayerSerializationContext {
        unsigned int nDPU;
        unsigned int nTiles;
        VPUTilingStrategy strategy;
        VPUDevice device;
    };

private:
    const LayerSerializationContext context;  // const info from user
    const std::string info;                   ///< additional info, to be used in serialization to complete info column
                                              ///< example: in layer_cycles info will be  "/cluster_"
                                              ///< in layer_pre_split_cycles it will be "/#"

private:
    // next are context immutable, and are kind of specific for an application like layer
    const LayersValidation& layer_validator;
    const VPUCostModel& cost_model;

    const std::unique_ptr<LayerSplitInfo> internal_detailed_split_guard;  ///< internal guard for detailed split info,
                                                                          ///< used if caller does not provide one

public:
    inline static const std::string layer_info_content{"/cluster_"};
    inline static const std::string pre_split_info_content{"/#_"};
    /**
     * @brief Constructs a CostSerializationWrap object
     *
     * @param ser            Reference to the CSVSerializer used for output
     * @param validator      Reference to the LayersValidation object for layer validation
     * @param model          Reference to the VPUCostModel used for cost calculations
     * @param split_context_ Specifies the serialization context (LayerCycles or LayerPreSplitCycles)
     * @param detailed_split Reference to a pointer for detailed split information
     *                      - If the caller provides a non-null pointer, it will be used as-is
     *                      - If the caller provides nullptr and serialization is enabled,
     *                        the constructor allocates and manages a local LayerSplitInfo,
     *                        and updates the caller's pointer to point to it
     * @param inhibit        If true, disables serialization regardless of environment settings (default: false).
     *
     * The constructor ensures that detailed_split always points to a valid LayerSplitInfo
     * during the lifetime of this object if serialization is enabled, and manages its lifetime
     * internally if it was allocated here
     */
    L2CostSerializationWrap(CSVSerializer& ser, const LayerSerializationContext ctx, const LayersValidation& validator,
                            const VPUCostModel& model, const std::string& info_,
                            LayerSplitInfo*& detailed_split /*in out*/, bool inhibit = false, size_t the_uid = 0)
            : CostSerializationWrap(ser, inhibit, the_uid),  // initialize the base class
              context{ctx},
              info(info_),
              layer_validator(validator),
              cost_model(model),
              internal_detailed_split_guard{
                      ((is_serialization_enabled(inhibit, ser)) &&
                       (nullptr ==
                        detailed_split)) /* use here the constructor's parameters to verify if serialization is enabled,
                                            because the 'this' object is not fully constructed yet */
                              ? new LayerSplitInfo()  // create empty object
                              : nullptr}              // no object required!

    {
        // allocated here, use it outside
        if (internal_detailed_split_guard) {
            detailed_split = internal_detailed_split_guard.get();
        }
    }

    ~L2CostSerializationWrap() = default;

    /// prevents to multiple wrappers to share the same references
    L2CostSerializationWrap(const L2CostSerializationWrap&) = delete;
    L2CostSerializationWrap& operator=(const L2CostSerializationWrap&) = delete;

    /// vpunn_cycles, n_computed_tiles
    void serializeCyclesAndTilesCnt_closeLine(CyclesInterfaceType cost, const size_t computed_tiles) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;
        // Cost model software DPU development
        try {
            serializer.serialize(SerializableField<CyclesInterfaceType>{"vpunn_cycles", cost});
            serializer.serialize(SerializableField{"n_computed_tiles", computed_tiles});

            serializer.end();  // is it OK to end during a series of writings
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
    }

    /// vpunn_cycles
    /// n_computed_tiles,level=layer, layer_uid
    void serializeCyclesAndLayerTilesInfo_closeLine(CyclesInterfaceType cost, const size_t computed_tiles) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;
        // Cost model software DPU development
        try {
            serializer.serialize(SerializableField<CyclesInterfaceType>{"vpunn_cycles", cost});
            serializer.serialize(SerializableField{"n_computed_tiles", computed_tiles});
            serializer.serialize(SerializableField<std::string>{"level", "layer"});
            serializer.serialize(SerializableField<std::string>{"layer_uid", std::to_string(serializer_operation_uid)});

            serializer.end();  // new line in csv!
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
    }

    /// generic layer info, no cost available
    /// n_requested_tiles, n_dpu, tiling_strategy, level=layer, layer_uid, info, name
    void serializeLayerInformation_header_and_compute_layer_uid(const DPULayer& layer) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;
        try {
            auto& cfg = layer_validator.getDeviceConfiguratorForTiles(context.device);
            const auto op = DPUOperation(layer, cfg);  // not all wl details are relevant at layer
            serializer_operation_uid = op.hash() ^ std::hash<unsigned int>{}(context.nTiles);

            serializer.serialize(op, SerializableField<decltype(context.nTiles)>{"n_requested_tiles", context.nTiles},
                                 SerializableField<decltype(context.nDPU)>{"n_dpu", context.nDPU},
                                 SerializableField<decltype(context.strategy)>{"tiling_strategy", context.strategy},
                                 SerializableField<std::string>{"level", "layer"},
                                 SerializableField<std::string>{"layer_uid", std::to_string(serializer_operation_uid)},
                                 SerializableField<std::string>{"info", layer.get_layer_name()},
                                 SerializableField<std::string>{"name", layer.get_compiler_pass()});

        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
            serializer_operation_uid = 0;  // reset the layer uid in case of error
        }

        return;
        // not ending the csv line!!
    }

    void serializeLayerSplitInfo(const size_t computed_tiles, const LayerSplitInfo& detailed_split) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;

        try {
            auto& cfg = layer_validator.getDeviceConfiguratorForTiles(context.device);
            int idx = 0;

            for (const auto& split : detailed_split) {
                // info  with cluster_
                serializeOneTileLayerInfo_genericInfo(split, cfg, computed_tiles,
                                                      idx);  // own line in csv
                serializeOneTileLayerInfo_allSplits(split, computed_tiles,
                                                    idx);  // info  with cluster_
                idx++;
            }
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
    }

private:
    ///
    /// n_requested_tiles, n_computed_tiles, n_dpu, tiling_strategy, level=layer_tile_split, layer_uid,
    /// info with /cluster_, (is this generic?)
    /// name
    /// vpunn_cycle
    void serializeOneTileLayerInfo_genericInfo(const OneTileLayerInfo& split, const IDeviceValidValues& cfg,
                                               const size_t computed_tiles, int idx) {
        const auto op = DPUOperation(split.inter_tile_split_layer, cfg);

        serializer.serialize(op, SerializableField<decltype(context.nTiles)>{"n_requested_tiles", context.nTiles},
                             SerializableField{"n_computed_tiles", computed_tiles},
                             SerializableField<decltype(context.nDPU)>{"n_dpu", context.nDPU},
                             SerializableField<decltype(context.strategy)>{"tiling_strategy", context.strategy},
                             SerializableField<std::string>{"level", "layer_tile_split"},
                             SerializableField<std::string>{"layer_uid", std::to_string(serializer_operation_uid)},
                             SerializableField<std::string>{"info", split.inter_tile_split_layer.get_layer_name() +
                                                                            info + std::to_string(idx)},
                             SerializableField<std::string>{"name", split.inter_tile_split_layer.get_compiler_pass()});

        serializer.serialize(SerializableField<CyclesInterfaceType>{"vpunn_cycles", split.best_intra_tile_split.first});
        serializer.end();  // new line in csv!
    }

    /// each workload on a line
    /// n_requested_tiles, n_computed_tiles, n_dpu, tiling_strategy, level=intra_tile_split,
    /// layer_uid, info with /cluster_,
    /// intra_tile_seq_id + "its_",
    /// name, workload_uid
    /// vpunn_cycles
    void serializeOneTileLayerInfo_allSplits(const OneTileLayerInfo& split, const size_t computed_tiles, int idx) {
        for (size_t split_idx = 0; split_idx < split.all_intra_tile_splits.size(); split_idx++) {
            const auto& wls = split.all_intra_tile_splits[split_idx];

            for (size_t wl_idx = 0; wl_idx < wls.workloads.size(); wl_idx++) {
                const auto& wl = wls.workloads[wl_idx];
                const auto& wl_cost = wls.cycles[wl_idx];
                auto wl_op = DPUOperation(wl, cost_model.getSanitizerDeviceConfiguration(wl.device));

                auto ss = std::stringstream();
                ss << wl_op;
                std::hash<std::string> hasher;
                const size_t wl_uid = hasher(ss.str());

                serializer.serialize(
                        wl_op, SerializableField<decltype(context.nTiles)>{"n_requested_tiles", context.nTiles},
                        SerializableField{"n_computed_tiles", computed_tiles},
                        SerializableField<decltype(context.nDPU)>{"n_dpu", context.nDPU},
                        SerializableField<decltype(context.strategy)>{"tiling_strategy", context.strategy},
                        SerializableField<std::string>{"level", "intra_tile_split"},
                        SerializableField<std::string>{"layer_uid", std::to_string(serializer_operation_uid)},
                        SerializableField<std::string>{
                                "info", split.inter_tile_split_layer.get_layer_name() + info + std::to_string(idx)},
                        SerializableField<std::string>{"intra_tile_seq_id", "its_" + std::to_string(split_idx)},
                        SerializableField<std::string>{"name", split.inter_tile_split_layer.get_compiler_pass()},
                        SerializableField<std::string>{"workload_uid", std::to_string(wl_uid)});

                serializer.serialize(SerializableField<CyclesInterfaceType>{"vpunn_cycles", wl_cost});
                serializer.end();  // new line in csv!
            }
        }
    }
};

}  // namespace VPUNN

#endif  // VPUNN_SERIALIZATION_WRAPPER_H
