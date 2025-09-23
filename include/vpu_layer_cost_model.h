// Copyright © 2024 Intel Corporation
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
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

#include "core/logger.h"
#include "core/serializer.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dpu_defaults.h"
#include "vpu/layer.h"
#include "vpu/optimization/workload_optimization.h"
#include "vpu/performance.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpu/validation/interface_valid_values.h"
#include "vpu/validation/layer_sanitizer.h"
#include "vpu_cost_model.h"

#include "vpu/vpu_tiling_strategy.h"
#include "vpu_dma_cost_model.h"
#include "vpu_dma_cost_model_variant.h"
#include "vpu_layer_strategy.h"

#include "vpu/layer_split_info.h"

#include "vpu/serialization/l2_cost_serialization_wrapper.h"
#include "vpu/device_layer_properties/device_layer_properties_holder.h"

namespace VPUNN {

/// @brief The VPUNN layer cost model (also called VPUNN Level2 API)
class VPUNN_API VPULayerCostModel {
private:
    /// DPU cost provider, used for DPU workloads
    std::shared_ptr<VPUCostModel> ptr_internal_dpu_cost_provider;  //< shared ownership of L1 , either received from
                                                                   // outside or internally created
    VPUCostModel& internal_dpu_cost_provider{
            *ptr_internal_dpu_cost_provider};  ///< DPU cost provider, used for DPU workloads

    const LayersValidation the_layer_validator{};  ///< used for validating the un-split layers and split layers
    static constexpr unsigned int default_maxWorkloadsPerIntraTileSplit{128U};  ///< default max splits for a tile
    unsigned int maxWorkloadsPerIntraTileSplit{default_maxWorkloadsPerIntraTileSplit};  ///< max splits for a tile

    const DMACostModelVariant the_dma_cost_model{static_cast<DMACostModel<DMANNWorkload_NPU27>*>(
            nullptr)};  ///< Variant that holds a DMACostModel pointer (non const). External provider!

    mutable CSVSerializer serializer{};  ///< Serializer for the VPULayerCostModel, has its own file as output
    mutable CSVSerializer
            presplit_serializer{};  ///< Serializer for the VPULayerCostModel (presplit api), has its own file as output

public:
    /// @brief Get the CM, either base or a contained object or maybe a parametric attribute
    VPUCostModel& get_cost_model() noexcept {
        return internal_dpu_cost_provider;
    }

    const VPUCostModel& get_cost_model() const noexcept {
        return internal_dpu_cost_provider;
    }

    const VPUCostModel& get_SHV_cost_model() const noexcept {
        const VPUCostModel& retv{internal_dpu_cost_provider};
        return retv;
    }
    const VPUCostModel& get_TheoreticalDMA_cost_model() const noexcept {
        const VPUCostModel& retv{internal_dpu_cost_provider};
        return retv;
    }

    const HWPerformanceModel& get_HWPerformance() const {
        return internal_dpu_cost_provider.getPerformanceModel();
    }

    // sharing the internal  shared pointers for a temporary easier integration in VPUX
    std::shared_ptr<VPUCostModel> get_cost_model_shared() noexcept {
        return ptr_internal_dpu_cost_provider;
    }
    const std::shared_ptr<VPUCostModel> get_cost_model_shared() const noexcept {
        return ptr_internal_dpu_cost_provider;
    }
    const std::shared_ptr<VPUCostModel> get_SHV_cost_model_shared() const noexcept {
        return ptr_internal_dpu_cost_provider;
    }
    const std::shared_ptr<VPUCostModel> get_TheoreticalDMA_cost_model_shared() const noexcept {
        return ptr_internal_dpu_cost_provider;
    }

    /// Shortcut to DPU counter.
    /// in the future handle also non existing case, or add aggregate counters
    /// Do not hold this reference, it may become obsolete if the Layer/DPU is reconfigured
    const AccessCounter& getDPUPreloadedCacheCounter() const {
        return get_cost_model().getPreloadedCacheCounter();
    }

    /// @brief Get a reference to the serializer.
    /// temporary only for testing aspects (extra save ). TO BE REFACTORED
    CSVSerializer& get_serializer() noexcept {
        // here we return a reference, we do not need to protect the serializer => suppression
        /* coverity[missing_lock:FALSE] */
        return serializer;
    }

    //////////////////// Constructors section

    /// In order to inject a DMACostModel, need to extend base constructor
    /**
     * @brief Construct a new VPULayerCostModel object,  DPUL1 from File, DMA from outside
     *
     * @param dma_cost_model Pointer to a valid DMACostModel - see DMACostModelVariant for available types
     * @param filename the name of the .vpunn model
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit VPULayerCostModel(
            const DMACostModelVariant dma_cost_model,      ///< dma from outside, as pointer, non owning
            const std::string& filename = "",              ///< *.vpunn file for DPU model
            bool profile = false,                          ///< enable/disable profiling (DPU?), LEGACY
            const unsigned int cache_size = 16384,         ///< the size of the dynamic cache for DPU model
            const unsigned int batch_size = 1,             ///< model batch size for DPU
            const std::string& dpu_cache_filename = "",    /// < the name of the preloaded cache file for DPU model
            const std::string& shave_cache_filename = "",  /// < the name of the preloaded cache file for SHAVE
            bool use_shave_2_api = false,                  /// < Use the Shave2 API 
            bool tryToLoadPairedCache = false              ///< see L1 DPU constructor details for this
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(filename, profile, cache_size, batch_size,
                                                                            dpu_cache_filename, shave_cache_filename,
                                                                            use_shave_2_api, tryToLoadPairedCache)},
              the_dma_cost_model(dma_cost_model) {
        initialize_serializers();
    }

    /**
     * @brief Construct a new VPULayerCostModel object,  DPUL1 from buffer, DMA from outside
     *
     * @param model_data a buffer containing a .vpunn model
     * @param model_data_length the size of the model_data buffer
     * @param copy_model_data enable/disable the memcopy of the buffer
     * @param dma_cost_model Pointer to a valid DMACostModel - see DMACostModelVariant for available types
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit VPULayerCostModel(
            const char* model_data,                    ///> a buffer containing a .vpunn model
            size_t model_data_length,                  ///< the size of the model_data buffer
            bool copy_model_data,                      ///> enable/disable the memcopy of the buffer
            const DMACostModelVariant dma_cost_model,  ///< dma from outside, as pointer, non owning
            bool profile = false,                      ///< enable/disable profiling (DPU?), LEGACY
            const unsigned int cache_size = 16384,     ///< the size of the dynamic cache for DPU model
            const unsigned int batch_size = 1,         ///< model batch size for DPU
            const char* dpu_cache_data = nullptr,      /// < the content of the preloaded cache file for DPU model
            size_t dpu_cache_data_length = 0,          ///< the size of the dpu_cache_data buffer
            const char* shave_cache_data = nullptr,    ///< the content of the preloaded cache file for SHAVE
            size_t shave_cache_data_length = 0,        /// < the size of the shave_cache_data buffer
            bool use_shave_2_api = false               /// < Enable usage of SHAVE2 api
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(
                      model_data, model_data_length, copy_model_data, profile, cache_size, batch_size, dpu_cache_data,
                      dpu_cache_data_length, shave_cache_data, shave_cache_data_length, use_shave_2_api)},
              the_dma_cost_model(dma_cost_model) {
        initialize_serializers();
    }

    /// Allow creation of VPULayerCostModel without DMA model (fallback to theoretical model in order to maintain older
    /// uses of VPULayerCostModel). To be redesigned to receive a L1 DMA that used theoretical model, agnostic L2..
    /**
     * @brief Construct a new VPULayerCostModel object,   DPU L1 from File, DMA from outsideNULL, use theoretical
     * model
     *
     * @param filename the name of the .vpunn model
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit VPULayerCostModel(
            const std::string& filename = "",              ///< *.vpunn file for DPU model
            bool profile = false,                          ///< enable/disable profiling (DPU?), LEGACY
            const unsigned int cache_size = 16384,         ///< the size of the dynamic cache for DPU model
            const unsigned int batch_size = 1,             ///< model batch size for DPU
            const std::string& dpu_cache_filename = "",    /// < the name of the preloaded cache file for DPU model
            const std::string& shave_cache_filename = "",  /// < the name of the preloaded cache file for SHAVE
            bool use_shave_2_api = false,                  /// < Enable usage of Shave2 API
            bool tryToLoadPairedCache = false              ///< see L1 DPU constructor details for this
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(filename, profile, cache_size, batch_size,
                                                                            dpu_cache_filename, shave_cache_filename, 
                                                                            use_shave_2_api, tryToLoadPairedCache)} {
        initialize_serializers();
    }

    /**
     * @brief Construct a new VPULayerCostModel object ,   DPU L1 from buffer, DMA from outsideNULL, use theoretical
     * model
     *
     * @param model_data a buffer containing a .vpunn model
     * @param model_data_length the size of the model_data buffer
     * @param copy_model_data enable/disable the memcopy of the buffer
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit VPULayerCostModel(
            const char* model_data,                  ///> a buffer containing a .vpunn model
            size_t model_data_length,                ///< the size of the model_data buffer
            bool copy_model_data,                    ///> enable/disable the memcopy of the buffer
            bool profile = false,                    ///< enable/disable profiling (DPU?), LEGACY
            const unsigned int cache_size = 16384,   ///< the size of the dynamic cache for DPU model
            const unsigned int batch_size = 1,       ///< model batch size for DPU
            const char* dpu_cache_data = nullptr,    /// < the content of the preloaded cache file for DPU model
            size_t dpu_cache_data_length = 0,        ///< the size of the dpu_cache_data buffer
            const char* shave_cache_data = nullptr,  ///< the content of the preloaded cache file for SHAVE
            size_t shave_cache_data_length = 0,       /// < the size of the shave_cache_data buffer
            bool use_shave_2_api = false               /// < Enable usage of SHAVE2 api
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(
                      model_data, model_data_length, copy_model_data, profile, cache_size, batch_size, dpu_cache_data,
                      dpu_cache_data_length, shave_cache_data, shave_cache_data_length, use_shave_2_api)} {
        initialize_serializers();
    }

    /// @brief Construct a new VPULayerCostModel object,  DPU L1 from outside, DMA from outside
    explicit VPULayerCostModel(
            std::shared_ptr<VPUCostModel> dpu_cost_provider,  ///< shared pointer to a valid VPUCostModel
            const DMACostModelVariant dma_cost_model          ///< dma from outside, as pointer in a variant, non owning
            )
            : ptr_internal_dpu_cost_provider{std::move(dpu_cost_provider)},
              the_dma_cost_model(std::move(dma_cost_model)) {
        initialize_serializers();
    }

    /// @brief Construct a new VPULayerCostModel object,  DPU L1 from outside, DMA from inside, theoretical
    explicit VPULayerCostModel(
            std::shared_ptr<VPUCostModel> dpu_cost_provider  ///< shared pointer to a valid VPUCostModel
            )
            : ptr_internal_dpu_cost_provider{std::move(dpu_cost_provider)} {
        initialize_serializers();
    }

    //////////////////////////// Constructors section END

protected:
    /// common code for initializing serializers
    void initialize_serializers() {
        // TODO: better move the serializers in a subobject that makes the init in its constructor
        serializer.initialize("l2_dpu_workloads", FileMode::READ_WRITE, get_names_for_serializer());
        presplit_serializer.initialize("l2_dpu_workloads_presplit", FileMode::READ_WRITE, get_names_for_serializer());
    }

public:
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
        // VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
        return layer_cycles(internal_dpu_cost_provider, layer, strategy.tiling_strategy, strategy.nDPUs,
                            strategy.nTiles, strategy.input_fetching, strategy.output_spilling, strategy.prefetching);
    }

    CyclesInterfaceType Layer(DPULayer& layer, VPULayerStrategy strategy, LayerSplitInfo& detailed_split) {
        // VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
        return layer_cycles(internal_dpu_cost_provider, layer, strategy.tiling_strategy, strategy.nDPUs,
                            strategy.nTiles, strategy.input_fetching, strategy.output_spilling, strategy.prefetching,
                            &detailed_split);
    }

    // layer for dCIiM targeted layers, just rename the prev 2 methods. Alternative is to absorb the dCiM or SCL flag
    // inside of VPULayerStrategy

    CyclesInterfaceType Layer_dCiM(DPULayer& layer, VPULayerStrategy strategy) {
        Logger::info() << "DCIM LAYER:" << layer.get_layer_info();
        Logger::info() << strategy;
        return Cycles::ERROR_TILE_SPLIT_EXCEPTION;
    }

    CyclesInterfaceType Layer_dCiM(DPULayer& layer, VPULayerStrategy strategy, LayerSplitInfo& detailed_split) {
        Logger::info() << "DCIM LAYER:" << layer.get_layer_info();
        Logger::info() << strategy;
        detailed_split.clear();
        return Cycles::ERROR_TILE_SPLIT_EXCEPTION;
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
        // VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
        return layer_cycles(internal_dpu_cost_provider, layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr,
                            prefetching, nullptr);
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
        //        VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
        return layer_cycles(internal_dpu_cost_provider, layer, strategy, nDPU, nTiles, input_in_ddr, output_in_ddr,
                            prefetching, &detailed_split);
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
     * @param fullLayerHash [in] is a hash computed by caller, should reflect the initial Layer. Will be used to group
     * the splits in the statistics
     * @param strategyOfSplit [in] is a strategy that was used to split the layer. Should be passed where the strategy
     * is selected/decided(MC pass). If not available the vpunn might decide not to do any data serialization fro
     * statistics at L2
     *
     * @return measured best cycles for the overall vector of layers or error code . \see Cycles for error codes
     */
    CyclesInterfaceType LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                       bool input_in_ddr, bool output_in_ddr, bool prefetching,
                                       LayerSplitInfo& detailed_split,
                                       const size_t fullLayerHash = 0,  // hash on layer only, computed by VPUX
                                       const std::optional<VPUTilingStrategy> strategyOfSplit =
                                               (std::optional<VPUTilingStrategy>())  // to be sent only for MC pass
    ) {
        //        VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
        return layer_pre_split_cycles(internal_dpu_cost_provider, layers_pre_split, nDPU, input_in_ddr, output_in_ddr,
                                      prefetching, &detailed_split, fullLayerHash, strategyOfSplit);
    }

    /// version without detailed split output parameter and no hash or tiling strategy.
    CyclesInterfaceType LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                       bool input_in_ddr, bool output_in_ddr, bool prefetching) {
        //        VPUCostModel& dpu_cost_provider(*this);  // for now the DPU cost provider is inherited
        return layer_pre_split_cycles(internal_dpu_cost_provider, layers_pre_split, nDPU, input_in_ddr, output_in_ddr,
                                      prefetching, nullptr);
    }

    /// version without detailed split output parameter and no hash or tiling strategy.
    CyclesInterfaceType LayersPreSplit(const std::vector<SHAVEWorkload>& layers_pre_split, unsigned int nSHV,
                                               bool input_in_ddr, bool output_in_ddr) {
        return layer_pre_split_cycles(internal_dpu_cost_provider, layers_pre_split, nSHV, input_in_ddr, output_in_ddr);
    }

protected:
    /**
     * @brief Check if the DMACostModelVariant holds a valid DMACostModel
     *
     * @return true if a DMACostModel is initialized, false otherwise
     */
    bool is_dma_model_variant() const {
        return std::visit(
                [](const auto& dma_model) {
                    return dma_model && dma_model->nn_initialized();
                },
                the_dma_cost_model);
    };
    /**
     * @brief Compute the cost of a DMA operation
     *
     * If the DMACostModel is not valid it will try to use the theoretical model.
     * Otherwise, it will check if the device set in the workload corresponds to initialized
     * DMACostModel and will compute the cost
     * If the final cost (from DMA NN model or theoretical model) is an error code, it will log an error message
     * @param dwl A simple DMA descriptor (considers a simplistic 1D array transfer)
     * @return measured best cycles or error code. \see Cycles for error codes
     */
    CyclesInterfaceType compute_dma_cycles(const DMATransfer1D& dwl) const {
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
                        case VPUDevice::NPU_RESERVED:
                        case VPUDevice::NPU_RESERVED_W:
                            if constexpr (std::is_same_v<WLType, DMANNWorkload_NPU_RESERVED>)
                                device_match = true;
                            break;
                        default:
                            break;
                        }
                        return device_match
                                       ? dma_model->computeCycles(DMANNWorkloadCreator<WLType>::create_workload(dwl))
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

    // template <bool B = serialization_enabled, typename std::enable_if<B, int>::type = 0>
    static const std::vector<std::string> get_names_for_serializer() {
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
     * @return measured best cycles or error code. \see Cycles for error codes
     */
    CyclesInterfaceType layer_cycles(VPUCostModel& dpu_cost_provider,  // cost model to be used
                                     DPULayer& layer, VPUTilingStrategy strategy, unsigned int nDPU = 1,
                                     unsigned int nTiles = 1, bool input_in_ddr = false, bool output_in_ddr = false,
                                     bool prefetching = true, LayerSplitInfo* detailed_split = nullptr) const {
        dpu_cost_provider.swizzling_turn_OFF(layer);

        if (nTiles == 0) {
            Logger::warning() << "Number of tiles can't be zero, should be at least one!";
            return Cycles::ERROR_L2_INVALID_PARAMETERS;
        }

        VPUDevice device{nTiles ? layer.device : VPUDevice::__size};

        // serialization init logic
        const L2CostSerializationWrap::LayerSerializationContext layer_cyc_ctxt{nDPU, nTiles, strategy, device};
        L2CostSerializationWrap serialization_handler(serializer, layer_cyc_ctxt, the_layer_validator,
                                                      dpu_cost_provider, L2CostSerializationWrap::layer_info_content,
                                                      detailed_split);

        if (detailed_split) {  // always
            detailed_split->clear();
        }

        serialization_handler.serializeLayerInformation_header_and_compute_layer_uid(
                layer);  // must keep the csv line open!

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
                    Logger::warning() << "\n Layer is NOT Valid \n *** INFO from LayerValidator:\n "
                                      << unsplit_result.info << "\n"
                                      << "\n *** LAYER: "
                                      << "\n " << layer << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU
                                      << ", nTiles: " << nTiles
                                      << "\n Result: Early termination with Error code: " << unsplit_result.value()
                                      << " : " << Cycles::toErrorText(unsplit_result.value()) << "\n";

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
                                      << "\n Exception: " << e.what() << "\n " << layer
                                      << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU
                                      << ", nTiles: " << nTiles
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
                const auto pipelined_cost = dpu_schedule(get_HWPerformance()
                                                                 .get_hw_info(layer.device)
                                                                 .get_dma_ports(),  // get the number of DMA channels
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
                            << " \n strategy: " << (int)strategy << ", nDPU: " << nDPU << ", nTiles: " << nTiles
                            << "\n";
        }

        // serialization_handler.cleanBuffers();

        return cost;
    }

    /// like Layer but with pre-split layers
    CyclesInterfaceType layer_pre_split_cycles(
            VPUCostModel& dpu_cost_provider,  // cost model to be used
            const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU = 1, bool input_in_ddr = false,
            bool output_in_ddr = false, bool prefetching = true, LayerSplitInfo* detailed_split = nullptr,
            const size_t fullLayerHash = 0,  // hash on layer only, computed by VPUX
            const std::optional<VPUTilingStrategy> strategyOfSplit =
                    (std::optional<VPUTilingStrategy>())  // to be sent only for MC pass
    ) const {
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
                                                      dpu_cost_provider,
                                                      L2CostSerializationWrap::pre_split_info_content, detailed_split,
                                                      is_serialization_inhibited, fullLayerHash);

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
                                          << VPUTilingStrategy_ToText.at(static_cast<int>(strategy))
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
        serialization_handler.serializeLayerSplitInfo(
                tiles_layer.size(), *detailed_split);  // info with cluster_ not with /#, but it should
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
                const auto pipelined_cost = dpu_schedule(
                        get_HWPerformance().get_hw_info(device).get_dma_ports(),  // get the number of DMA channels
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
                    static_assert(std::tuple_size<decltype(one_tile_layer.inputs)>::value == 1,
                                  "one input restriction");

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
                    static_assert(std::tuple_size<decltype(one_tile_layer.outputs)>::value == 1,
                                  "one input restriction");

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

    CyclesInterfaceType layer_pre_split_cycles(
            VPUCostModel& shave_cost_provider, // cost model to be used
            const std::vector<SHAVEWorkload>& layers_pre_split, unsigned int nSHV = 1, 
            bool input_in_ddr = false, bool output_in_ddr = false
    ) const {
        // add missing info by deducing it (not anymore received by params)
        unsigned int nTiles{(unsigned int)layers_pre_split.size()};
        VPUDevice device{nTiles ? layers_pre_split[0].get_device() : VPUDevice::__size};
        //  end of missing info

        // sanitize
        if (nTiles == 0) {
            Logger::warning() << "Number of tiles can't be zero, should be at least one!";
            return Cycles::ERROR_L2_INVALID_PARAMETERS;
        }

        std::vector<CyclesInterfaceType> tiles_cost;          // cost of each tile
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
                    const auto cycles =static_cast<CyclesInterfaceType>((float)shave_cost_provider.SHAVE(one_tile_layer, infoOut) / (float)nSHV);
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
                    const auto& inTs{one_tile_layer.get_inputs()}; // Multiple inputs possible
                    CyclesInterfaceType one_tile_cost{};
                    for(const auto& inT : inTs)
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
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType Layer(const SWOperation& layer, const VPULayerStrategy& strategy) const {
        return Layer(layer, strategy.nSHVs, strategy.nTiles, strategy.input_fetching, strategy.output_spilling);
    }

    /**
     * @brief Compute the optimal cost of a SHAVEWorkload
     *
     * @param layer the SHV kernel
     * @param strategy the layer strategy
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType Layer(const SHAVEWorkload& layer, const VPULayerStrategy& strategy) const {
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
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType Layer(const SWOperation& layer, const unsigned int nSHV = 1, const unsigned int nTiles = 1,
                              const bool input_in_ddr = false, const bool output_in_ddr = false) const {
        // For shave layer we use a simplistic model, as we assume the cost can be scale up to 4 tiles
        const CyclesInterfaceType single_shv_cost = get_SHV_cost_model().SHAVE(layer);
        CyclesInterfaceType cost = static_cast<CyclesInterfaceType>((float)single_shv_cost / ((float)(nSHV * nTiles)));

        if (input_in_ddr) {
            // Add cost of loading input activation from DDR to CMX
            for (auto& inT : layer.inputs) {
                auto in_ddr_dma =
                        compute_dma_cycles({layer.device, static_cast<int>(inT.size()), MemoryDirection::DDR2CMX});
                cost = Cycles::cost_adder(cost, in_ddr_dma);
            }
        }

        if (output_in_ddr) {
            // Add cost of spilling output activation from DDR to CMX
            for (auto& outT : layer.outputs) {
                auto out_ddr_dma =
                        compute_dma_cycles({layer.device, static_cast<int>(outT.size()), MemoryDirection::CMX2DDR});
                cost = Cycles::cost_adder(cost, out_ddr_dma);
            }
        }

        return cost;
    }

    /**
     * @brief Compute the optimal cost of a SHV kernel
     *
     * @param layer the SHV kernel
     * @param nSHV the number of SHV/tile
     * @param nTiles the number of CMX tiles
     * @param input_in_ddr enable/disable input in DDR (require extra DMA to fetch data in CMX)
     * @param output_in_ddr enable/disable output in DDR (require extra DMA to spill data in CMX)
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType Layer(const SHAVEWorkload& layer, const unsigned int nSHV = 1, const unsigned int nTiles = 1,
                              const bool input_in_ddr = false, const bool output_in_ddr = false) const {
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
                auto in_ddr_dma = compute_dma_cycles(
                        {layer.get_device(), static_cast<int>(inT.size()), MemoryDirection::DDR2CMX});
                cost = Cycles::cost_adder(cost, in_ddr_dma);
            }
        }

        if (output_in_ddr) {
            // Add cost of spilling output activation from DDR to CMX
            for (auto& outT : layer.get_outputs()) {
                auto out_ddr_dma = compute_dma_cycles(
                        {layer.get_device(), static_cast<int>(outT.size()), MemoryDirection::CMX2DDR});
                cost = Cycles::cost_adder(cost, out_ddr_dma);
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
    CyclesInterfaceType OneTileWeightsPrefetching(const DPULayer& layer) const {
        const auto& config = getDeviceConfiguratorForTiles(layer.device);

        const auto in_1_movable_size = layer.weight_footprint(config);

        const VPUTensor weight_plus_table_tensor{{static_cast<unsigned int>(in_1_movable_size), 1, 1, 1},
                                                 DataType::UINT8};  // weight_footprint is in bytes
        auto cost = compute_dma_cycles(
                {layer.device, static_cast<int>(weight_plus_table_tensor.size()), MemoryDirection::DDR2CMX});
        return cost;
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
        get_cost_model().avgpool_replace_by(wl);
        get_cost_model().compressConv_replace_by_CM_CONV_VPU27(wl);
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
        if (device >= VPUDevice::__size)
            return {};
        else
            return LayerPropertiesHolder::get_properties(device).getValidTilingStrategies();
    }
};

}  // namespace VPUNN

#endif  // VPUNN_LAYER_COST_MODEL_H
