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

#include <memory>
#include <optional>
#include <string>  // for std::string
#include <variant>
#include <vector>                        // for std::vector
#include "core/dma_map_type_selector.h"  // need this to instantiate the template with specifics like DMANNWorkload_NPU27
#include "core/serializer.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/layer.h"
#include "vpu/layer_split_info.h"
#include "vpu/types.h"
#include "vpu/vpu_tiling_strategy.h"
#include "vpu_cost_model.h"
#include "vpu_dma_cost_model_variant.h"
#include "vpu_layer_strategy.h"

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
            bool tryToLoadPairedCache = false              ///< see L1 DPU constructor details for this
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(filename, profile, cache_size, batch_size,
                                                                            dpu_cache_filename, shave_cache_filename,
                                                                            tryToLoadPairedCache)},
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
            size_t shave_cache_data_length = 0         /// < the size of the shave_cache_data buffer
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(
                      model_data, model_data_length, copy_model_data, profile, cache_size, batch_size, dpu_cache_data,
                      dpu_cache_data_length, shave_cache_data, shave_cache_data_length)},
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
            bool tryToLoadPairedCache = false              ///< see L1 DPU constructor details for this
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(filename, profile, cache_size, batch_size,
                                                                            dpu_cache_filename, shave_cache_filename,
                                                                            tryToLoadPairedCache)} {
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
            size_t shave_cache_data_length = 0       /// < the size of the shave_cache_data buffer
            )
            : ptr_internal_dpu_cost_provider{std::make_shared<VPUCostModel>(
                      model_data, model_data_length, copy_model_data, profile, cache_size, batch_size, dpu_cache_data,
                      dpu_cache_data_length, shave_cache_data, shave_cache_data_length)} {
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
    void initialize_serializers();

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
    CyclesInterfaceType Layer(DPULayer& layer, VPULayerStrategy strategy);

    CyclesInterfaceType Layer(DPULayer& layer, VPULayerStrategy strategy, LayerSplitInfo& detailed_split);

    // layer for dCIiM targeted layers, just rename the prev 2 methods. Alternative is to absorb the dCiM or SCL flag
    // inside of VPULayerStrategy

    CyclesInterfaceType Layer_dCiM(DPULayer& layer, VPULayerStrategy strategy);

    CyclesInterfaceType Layer_dCiM(DPULayer& layer, VPULayerStrategy strategy, LayerSplitInfo& detailed_split);

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
                              bool prefetching = true);
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
                              bool input_in_ddr, bool output_in_ddr, bool prefetching, LayerSplitInfo& detailed_split);

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
    );

    /// version without detailed split output parameter and no hash or tiling strategy.
    CyclesInterfaceType LayersPreSplit(const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU,
                                       bool input_in_ddr, bool output_in_ddr, bool prefetching);

    /// version without detailed split output parameter and no hash or tiling strategy.
    CyclesInterfaceType LayersPreSplit(const std::vector<SHAVEWorkload>& layers_pre_split, unsigned int nSHV,
                                       bool input_in_ddr, bool output_in_ddr);

protected:
    /**
     * @brief Check if the DMACostModelVariant holds a valid DMACostModel
     *
     * @return true if a DMACostModels priority list has nn dma initialized, false otherwise
     */
    bool is_dma_model_variant() const;
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
    CyclesInterfaceType compute_dma_cycles(const DMATransfer1D& dwl) const;

    // template <bool B = serialization_enabled, typename std::enable_if<B, int>::type = 0>
    static const std::vector<std::string> get_names_for_serializer();

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
                                     bool prefetching = true, LayerSplitInfo* detailed_split = nullptr) const;

    /// like Layer but with pre-split layers
    CyclesInterfaceType layer_pre_split_cycles(
            VPUCostModel& dpu_cost_provider,  // cost model to be used
            const std::vector<DPULayer>& layers_pre_split, unsigned int nDPU = 1, bool input_in_ddr = false,
            bool output_in_ddr = false, bool prefetching = true, LayerSplitInfo* detailed_split = nullptr,
            const size_t fullLayerHash = 0,  // hash on layer only, computed by VPUX
            const std::optional<VPUTilingStrategy> strategyOfSplit =
                    (std::optional<VPUTilingStrategy>())  // to be sent only for MC pass
    ) const;

    CyclesInterfaceType layer_pre_split_cycles(VPUCostModel& shave_cost_provider,  // cost model to be used
                                               const std::vector<SHAVEWorkload>& layers_pre_split,
                                               unsigned int nSHV = 1, bool input_in_ddr = false,
                                               bool output_in_ddr = false) const;

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
     * @return measured best cycles or error code . \\see Cycles for error codes
     */
    CyclesInterfaceType Layer(DPULayer& layer, unsigned int nDPU = 1, unsigned int nTiles = 1,
                              bool input_in_ddr = false, bool output_in_ddr = false, bool prefetching = true);

    // Shave Operations area is next

    /**
     * @brief Compute the optimal cost of a SHAVEWorkload
     *
     * @param layer the SHV kernel
     * @param strategy the layer strategy
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType Layer(const SHAVEWorkload& layer, const VPULayerStrategy& strategy) const;

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
                              const bool input_in_ddr = false, const bool output_in_ddr = false) const;

protected:
    const IDeviceValidValues& getDeviceConfiguratorForTiles(VPUDevice device) const;
    /**
     * @brief The cycles it takes to prefetch the weights of one tile, not considering any pipelining
     * Takes in consideration sparsity (enabled and value) for this transfer
     *
     * @param layer the DPULayer, expected to be a layer allocated to this tile
     * @return CyclesInterfaceType
     */
    CyclesInterfaceType OneTileWeightsPrefetching(const DPULayer& layer) const;

protected:
    /// @brief gets the overall DPU cycles considering also error cases
    /// If at least one time is an error (one tile is bad) the result will be an error with code
    /// Cycles::ERROR_TILE_OUTPUT
    ///
    /// @param tiles_cost is the container with time values CyclesInterfaceType
    /// @returns the largest if all good, and error if not all are good ()
    CyclesInterfaceType extractLargestTime(const std::vector<CyclesInterfaceType>& tiles_cost) const;

    void operation_sanitisation(DPULayer& wl) const;

    // static members
public:
    /**
     * @brief Get the valid tiling strategy for a device
     *
     * @param device the VPUDevice
     * @return std::vector<VPUTilingStrategy>
     */
    static std::vector<VPUTilingStrategy> getValidTilingStrategies(const VPUDevice& device);
};

}  // namespace VPUNN

#endif  // VPUNN_LAYER_COST_MODEL_H
