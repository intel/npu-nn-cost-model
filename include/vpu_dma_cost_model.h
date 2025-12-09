// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPU_DMA_COST_MODEL_H
#define VPU_DMA_COST_MODEL_H

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/cache.h"
#include "core/logger.h"

#include "inference/dma_preprocessing.h"
#include "inference/dma_preprop_factory.h"
#include "inference/post_process.h"

#include "vpu/dma_types.h"
#include "vpu/dma_workload.h"
#include "vpu/performance.h"
#include "vpu/power.h"
#include "vpu/types.h"

#include "inference/dma_post_process.h"
#include "inference/dma_postprocessing_factory.h"
#include "inference/vpunn_runtime.h"
#include "vpu/utils.h"
#include "vpu/validation/checker_utils.h"

#include "vpu/cycles_interface_types.h"
#include "vpu/validation/dpu_operations_sanitizer.h"

#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"

#include "core/serializer.h"
#include "vpu/vpu_mutex.h"

#include "vpu/dma_theoretical_cost_provider.h"
#include "vpu/serialization/dma_cost_serialization_wrapper.h"

#include "vpu/dmann_cost_provider.h"

#include <typeinfo>

namespace VPUNN {

/**
 * @brief
 *Has to be factored to own file or to be redesigned, why we need this, what's the purpose
 * WE want from the VPUX to use same interface or same descriptor as we use in DMANN part
 * ALso the theoretical DMA should have a common interface(datatype dma workload) with the DMANN
 */
class VPUNN_API DMATheoreticalCostModel {
private:
    const DMATheoreticalCostProvider dma_theoretical{};

public:
    explicit DMATheoreticalCostModel() {
        Logger::initialize();
    }

protected:
    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param device DMA VPUDevice
     * @param input DMA input Tensor
     * @param output DMA output Tensor
     * @param input_location where is the source memory
     * @param output_location where is the destination memory
     * @param output_write_tiles how many CMX tiles the DMA broadcast
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(VPUDevice device, const VPUTensor& input, const VPUTensor& output,
                     MemoryLocation input_location = MemoryLocation::DRAM,
                     MemoryLocation output_location = MemoryLocation::CMX, unsigned int output_write_tiles = 1) const {
        // Call the helper function
        return dma_theoretical.DMATheoreticalCycles(
                {device, input, output, input_location, output_location, output_write_tiles});
    }

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(const DMAWorkload& wl) const {
        // Call the helper function
        return dma_theoretical.DMATheoreticalCycles(wl);
    }
};

/**
 * @brief The DMACostModel class
 *
 * Has behind a loaded DMACostModel neural network that infers cycle times for DMA
 * /tparam DMADesc  the actual type of DMADescriptor that will be used
 *
 */
template <class DMADesc>
class VPUNN_API DMACostModel {
protected:
public:
    using DescType = DMADesc;  ///< Useful for deducing the type of the descriptor

private:
    const DMANNCostProvider<DMADesc> nn_cost_provider; ///< NN cost provider for DMA

    // DMA cost providers
    // No NN or measured DMA cost provider available
    const DMATheoreticalCostProvider dma_theoretical{};  ///< theoretical cost provider for DMA
    
    mutable CSVSerializer interogation_serializer;  ///< serializes DMADesc workloads to csv file.
public:
    /**
     * @brief Construct a new VPUCostModel object
     *
     * @param filename the name of the .vpunn model
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit DMACostModel(const std::string& filename = "", bool profile = false, const unsigned int cache_size = 16384,
                          const unsigned int batch_size = 1, const std::string& cache_filename = "")
            : nn_cost_provider(filename, batch_size, profile, cache_size, cache_filename) {
        Logger::initialize();

        if (!nn_cost_provider.is_initialized()) {
            return;
        }
        // is_profiling_service_enabled = init_profiling_service();
        interogation_serializer.initialize(DescType::get_wl_name(), FileMode::READ_WRITE, nn_cost_provider.get_names_for_serializer());
    }
    // VPUCostModel(const VPUCostModel&) = delete;
    // VPUCostModel(VPUCostModel&&) = default;

    /**
     * @brief Construct a new VPUCostModel object
     *
     * @param model_data a buffer containing a .vpunn model
     * @param model_data_length the size of the model_data buffer
     * @param copy_model_data enable/disable the memcopy of the buffer
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     */
    explicit DMACostModel(const char* model_data, size_t model_data_length, bool copy_model_data, bool profile = false,
                          const unsigned int cache_size = 16384, const unsigned int batch_size = 1,
                          const char* cache_data = nullptr, size_t cache_data_length = 0)
            : nn_cost_provider(model_data, model_data_length, copy_model_data, profile, cache_size,
                               batch_size, cache_data, cache_data_length) {
        Logger::initialize();

        if (!nn_cost_provider.is_initialized()) {
            return;
        }
        // is_profiling_service_enabled = init_profiling_service();
        interogation_serializer.initialize(DescType::get_wl_name(), FileMode::READ_WRITE, nn_cost_provider.get_names_for_serializer());
    }

    /**
     * @brief Check if the internal VPUNN is initialized
     *
     * @return true the VPUNN neural network is initialized
     * @return false the VPUNN neural network is not initialized
     */
    bool nn_initialized() const {
        return nn_cost_provider.is_initialized();
    }

    /// @brief provides the value interval where the NN raw outputs are considered valid and will be used to further
    /// compute information
    ///
    /// @returns a pair containing (minimum_valid_value maximum_valid_value)
    std::pair<float, float> get_NN_Valid_interval() const noexcept {
        return nn_cost_provider.get_NN_Valid_interval();
    }

protected:
    ///@ brief checks some validity criteria and performs sanitization that does not alter relevance
    ///
    /// @sa DPU_OperationSanitizer::check_and_sanitize for details
    /// from legacy behavior is ensures that input channels are equal to output channels for channel preserving
    /// operations
    /// @param workload [in, out] to be checked and changed
    /// @param result [out] holds error code
    /// @returns true if checks were OK, false if this wl is not to be used
    bool sanitize_workload(DMADesc&, SanityReport& result) const {
        // sanitizer.check_and_sanitize(workload, result);
        return result.is_usable();
    }

public:
    /**
     * @brief Return the number of cycles needed to compute a workload
     *
     * Important: If no NN is available it will return Cycles::ERROR_INFERENCE_NOT_POSSIBLE. Check if NN is loaded with
     * nn_initialized()
     *
     * A sanity check will be performed: NO
     *
     * List of Error codes is available in CyclesInterfaceType doc.
     *
     * A sanity check will be performed also on the NN output, in case the NN  raw value is not reliable it will not be
     * returned but an error code will be given, e.g. ERROR_INVALID_OUTPUT_RANGE
     *
     * To see the limits of valid NN values interval , use @sa get_NN_Valid_interval().  Zero is a value that will NOT
     * be filtered out.
     *
     * Behind the DPU computation is a trained Neural Network that might give unexpected results in case is asked about
     * a workload that is odd/(not well formed) or was not trained in that area or workloads.
     * The workload passed as parameter for inference should be a valid one, a one that makes sense, we are checking
     * some sanity, but ,for now, not a strict/extensive sanity check is performed. A workload with unrealistic
     * combinations of  parameters  will not be detected.
     *
     * In case the wl configuration is unrealistic the network will give undefined(aberrant) results (it was not trained
     * on invalid data). The NN raw output is filtered for  generic valid interval  but the user can also be aware of
     * this behavior and use its own narrower ranges
     *
     * e.g.  Depending on the wl a cycle values of 10 might be unrealistic, also a value of 100milion cycles (@1Ghz is
     * ~100ms),  The user should be aware that not all aberrant/unrealistic NN outputs are handled inside.
     *
     *
     * @param wl a workload to be evaluated.
     * @return unsigned int workload execution cycles or an error code.
     *
     * @throws out_of_range : cache problems, cannot pre-process data , generate the NN descriptor due to data unknown
     * @throws runtime_error: cannot generate the NN descriptor, e.g expected sizes do not match
     *
     */
    CyclesInterfaceType computeCycles(const DMADesc& wl) {
        std::string dummy_info{};
        return computeCycles(wl, dummy_info);
    }

    /// @brief same like  @see computeCycles(DMANNWorkload wl) , the extra param is to have as output the textual
    /// errors/findings discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param li will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    std::tuple<CyclesInterfaceType, std::string> computeCyclesMsg(DMADesc wl) {
        std::string dummy_info{};
        auto previous_print_mode{Checker::set_print_tags(false)};
        const auto r{computeCycles(wl, dummy_info)};
        Checker::set_print_tags(previous_print_mode);
        return std::make_tuple(r, dummy_info);
    }

    /// @brief same like  @see computeCycles(DMANNWorkload wl) , the extra param is to have as output the textual
    /// errors/findings discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param info [out] will collect error info regarding wl checking.
    CyclesInterfaceType computeCycles(const DMADesc& wl, std::string& info) {
        return Execute_and_sanitize(wl, info);
    }

    // just for debug purposes
    /* coverity[pass_by_value] */
    std::tuple<float, std::string> computeBandwidthMsg(DMADesc wl) {
        return nn_cost_provider.computeBandwidthMsg(std::move(wl));
    }

private:
    /* @brief Execution
     */
    CyclesInterfaceType Execute_and_sanitize(const DMADesc& wl, std::string& info) {
        DMACostSerializationWrap<DMADesc> serialization_handler(interogation_serializer);
        serialization_handler.serializeDMAWorkload(wl);

        // sanitize and check the input.
        SanityReport problems{};
        //const auto is_inference_relevant = sanitize_workload(wl, problems);
        info = problems.info;

        std::string cost_source = "unknown";
        CyclesInterfaceType cycles{problems.value()};  // neutral value or reported problems at sanitization
        
        //if (is_inference_relevant){
        cycles = get_cost(wl, info, &cost_source);
        //}

        serialization_handler.serializeCyclesAndCostInfo_closeLine(cycles, std::move(cost_source), info);

        return cycles;
    }

    /**
     * @brief Wrapper over run_cost_providers for handling situations where a workload cannot be resolved with only one
     * inference. 
     *
     * @param workload is the workload to be inferred
     * @return the runtime or error
     */
    CyclesInterfaceType get_cost(const DMADesc& workload, std::string& info, std::string* cost_source = nullptr) const {
        // @todo : impact on energy?, CHeck if energy/DPUINfo/Theoretical cycles/ops considers this situation to reduce
        // energy

        return run_cost_providers(workload, info, cost_source);  // normal wl handling
    }

    /**
     * @brief Run the cost providers for a given workload.
     *
     * This function checks if the DPU NN cost provider is initialized and retrieves the cost from the cache or
     * profiling service. If the profiling service is not available, it falls back to the DPU NN cost provider or
     * theoretical cycles.
     *
     * @param workload The DPU workload to be processed.
     * @param info A string to store additional information about the cost source.
     * @param cost_source A string to store the source of the cost.
     * @return The number of cycles required for the workload.
     */
    CyclesInterfaceType run_cost_providers(const DMADesc& workload, std::string& info,
                                           std::string* cost_source = nullptr) const {
        auto cycles = Cycles::NO_ERROR;
        const auto is_inference_posible = nn_cost_provider.is_initialized();

        // First look into nn_cost_provider cache.
        // This has to be redesigned for the fixed cache to be a standalone cost provider.
        const auto cached_cost{nn_cost_provider.get_cached(workload, cost_source)};
        if (!Cycles::isErrorCode(cached_cost)) {
            cycles = cached_cost;
        } else {
            // for now let the info be empty
            info = "";
            // Will enable on a further step where we have a profiling service available
//             if (is_profiling_service_enabled) {
//                 auto dpu_op = DPUOperation(workload, sanitizer.getDeviceConfiguration(workload.device));
//                 if (cost_source) *cost_source = "profiling_service_" + profiling_backend;
// #ifdef VPUNN_BUILD_HTTP_CLIENT
//                 cycles = http_dpu_cost_provider->getCost(dpu_op, info, profiling_backend);
// #else
//                 info = "";  // Avoid unreferenced var warning
// #endif
//             } else {
//                 cycles = Cycles::ERROR_PROFILING_SERVICE;
//             }

            if (Cycles::isErrorCode(cycles) || cycles == Cycles::NO_ERROR) {
                // if the profiling service is not available, we will use the NN cost provider
                // if the NN is not available, we will use the theoretical cycles
                if (is_inference_posible) {
                    if (cost_source) *cost_source = "nn_" + nn_cost_provider.get_model_nickname();
                    cycles = nn_cost_provider.get_cost(workload);
                } else {
                    if (cost_source) *cost_source = "theoretical";
                    cycles = dma_theoretical.DMATheoreticalCycles(DMAWorkloadTransformer::create_workload<DMADesc>(workload));
                }
            }

            // Concerns mainly cycles returned from providers other than NNCostProvider
            // We need to share NNCostProvider's cache in order to have access to the fixed cache that's stored as part
            // of the dynamic cache (LRUCache) (will be separated in near future).
            if (Cycles::isErrorCode(nn_cost_provider.get_cached(workload)) && !Cycles::isErrorCode(cycles)) {
                nn_cost_provider.add_to_cache(workload, static_cast<float>(cycles));
            }
        }

        return cycles;
    }

};

}  // namespace VPUNN

#endif  // VPU_COST_MODEL_H
