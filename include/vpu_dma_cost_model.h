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

#include <memory>
#include <string>
#include <tuple>

#include "core/cache.h"
#include "core/dma_map_type_selector.h"  // need this to instantiate the template with specifics like DMANNWorkload_NPU27 and other...
#include "core/serializer.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"
#include "vpu/dma_cost_providers/dmann_cost_provider.h"
#include "vpu/dma_cost_providers/priority_dma_cost_provider.h"
#include "vpu/dma_workload.h"
#include "vpu/types.h"
#include "vpu/validation/sanity_report.h"  // For SanityReport
#include "vpu/http_cost_provider_intf.h"
#include "vpu/http_cost_provider_factory.h"

namespace VPUNN {

/**
 * @brief
 *Has to be factored to own file or to be redesigned, why we need this, what's the purpose
 * We want from the VPUX to use same interface or same descriptor as we use in DMANN part
 * Also the theoretical DMA should have a common interface(datatype dma workload) with the DMANN
 */
class VPUNN_API DMATheoreticalCostModel {
private:
    const DMATheoreticalCostProvider dma_theoretical{};

public:
    explicit DMATheoreticalCostModel();

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
                     MemoryLocation output_location = MemoryLocation::CMX, unsigned int output_write_tiles = 1) const;

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(const DMAWorkload& wl) const;
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
    std::shared_ptr<IDMACostProvider<DMADesc>>
            ptr_internal_dma_cost_provider;  ///< shared ownership of DMA cost provider
                                             ///< bundle with priority fallback mechanism

    IDMACostProvider<DMADesc>& dma_cost_provider{
            *ptr_internal_dma_cost_provider};  ///< provides cycles through priority-based provider selection

    mutable LRUCache<DMADesc, float> cache;  ///< all devices cache/LUT for DMA ops
                                             ///< this is a preloaded cache that features also a dynamic one

    const std::unique_ptr<IHttpCostProvider> http_dma_cost_provider;  ///< HTTP cost provider for DMA
    mutable CSVSerializer interogation_serializer;  ///< serializes DMADesc workloads to csv file.
public:
    /**
     * @brief Construct a new DMACostModel object
     *
     * @param filename the name of the .vpunn model
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     * @param cache_filename filename for cache persistence
     */
    explicit DMACostModel(const std::string& filename = "", bool profile = false, const unsigned int cache_size = 16384,
                          const unsigned int batch_size = 1, const std::string& cache_filename = "");

    /**
     * @brief Construct a new DMACostModel object
     *
     * @param model_data a buffer containing a .vpunn model
     * @param model_data_length the size of the model_data buffer
     * @param copy_model_data enable/disable the memcopy of the buffer
     * @param profile enable/disable profiling
     * @param cache_size the size of the LRUCache
     * @param batch_size model batch size
     * @param cache_data buffer containing cache data
     * @param cache_data_length size of cache data buffer
     */
    explicit DMACostModel(const char* model_data, size_t model_data_length, bool copy_model_data, bool profile = false,
                          const unsigned int cache_size = 16384, const unsigned int batch_size = 1,
                          const char* cache_data = nullptr, size_t cache_data_length = 0);

    virtual ~DMACostModel() = default;

public:
    /**
     * @brief Check if the internal VPUNN is initialized
     *
     * @return true the VPUNN network is initialized
     * @return false the VPUNN network is not initialized
     */
    bool nn_initialized() const;

    /**
     * @brief Get access to the preloaded cache counter for statistics
     *
     * @return const AccessCounter& Reference to the cache counter
     */
    const AccessCounter& getPreloadedCacheCounter() const;

protected:
    ///@ brief checks some validity criteria and performs sanitization that does not alter relevance
    ///
    /// @sa DPU_OperationSanitizer::check_and_sanitize for details
    /// from legacy behavior is ensures that input channels are equal to output channels for channel preserving
    /// operations
    /// @param workload [in, out] to be checked and changed
    /// @param result [out] holds error code
    /// @returns true if checks were OK, false if this wl is not to be used
    bool sanitize_workload(DMADesc&, SanityReport& result) const;

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
    CyclesInterfaceType computeCycles(const DMADesc& wl);

    /// @brief same like  @see computeCycles(DMANNWorkload wl) , the extra param is to have as output the textual
    /// errors/findings discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param li will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    std::tuple<CyclesInterfaceType, std::string> computeCyclesMsg(DMADesc wl);

    /// @brief same like  @see computeCycles(DMANNWorkload wl) , the extra param is to have as output the textual
    /// errors/findings discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param info [out] will collect error info regarding wl checking.
    CyclesInterfaceType computeCycles(const DMADesc& wl, std::string& info);

private:
    /* @brief Execution
     */
    CyclesInterfaceType Execute_and_sanitize(const DMADesc& wl, std::string& info);

    /**
     * @brief Wrapper over run_cost_providers for handling situations where a workload cannot be resolved with only one
     * inference.
     *
     * @param workload is the workload to be inferred
     * @return the runtime or error
     */
    CyclesInterfaceType get_cost(const DMADesc& workload, std::string& info, std::string* cost_source = nullptr) const;

    /**
     * @brief Run the cost providers for a given workload.
     *
     * This function retrieves the cost from the cache or from the priority-based cost provider chain.
     * The cache is checked first for performance. If not found, the priority provider is used (NN -> theoretical).
     *
     * @param workload The DMA workload to be processed.
     * @param info A string to store additional information about the cost source.
     * @param cost_source A string to store the source of the cost.
     * @return The number of cycles required for the workload.
     */
    CyclesInterfaceType run_cost_providers(const DMADesc& workload, std::string& info,
                                           std::string* cost_source = nullptr) const;
};

}  // namespace VPUNN

#endif  // VPU_COST_MODEL_H
