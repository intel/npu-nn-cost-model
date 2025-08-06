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
#include "vpu/vpu_performance_model.h"

#include "vpu/cycles_interface_types.h"
#include "vpu/validation/dpu_operations_sanitizer.h"

#include "vpu/shave/shave_collection.h"
#include "vpu/shave/shave_devices.h"

#include "core/serializer.h"
#include "vpu/vpu_mutex.h"

#include <typeinfo>

namespace VPUNN {

/**
 * @brief The DMACostModel class
 *
 * Has behind a loaded DMACostModel neural network that infers cycle times for DMA
 *
 */
class VPUNN_API DMATheoreticalCostModel /*: protected VPUNNPerformanceModel */ {
private:
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
        VPUNNPerformanceModel pm;
        return pm.DMATheoreticalCycles({device, input, output, input_location, output_location, output_write_tiles});
    }

    /**
     * @brief Return the number of cycles needed to compute a DMA transfer
     *
     * @param wl a DMAWorkload
     * @return unsigned int the number of cycles of the DMA transfer
     */
    unsigned int DMA(const DMAWorkload& wl) const {
        // Call the helper function
        VPUNNPerformanceModel pm;
        return pm.DMATheoreticalCycles(wl);
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
class VPUNN_API DMACostModel :
        virtual protected VPU_MutexAcces  // for mutex access
{
protected:
public:
    using DescType = DMADesc;  ///< Useful for deducing the type of the descriptor

private:
    const DMARuntimeProcessingFactory<DMADesc> preprocessing_factory;  ///< provides Preprocessing objects
    const DMAPostProcessingFactory<DMADesc> postprocessing_factory;
    const Runtime vpunn_runtime;                 ///< the loaded inference model is here, used for FW propagation
    InferenceExecutionData runtime_buffer_data;  ///< the memory/buffers used for executing a model (in/out and inter
                                                 ///< layer buffers). It is paired with the model at creation.

    IPreprocessingDMA<float, DMADesc>&
            preprocessing;  ///< prepares the input vector for the runtime, configured at ctor
    const DMAPostProcessSupport results_config;
    const IPostProcessDMA<DMADesc>& post_processing;
    LRUCache<std::vector<float>, float>
            cache;  ///< cache for inferred values, used only for single workload, not for batches, raw NN out values
    mutable CSVSerializer interogation_serializer;  ///< serializes DMADesc workloads to csv file.
    mutable CSVSerializer cache_miss_serializer;    ///< serializer for missed cache DMADesc workloads

    /// @brief obtains the actual preprocessing instance from factory. The factory must live longer than the instance
    /// created. warning: Throws if not possible
    static IPreprocessingDMA<float, DMADesc>& init_preproc(const DMARuntimeProcessingFactory<DMADesc>& factory,
                                                           const ModelVersion& version_service,
                                                           const std::string& filename, const unsigned int batch_size) {
        // let's initialize the preproc aspects based on input version
        const int input_version = version_service.get_input_interface_version();
        //  checking if either we have some preprocess or we have a unsupported version
        if (factory.exists_preprocessing(input_version)) {
            auto& preproc = factory.make_preprocessing(input_version);
            preproc.set_probable_batch(batch_size);
            return preproc;
        } else {
            std::stringstream buffer;
            buffer << "Cannot create DMA preprocessing (DMA NN descriptor generator ) stage!.Preprocessing (NN model) "
                      "with "
                      "version ("
                   << input_version
                   << ") is not known/supported by requested DMADescriptor template param: " << typeid(DMADesc).name()
                   << "\nFilename: " << filename
                   << " , DMANN file Version info (raw): " << version_service.get_raw_name();
            std::string details = buffer.str();
            Logger::error() << details;

            throw std::runtime_error(details);
        }
    }

    static const IPostProcessDMA<DMADesc>& init_postproc(const DMAPostProcessingFactory<DMADesc>& factory,
                                                         const ModelVersion& version_service,
                                                         std::string_view filename) {
        const int version = version_service.get_output_interface_version();
        //  checking if either we have some process or we have a unsupported version
        if (factory.exists(version)) {
            auto& proc = factory.make(version);
            return proc;
        } else {
            std::stringstream buffer;
            buffer << "Cannot create DMA post processing stage!.Post processing with version (" << version
                   << ") is not known/supported by requested DMADescriptor template param: " << typeid(DMADesc).name()
                   << "\nFilename: " << filename
                   << " , DMANN file Version info (raw): " << version_service.get_raw_name();
            std::string details = buffer.str();
            Logger::error() << details;

            throw std::runtime_error(details);
        }
    }

    static const std::vector<std::string> get_names_for_serializer() {
        std::vector<std::string> names;

        for (const auto& name : DescType::_get_member_names()) {
            names.push_back(name);
        }

        names.push_back("cycles");
        names.push_back("info");

        return names;
    }

    /**
     * @brief check post config will check if we either have an empty model, or the output version is supported
     *
     * If we have an empty ideal model or we got the output version supported this function will be exited with
     * no errors. In case that we got a model with an unsupported output version this function will throw a runtime
     * error specifying the output version and the full raw name of it.
     *
     * @throws std::runtime_error In case that we got a model with an unsupported output version you will get a runtime
     * error
     *
     * @param v is the model version we took info about the full_raw_name in case we have an empty model
     */
    void check_post_config(const ModelVersion& v) {
        const auto raw_name_intf = v.get_raw_name();

        // in case we have an empty ideal model the raw_name is defaulted to none and we should contiune the run
        if (raw_name_intf == "none") {
            return;
        }

        if (!results_config.is_output_supported()) {
            std::stringstream buffer;
            buffer << "Cannot load/handle Models output version. The output version: ("
                   << v.get_output_interface_version()
                   << ") is not known/supported. Version info (raw):" << v.get_raw_name();
            std::string details = buffer.str();
            Logger::error() << details;

            throw std::runtime_error(details);
        }
    }

    /// @brief  check and try to make the preprocessing output to be the same as what model expects
    /// This mechanism is unsafe at this moment and lets you change the preprocessing output to a bigger or smaller size
    /// The result may be impossibility to run (if smaller) or leaving empty zeros at the end (only partial fill)
    ///
    void correlate_preprocessor_with_model_inputs() {
        const auto model_input_size{(runtime_buffer_data.input_shapes()[0])[1]};
        const auto preprocessing_output_size = preprocessing.output_size();
        if (model_input_size != preprocessing_output_size) {
            Logger::warning() << "Changing preprocessing DMA output size (" << preprocessing_output_size
                              << ") to the model input size (" << model_input_size << ")";
            preprocessing.set_size(model_input_size);
        }
    }

public:
    /// @brief provides the value interval where the NN raw outputs are considered valid and will be used to further
    /// compute information
    ///
    /// @returns a pair containing (minimum_valid_value maximum_valid_value)
    std::pair<float, float> get_NN_Valid_interval() const noexcept {
        return post_processing.get_NN_Valid_interval();
    }

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
            : vpunn_runtime(filename, profile),
              runtime_buffer_data(vpunn_runtime.createNewInferenceExecutionData(batch_size)),
              preprocessing(
                      init_preproc(preprocessing_factory, vpunn_runtime.model_version_info(), filename, batch_size)),
              results_config(vpunn_runtime.model_version_info().get_output_interface_version()),
              post_processing(init_postproc(postprocessing_factory, vpunn_runtime.model_version_info(), "")),
              cache(cache_size, /* preprocessing.output_size(),*/ cache_filename) {
        Logger::initialize();

        interogation_serializer.initialize(DescType::get_wl_name(), FileMode::READ_WRITE, get_names_for_serializer());
        cache_miss_serializer.initialize("cache_misses_" + DescType::get_wl_name(), FileMode::READ_WRITE,
                                         get_names_for_serializer());
        check_post_config(vpunn_runtime.model_version_info());

        if (!vpunn_runtime.initialized()) {
            return;
        }

        correlate_preprocessor_with_model_inputs();
        // preprocessing.set_probable_batch(batch_size);
        //  workloads_results_buffer.reserve(prealloc_results);
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
            : vpunn_runtime(model_data, model_data_length, copy_model_data, profile),
              runtime_buffer_data(vpunn_runtime.createNewInferenceExecutionData(batch_size)),
              preprocessing(init_preproc(preprocessing_factory, vpunn_runtime.model_version_info(), "ConstCharInit",
                                         batch_size)),
              results_config(vpunn_runtime.model_version_info().get_output_interface_version()),
              post_processing(init_postproc(postprocessing_factory, vpunn_runtime.model_version_info(), "")),
              cache(cache_size, /* preprocessing.output_size(), */ cache_data, cache_data_length) {
        Logger::initialize();

        interogation_serializer.initialize(DescType::get_wl_name(), FileMode::READ_WRITE, get_names_for_serializer());
        cache_miss_serializer.initialize("cache_misses_" + DescType::get_wl_name(), FileMode::READ_WRITE,
                                         get_names_for_serializer());

        check_post_config(vpunn_runtime.model_version_info());

        if (!vpunn_runtime.initialized()) {
            return;
        }
        correlate_preprocessor_with_model_inputs();
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
     * @brief Compute the NN Output of a specific workload
     * NO Threadsafe protection planned here. DO NOT USE directly!
     * takes in consideration the cache
     * no sanitation is done
     * no check if network exists
     *
     * @param workload a workload
     * @return float the NN raw output, not filtered
     */
    float run_NN(const DMADesc& workload) {
        const auto& vector = preprocessing.transformSingle(workload);
        // Check for cache hit
        const auto cached_value = cache.get(vector);
        if (!cached_value) {
            // run the model in case of a cache miss
            const auto infered_value = vpunn_runtime.predict<float>(vector, runtime_buffer_data)[0];
            // Add result to the cache
            cache.add(vector, infered_value);  // raw

            if (cache_miss_serializer.is_serialization_enabled()) {  // has to be factored out
                // do we need to post process?
                try {
                    cache_miss_serializer.serialize(SerializableField<VPUDevice>{"device", workload.device});
                    serialize_workload(workload, cache_miss_serializer);
                    cache_miss_serializer.end();
                } catch (const std::exception& e) {
                    Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                    cache_miss_serializer.clean_buffers();
                }
                cache_miss_serializer.clean_buffers();
            }

            return infered_value;  // raw
        }
        return *cached_value;  // raw
    }

public:
    /// @brief exposed only for debug purposes, populates the NN descriptor
    /// Does not run the model, or checks the validity of the workload.
    ///
    /// The method is not const because the preprocessing fills an internal buffer with the descriptor
    ///
    /// @returns the descriptor of the workload for the scenario that this workload reaches the transform stage.
    std::vector<float> getDmaDescriptor(DMADesc wl) {
        return preprocessing.transformSingle(wl);
    }

    /// @brief provides the input and output versions of the loaded NN (debug purposes)
    std::tuple<int, int> getNNVersion() const {
        const auto& version{vpunn_runtime.model_version_info()};
        return std::make_tuple(version.get_input_interface_version(), version.get_output_interface_version());
    }

    /**
     * @brief Check if the internal VPUNN is initialized
     *
     * @return true the VPUNN neural network is initialized
     * @return false the VPUNN neural network is not initialized
     */
    bool nn_initialized() const {
        return vpunn_runtime.initialized();
    }

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
        std::lock_guard<std::recursive_mutex> lock(L1_mutex);
        std::string dummy_info{};
        return computeCycles(wl, dummy_info);
    }

    /// @brief same like  @see computeCycles(DMANNWorkload wl) , the extra param is to have as output the textual
    /// errors/findings discovered when handling the workload
    /// @param wl the workload to infer on
    /// @param li will collect error info regarding wl checking.
    /* coverity[pass_by_value] */
    std::tuple<CyclesInterfaceType, std::string> computeCyclesMsg(DMADesc wl) {
        std::lock_guard<std::recursive_mutex> lock(L1_mutex);
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
        std::lock_guard<std::recursive_mutex> lock(L1_mutex);
        return Execute_and_sanitize(wl, info);
    }

    // just for debug purposes
    std::tuple<float, std::string> computeBandwidthMsg(DMADesc wl) {
        std::string dummy_info{};
        auto previous_print_mode{Checker::set_print_tags(false)};

        float nn_size_div_cycle{0.0f};
        {
            const auto is_inference_posible = nn_initialized();
            if (is_inference_posible) {
                nn_size_div_cycle = run_NN(wl);
            } else {
                nn_size_div_cycle = -1.0f;
                std::stringstream buffer;
                buffer << "\nThe NN is not initialized. The inference is not possible"
                       << ".  Exiting with : " << nn_size_div_cycle << "\n";
                std::string details = buffer.str();
                dummy_info = dummy_info + details;
                Logger::error() << details;
            }
        }

        Checker::set_print_tags(previous_print_mode);
        return std::make_tuple(nn_size_div_cycle, dummy_info);
    }

private:
    /* @brief Execution
     */
    CyclesInterfaceType Execute_and_sanitize(const DMADesc& wl, std::string& info) {
        // sanitize and check the input.
        const auto is_inference_posible = nn_initialized();
        SanityReport problems{};
        info = problems.info;

        if (interogation_serializer.is_serialization_enabled()) {  // has to be factored out
            try {
                interogation_serializer.serialize(SerializableField<VPUDevice>{"device", wl.device});

                serialize_workload(wl, interogation_serializer);

            } catch (const std::exception& e) {
                Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                interogation_serializer.clean_buffers();
            }
        }

        CyclesInterfaceType cycles{problems.value()};  // neutral value or reported problems at sanitization
        if (is_inference_posible) {
            const auto raw_nn_output = run_NN(wl);
            if (post_processing.is_NN_value_invalid(raw_nn_output)) {
                cycles = Cycles::ERROR_INVALID_OUTPUT_RANGE;
                // std::cout << "\n Problematic inf response: "<<nn_output_cycles<<std::endl<<wl<<"\n";
                {
                    std::stringstream buffer;
                    buffer << "The NN returned a value outside of accepted ranges and is considered invalid. "
                              "NN_returned value :  "
                           << raw_nn_output << ".  Exiting with : " << Cycles::toErrorText(cycles)
                           << "\nWorkload DMA: " << wl << "\n";
                    std::string details = buffer.str();
                    info = info + details;
                    Logger::error() << details;
                }
            } else {
                cycles = post_processing.process(raw_nn_output, wl, info);
            }
        } else {  // NN not available, use theoretical cycles?
            cycles = Cycles::ERROR_INFERENCE_NOT_POSSIBLE;
            {
                std::stringstream buffer;
                buffer << "\nThe DMA NN is not initialized. The inference is not possible"
                       << ".  Exiting with : " << Cycles::toErrorText(cycles) << "\n";
                std::string details = buffer.str();
                info = info + details;
                Logger::error() << details;
            }
        }

        if (interogation_serializer.is_serialization_enabled()) {  // has to be factored out
            try {
                if (!interogation_serializer.is_write_buffer_clean()) {
                    interogation_serializer.serialize(SerializableField<decltype(cycles)>{"cycles", cycles});
                    interogation_serializer.end();
                }
            } catch (const std::exception& e) {
                Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                interogation_serializer.clean_buffers();
            }
            interogation_serializer.clean_buffers();
        }

        return cycles;
    }

    void serialize_workload(const DMANNWorkload_NPU27& wl, CSVSerializer& serializer) {
        if (serializer.is_serialization_enabled()) {
            try {
                serializer.serialize(SerializableField{"num_planes", wl.num_planes});
                serializer.serialize(SerializableField{"length", wl.length});
                serializer.serialize(SerializableField{"src_width", wl.src_width});
                serializer.serialize(SerializableField{"dst_width", wl.dst_width});
                serializer.serialize(SerializableField{"src_stride", wl.src_stride});
                serializer.serialize(SerializableField{"dst_stride", wl.dst_stride});
                serializer.serialize(SerializableField{"src_plane_stride", wl.src_plane_stride});
                serializer.serialize(SerializableField{"dst_plane_stride", wl.dst_plane_stride});
                serializer.serialize(SerializableField{"transfer_direction", wl.transfer_direction});
            } catch (const std::exception& e) {
                Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                serializer.clean_buffers();
            }
        }
    }

    void serialize_workload(const DMANNWorkload_NPU40_RESERVED& wl, CSVSerializer& serializer) {
        if (serializer.is_serialization_enabled()) {
            try {
                serializer.serialize(SerializableField{"src_width", wl.src_width});
                serializer.serialize(SerializableField{"dst_width", wl.dst_width});
                serializer.serialize(SerializableField{"num_dim", wl.num_dim});

                for (int i = 0; i < wl.num_dim; i++) {
                    const auto& dim{wl.e_dim[i]};
                    serializer.serialize(SerializableField{"src_stride_" + std::to_string(i + 1), dim.src_stride});
                    serializer.serialize(SerializableField{"dst_stride_" + std::to_string(i + 1), dim.dst_stride});
                    serializer.serialize(SerializableField{"src_dim_size_" + std::to_string(i + 1), dim.src_dim_size});
                    serializer.serialize(SerializableField{"dst_dim_size_" + std::to_string(i + 1), dim.dst_dim_size});
                }

                for (int i = wl.num_dim; i < wl.MaxExtraDimensions; i++) {
                    serializer.serialize(SerializableField{"src_stride_" + std::to_string(i + 1), 0});
                    serializer.serialize(SerializableField{"dst_stride_" + std::to_string(i + 1), 0});
                    serializer.serialize(SerializableField{"src_dim_size_" + std::to_string(i + 1), 0});
                    serializer.serialize(SerializableField{"dst_dim_size_" + std::to_string(i + 1), 0});
                }

                serializer.serialize(SerializableField{"num_engine", wl.num_engine});
                serializer.serialize(SerializableField{"direction", wl.transfer_direction});
            } catch (const std::exception& e) {
                Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
                serializer.clean_buffers();
            }
        }
    }
};

}  // namespace VPUNN

#endif  // VPU_COST_MODEL_H
