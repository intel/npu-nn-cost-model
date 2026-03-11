// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu_cost_model.h"

#include <algorithm>
#include <memory>  // for std::make_shared, std::make_unique
#include <mutex>
#include <optional>   // for std::optional (if needed)
#include <stdexcept>  // for std::runtime_error
#include <string>     // for std::string, std::stoi
#include <tuple>      // for std::tuple, std::make_tuple
#include "core/logger.h"
#include "core/utils.h"
#include "vpu/serialization/l1_cost_serialization_wrapper.h"
#include "vpu/shave/shave_collection.h"
#include "vpu/types.h"
#include "vpu/utils.h"
#include "vpu/validation/checker_utils.h"
#include "vpu/validation/data_dpu_operation.h"  // for DPUOperation
#include "vpu/validation/sanity_report.h"       // for SanityReport

namespace VPUNN {

void VPUCostModel::channels_preserving_operations_consistency_check(DPUWorkload& workload) const {
    if (workload.op == Operation::ELTWISE || workload.op == Operation::DW_CONVOLUTION ||
        workload.op == Operation::MAXPOOL || workload.op == Operation::AVEPOOL) {
        if (!workload.is_output_autopad() && workload.outputs[0].channels() >= 16) {
            if (workload.inputs[0].channels() != workload.outputs[0].channels()) {
                Logger::warning() << "Changed channels from " << workload.inputs[0].channels() << " to "
                                  << workload.outputs[0].channels();

                workload.inputs[0].set_shape({workload.inputs[0].x(), workload.inputs[0].y(),
                                              workload.outputs[0].channels(), workload.inputs[0].b()});
            }
        }
    }
}

void VPUCostModel::avgpool_replace_by(DPUWorkload& workload) const {
    if (Operation::AVEPOOL == workload.op) {
        if (VPUDevice::NPU_5_0 > workload.device) {
            Logger::warning() << "Workload with AVEPOOL changed to DW_CONVOLUTION";
            workload.op = Operation::DW_CONVOLUTION;
        } else {
            Logger::warning() << "Workload with AVEPOOL  NOT changed!, Device should support it!, ";
        }
    }
}

void VPUCostModel::compressConv_replace_by_CM_CONV_VPU27(DPUWorkload& workload) const {
    if (workload.device >= VPUDevice::VPU_2_7) {
        if ((workload.op == Operation::CONVOLUTION) &&
            ((workload.inputs[0].channels() >= 1) && (workload.inputs[0].channels() < 16) &&
             !workload.is_input_autopad()  // qmark if CONV with ch <16 exists with autopad?
             )) {
            Logger::warning() << "Workload with CONVOLUTION compressed IC[1..15] transformed to CM_CONV ";
            workload.op = Operation::CM_CONVOLUTION;
        }
    }
}

void VPUCostModel::swizzling_turn_OFF(DPUWorkload& workload) const {
    if constexpr (false == PerformanceMode::allowLegacySwizzling_G5) {
        // only for some devices
        if (workload.device >= VPUDevice::NPU_5_0) {
            workload.set_all_swizzlings(Swizzling::KEY_0);
        }
    }
}

VPUCostModel::VPUCostModel(const std::string& filename, const std::string& dpu_cache_filename,
                           const std::string& shave_cache_filename, bool tryToLoadPairedCache)
        : VPUCostModel(filename, false, 16384, 1, dpu_cache_filename, shave_cache_filename, tryToLoadPairedCache) {
    /* coverity[uninit_member] */
}

VPUCostModel::VPUCostModel(const std::string& filename, bool profile, const unsigned int cache_size,
                           const unsigned int batch_size, const std::string& dpu_cache_filename,
                           const std::string& shave_cache_filename, bool tryToLoadPairedCache)
        : dpu_nn_cost_provider(filename, batch_size, profile, cache_size, dpu_cache_filename, tryToLoadPairedCache),
          ptr_internal_shave_cost_model(std::make_shared<SHAVECostModel>(shave_cache_filename, cache_size)),
          http_dpu_cost_provider(HttpCostProviderFactory::create()) {
    Logger::initialize();

    if (!dpu_nn_cost_provider.is_initialized()) {
        return;
    }

    serializer.initialize("l1_dpu_workloads", FileMode::READ_WRITE,
                          dpu_nn_cost_provider.get_names_for_serializer());
}

VPUCostModel::VPUCostModel(const char* model_data, size_t model_data_length, bool copy_model_data, bool profile,
                           const unsigned int cache_size, const unsigned int batch_size, const char* dpu_cache_data,
                           size_t dpu_cache_data_length, const char* shave_cache_data, size_t shave_cache_data_length)
        : dpu_nn_cost_provider(model_data, model_data_length, batch_size, copy_model_data, profile, cache_size,
                               dpu_cache_data, dpu_cache_data_length),
          ptr_internal_shave_cost_model(
                  std::make_shared<SHAVECostModel>(shave_cache_data, shave_cache_data_length, cache_size)),
          http_dpu_cost_provider(HttpCostProviderFactory::create()) {
    Logger::initialize();

    if (!dpu_nn_cost_provider.is_initialized()) {
        return;
    }

    serializer.initialize("l1_dpu_workloads", FileMode::READ_WRITE, dpu_nn_cost_provider.get_names_for_serializer());
}

bool VPUCostModel::sanitize_workload(DPUWorkload& workload, SanityReport& result) const {
    avgpool_replace_by(workload);  // AVEPOOL will be transformed to something equivalent
    compressConv_replace_by_CM_CONV_VPU27(workload);

    channels_preserving_operations_consistency_check(workload);  // old style sanitation

    sanitizer.check_and_sanitize(workload, result);
    return result.is_usable();
}

CyclesInterfaceType VPUCostModel::get_cost_dualsparsity(const DPUWorkload& workload) const {
    // run twice
    std::string info;
    const auto wl_noAct{cloneDeactivateActSparsity(workload)};
    const CyclesInterfaceType act_off = run_cost_providers(wl_noAct, info);

    const auto wl_noWt{cloneDeactivateWeightSParsity(workload)};
    const CyclesInterfaceType wt_off = run_cost_providers(wl_noWt, info);

    // case when act_off is invalid, also this catch the case when both of values are invalid
    //   does not count which of them we return if both are invalid
    if (Cycles::isErrorCode(act_off)) {
        return act_off;
    }

    // case when wt_off is invalid
    if (Cycles::isErrorCode(wt_off)) {
        return wt_off;
    }

    // MIN.. their combined runtime reduction should be at least like their independent one?
    // IN general CostMOdelis a best case (due to memory contention, DMA ), but here  we keep this not so best case
    // (too many variations) alternative: more policies and we chose which one has best FPS over models optimization
    // on min:  run only once , for max sparsity (maybe balanced...)
    CyclesInterfaceType ret{std::min(act_off, wt_off)};  // algorithm here

    return ret;  // exit point
}

CyclesInterfaceType VPUCostModel::get_cost(const DPUWorkload& workload, std::string& info,
                                           std::string* cost_source) const {
    // @todo : impact on energy?, CHeck if energy/DPUINfo/Theoretical cycles/ops considers this situation to reduce
    // energy

    // if weight and input (SEP irrelevant?) sparsity ON.
    if (is_dualsparsity_active(workload)) {
        return get_cost_dualsparsity(workload);
    }

    return run_cost_providers(workload, info, cost_source);  // normal wl handling
}

CyclesInterfaceType VPUCostModel::run_cost_providers(const DPUWorkload& workload, std::string& info,
                                                     std::string* cost_source) const {
    CyclesInterfaceType cycles{Cycles::NO_ERROR};
    const bool is_inference_possible = dpu_nn_cost_provider.is_initialized();

    const auto try_cache = [&]() -> CyclesInterfaceType {
        return dpu_nn_cost_provider.get_cached(workload, cost_source);
    };
    const auto try_profiling = [&]() -> CyclesInterfaceType {
        if (http_dpu_cost_provider) {
            if (cost_source) {
                *cost_source =
                        "profiling_service_" + http_dpu_cost_provider->profilingBackendToString(workload.profiling_service_backend_hint);
            }

            auto dpu_op = DPUOperation(workload, sanitizer.getDeviceConfiguration(workload.device));
            return http_dpu_cost_provider->getCost(dpu_op, info);
        } else {
            return Cycles::ERROR_PROFILING_SERVICE;
        }
    };
    const auto try_nn = [&]() -> CyclesInterfaceType {
        if (!is_inference_possible) {
            return Cycles::ERROR_INFERENCE_NOT_POSSIBLE;
        }
        if (cost_source) {
            *cost_source = "nn_" + dpu_nn_cost_provider.get_model_nickname();
        }

        if (is_linearly_extrapolation_necessary(workload)) {
            return get_cost_linearly_extrapolated(workload);
        }
        return dpu_nn_cost_provider.get_cost(workload);
    };
    const auto try_theoretical = [&]() -> CyclesInterfaceType {
        if (cost_source) {
            *cost_source = "theoretical";
        }
        return dpu_theoretical.DPUTheoreticalCycles(workload);
    };

    if (workload.cost_source_hint == CostSourceHint::AUTO) {
        // 1. Cache
        const auto cached = try_cache();
        if (!Cycles::isErrorCode(cached)) {
            return cached;
        }

        // 2. Profiling service
        cycles = try_profiling();

        // 3. Fallbacks (NN then theoretical)
        if (Cycles::isErrorCode(cycles) || cycles == Cycles::NO_ERROR) {
            cycles = try_nn();
            if (Cycles::isErrorCode(cycles)) {
                cycles = try_theoretical();
            }
        }

        // Share result with NN cache if needed (only if cache had no valid entry)
        if (Cycles::isErrorCode(dpu_nn_cost_provider.get_cached(workload)) && !Cycles::isErrorCode(cycles)) {
            dpu_nn_cost_provider.add_to_cache(workload, static_cast<float>(cycles));
        }
        return cycles;

    } else if (workload.cost_source_hint == CostSourceHint::PROFILING_SERVICE) {
        cycles = try_profiling();

    } else if (workload.cost_source_hint == CostSourceHint::NN) {
        cycles = try_nn();

    } else if (workload.cost_source_hint == CostSourceHint::THEORETICAL) {
        cycles = try_theoretical();
    }

    return cycles;
}

DPUWorkload VPUCostModel::cloneDeactivateActSparsity(const DPUWorkload& wl) const {
    DPUWorkload out{wl};
    out.act_sparsity = 0.0;
    out.inputs[0].set_sparsity(false);
    return out;
}

DPUWorkload VPUCostModel::cloneDeactivateWeightSParsity(const DPUWorkload& wl) const {
    DPUWorkload out{wl};
    out.weight_sparsity = 0.0;
    out.weight_sparsity_enabled = false;
    return out;
}

DPUWorkload VPUCostModel::cloneAndChangeInOutChannels(const DPUWorkload& wl_, const unsigned int channels) const {
    DPUWorkload wl{wl_};
    wl.inputs[0].set_shape({wl.inputs[0].width(), wl.inputs[0].height(), channels, wl.inputs[0].batches()});
    wl.outputs[0].set_shape({wl.outputs[0].width(), wl.outputs[0].height(), channels, wl.outputs[0].batches()});

    return wl;
}

const std::vector<CyclesInterfaceType> VPUCostModel::get_cost(const std::vector<DPUWorkload>& workloads) const {
    // here we check if at least one wl in vector workloads has act sparsity and weight sparsity active
    const bool exists_dual_spars{
            std::any_of(workloads.cbegin(), workloads.cend(), VPUCostModel::is_dualsparsity_active)};

    if (exists_dual_spars) {
        std::vector<CyclesInterfaceType> cycles_vector;
        cycles_vector.reserve(workloads.size());
        std::transform(workloads.cbegin(), workloads.cend(), cycles_vector.begin(), [this](const DPUWorkload& wl) {
            std::string info, source;
            return get_cost(wl, info, &source);
        });

        return cycles_vector;
    } else {
        return dpu_nn_cost_provider.get_cost(workloads);  // normal execution
    }
}

std::vector<float> VPUCostModel::getDescriptor(const DPUWorkload& wl) const {
    return dpu_nn_cost_provider.getDescriptor(wl);
}

/* coverity[pass_by_value] */
std::tuple<CyclesInterfaceType, std::string> VPUCostModel::DPUMsg(DPUWorkload wl) const {
    std::string dummy_info{};
    auto previous_print_mode{Checker::set_print_tags(false)};
    const auto r{DPU(wl, dummy_info)};
    Checker::set_print_tags(previous_print_mode);
    return std::make_tuple(r, dummy_info);
}

CyclesInterfaceType VPUCostModel::DPU(DPUWorkload wl, std::string& info) const {
    return DPU_and_sanitize(wl, info);
}

CyclesInterfaceType VPUCostModel::DPU_and_sanitize(DPUWorkload& wl, std::string& info) const {
    swizzling_turn_OFF(wl);  // swizz guard sanitization


    L1CostSerializationWrap serialization_handler(serializer);

    serialization_handler.serializeInfoAndComputeWorkloadUid(wl);

    // sanitize and check the input.
    SanityReport problems{};
    const auto is_inference_relevant = sanitize_workload(wl, problems);
    info = problems.info;

    std::string cost_source = "unknown";
    CyclesInterfaceType cycles{problems.value()};  // neutral value or reported problems at sanitization

    if (is_inference_relevant) {
        cycles = get_cost(wl, info, &cost_source);
    }

    serialization_handler.serializeCyclesAndCostInfo_closeLine(cycles, std::move(cost_source), info);

    return cycles;
}

std::vector<CyclesInterfaceType> VPUCostModel::DPU(std::vector<DPUWorkload> workloads) const {
    std::vector<DPUWorkload> serializer_orig_wls;  // should be const
    if (serializer.is_serialization_enabled()) {   // has to be factored out
        serializer_orig_wls = std::vector<DPUWorkload>(workloads.size());
        std::copy(workloads.begin(), workloads.end(), serializer_orig_wls.begin());
    }

    const auto number_of_workloads{workloads.size()};  ///< fixed value remembered here, workloads is non const
    std::vector<CyclesInterfaceType> cycles_vector = std::vector<CyclesInterfaceType>(number_of_workloads);
    const auto is_inference_posible = nn_initialized();

    /// @brief sanitization result element
    struct sanitizationOutcome {
        bool inference_relevance{false};
        SanityReport problems{};
    };
    std::vector<sanitizationOutcome> sanitization_results{number_of_workloads};

    // sanitize the input vector.
    for (unsigned int idx = 0; idx < number_of_workloads; ++idx) {
        auto& wl{workloads[idx]};
        auto& sanity{sanitization_results[idx]};
        swizzling_turn_OFF(wl);                                               // swizz guard sanitization
        sanity.inference_relevance = sanitize_workload(wl, sanity.problems);  // workloads are changed
    }

    // Compute using NN. Should not run if not initialized (fills a default value)
    const auto costs = get_cost(workloads);  // always tentative run, what if throws?

    // parse all and decide individually
    for (unsigned int idx = 0; idx < workloads.size(); ++idx) {
        auto& wl{workloads[idx]};
        const SanityReport& problems{sanitization_results[idx].problems};
        const auto is_inference_relevant{sanitization_results[idx].inference_relevance};
        const auto theoretical_cycles = dpu_theoretical.DPUTheoreticalCycles(wl);

        CyclesInterfaceType cycles{problems.value()};  // neutral value or sanitization error
        if (is_inference_relevant) {
            if (is_inference_posible) {
                cycles = costs[idx];

            } else {  // NN not available, use theoretical cycles
                cycles = theoretical_cycles;
            }
        }

        cycles_vector[idx] = cycles;
    }

    const std::string dpu_nickname{get_NN_cost_provider().get_model_nickname()};
    L1CostSerializationWrap serialization_handler(serializer);
    serialization_handler.serializeCyclesAndComputeWorkloadUid_closeLine(std::move(serializer_orig_wls), cycles_vector,
                                                                         dpu_nickname);

    return cycles_vector;
}

unsigned int VPUCostModel::DMA(VPUDevice device, const VPUTensor& input, const VPUTensor& output,
                               MemoryLocation input_location, MemoryLocation output_location,
                               unsigned int output_write_tiles) const {
    // Call the helper function. TO DO Adjust theoretical based on some measured data!
    return dma_theoretical.get_cost({device, input, output, input_location, output_location, output_write_tiles});
}

unsigned int VPUCostModel::DMA(const DMAWorkload& wl) const {
    return DMA(wl.device, wl.input, wl.output, wl.input_location, wl.output_location, wl.output_write_tiles);
}

CyclesInterfaceType VPUCostModel::SHAVE(const SHAVEWorkload& shave_wl, std::string& infoOut) const {
    return internal_shave_cost_model.computeCycles(shave_wl, infoOut);
}

CyclesInterfaceType VPUCostModel::SHAVE(const SHAVEWorkload& shave_wl) const {
    return internal_shave_cost_model.computeCycles(shave_wl);
}

CyclesInterfaceType VPUCostModel::SHAVE(const SHAVEWorkload& shave_wl, std::string& infoOut,
                                        bool skipCacheValues) const {
    return internal_shave_cost_model.computeCycles(shave_wl, infoOut, skipCacheValues);
}

std::vector<std::string> VPUCostModel::getShaveSupportedOperations(VPUDevice device) const {
    return internal_shave_cost_model.getShaveSupportedOperations(device);
}

const ShaveOpExecutor& VPUCostModel::getShaveInstance(std::string name, VPUDevice device) const {
    const std::string original_name = name;  // Store the name before moving
    auto result = internal_shave_cost_model.getShaveInstance(std::move(name), device);
    if (!result.has_value()) {
        throw std::runtime_error("Shave instance not found for name: " + original_name);
    }
    return result.value().get();
}

float VPUCostModel::DPUEnergy(const DPUWorkload& wl) const {
    return getEnergyInterface().DPUEnergy(wl);
}

float VPUCostModel::SHAVEEnergy(const SHAVEWorkload& swl) const {
    return getEnergyInterface().SHAVEEnergy(swl);
}

DPUInfoPack VPUCostModel::DPUInfo(const DPUWorkload& workload) const {
    DPUInfoPack allData;      // expect RVO when returning it!
    DPUWorkload w{workload};  // local clone

    allData.DPUCycles = DPU_and_sanitize(
            w, allData.errInfo);  // do this first, might change w. It considers both sparsities if activated

    getEnergyInterface().fillDPUInfo(allData, w);

    allData.hw_theoretical_cycles = dpu_theoretical.DPUTheoreticalCycles(w);

    return allData;  // rvo
}

bool VPUCostModel::is_linearly_extrapolation_necessary(const DPUWorkload& wl) const {
    const bool is_intratile_like_op{wl.op == Operation::AVEPOOL || wl.op == Operation::DW_CONVOLUTION ||
                                    wl.op == Operation::MAXPOOL};

    const bool is_ch_greater_than_64{wl.inputs[0].channels() > 64};

    return ((is_intratile_like_op && is_ch_greater_than_64)
                    ?  //
                       //(!dpu_nn_cost_provider.get_preprocessing().supportsProperty(
                       //        "DW_MXP_AVP_SupportsMoreThan64Ch")  // TODO: to be removed in the future, calling a
                       //        guts
                       //                                            // of a sub object is in principle a violation of
                       //                                            // Demeter's law
                       // )
                    (!is_linearly_extrapolation_necessary_cache_capability)
                    : false);
}

CyclesInterfaceType VPUCostModel::get_cost_linearly_extrapolated(const DPUWorkload& workload) const {
    const unsigned int ch32 = 32;  // 32 channels
    const unsigned int ch64 = 64;  // 64 channels

    // Get costs for the two reference points
    const auto wl_32ch{cloneAndChangeInOutChannels(workload, ch32)};
    const CyclesInterfaceType cost_32ch = dpu_nn_cost_provider.get_cost(wl_32ch);

    const auto wl_64ch{cloneAndChangeInOutChannels(workload, ch64)};
    const CyclesInterfaceType cost_64ch = dpu_nn_cost_provider.get_cost(wl_64ch);

    // Handle error cases - if either reference point fails, return the error
    if (Cycles::isErrorCode(cost_32ch)) {
        return cost_32ch;
    }

    if (Cycles::isErrorCode(cost_64ch)) {
        return cost_64ch;
    }

    // Both reference points are valid - proceed with linear extrapolation
    // Mathematical derivation:
    //   Linear function: f(ch) = slope × ch + intercept
    //   Using the two reference points:
    //     slope = delta_cost/delta_ch = (cost_64ch - cost_32ch) / (64 - 32) = (cost_64ch - cost_32ch) / 32
    //     intercept = cost_32ch - slope × 32

    const unsigned int original_channels = workload.outputs[0].channels();

    const CyclesInterfaceType extrapolated_cost =
            Cycles::extrapolate_cost(original_channels, ch32, cost_32ch, ch64, cost_64ch);

    return extrapolated_cost;
}

}  // namespace VPUNN
