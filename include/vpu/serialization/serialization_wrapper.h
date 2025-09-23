// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZATION_WRAPPER_H
#define VPUNN_SERIALIZATION_WRAPPER_H

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

#include "core/logger.h"
#include "core/serializer.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/dpu_types.h"
#include "vpu/layer_split_info.h"
#include "vpu/validation/dpu_operations_sanitizer.h"
#include "vpu/validation/layer_sanitizer.h"
#include "vpu/vpu_tiling_strategy.h"


namespace VPUNN {

//@todo: add serialization functions for the rest of the classes
class CostSerializationWrap {
protected:
    CSVSerializer& serializer;  ///< Serializer to actually serialize values.
    const bool is_serialization_inhibited{false};

    bool is_serialization_enabled() const {
        // if the environment variable is set, then serialization is enabled
        // return !is_serialization_inhibited && serializer.is_serialization_enabled();
        return is_serialization_enabled(is_serialization_inhibited, serializer);
    }
    inline static bool is_serialization_enabled(const bool inhibit, const CSVSerializer& active_serializer) {
        // if the environment variable is set, then serialization is enabled
        return !inhibit && active_serializer.is_serialization_enabled();
    }

    /// will be set when we catch an error, so teh next operations to know something is not OK
    bool error_present{false};

    /// function that ensures that no error occurred during serialization
    bool is_error_present_during_serialization() const {
        return error_present;
    }
    void set_error() {
        error_present = true;
    }

    size_t serializer_operation_uid{
            0};  ///<  uid used for serialization, computed from layer and context info. Computed in a specific
                 ///< operation and then used. If not computed , default value is OK

private:
public:
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
    CostSerializationWrap(CSVSerializer& ser, bool inhibit = false, size_t the_uid = 0)
            : serializer(ser),
              is_serialization_inhibited(inhibit),

              serializer_operation_uid{the_uid} {
    }

    ~CostSerializationWrap() noexcept{
        // make a clean exit
        cleanBuffers();
    };

    /// prevents to multiple wrappers to share the same references
    CostSerializationWrap(const CostSerializationWrap&) = delete;
    CostSerializationWrap& operator=(const CostSerializationWrap&) = delete;

    /// vpunn_cycles
    void serializeCycles_closeLine(CyclesInterfaceType cost) {
        if (!is_serialization_enabled() || is_error_present_during_serialization())
            return;
        // Cost model software DPU development
        try {
            serializer.serialize(SerializableField<CyclesInterfaceType>{"vpunn_cycles", cost});

            serializer.end();  // is it OK to end during a series of writings?(NO)
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
    }

private:
    /// because this function is used into descriptor it should be noexcept
    void cleanBuffers() noexcept{ 
        try {
            if (is_serialization_enabled()) {
                serializer.clean_buffers();
            }
        } catch (...) {
            // ignore
            }
    }

private:
};
}  // namespace VPUNN

#endif  // VPUNN_SERIALIZATION_WRAPPER_H
