// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef DMA_COST_SERIALIZATION_WRAPPER_H
#define DMA_COST_SERIALIZATION_WRAPPER_H

#include "vpu/serialization/serialization_wrapper.h"
namespace VPUNN {
template <class DMADesc>
class DMACostSerializationWrap : public CostSerializationWrap {
private:
    void serialize_workload(const DMANNWorkload_NPU27& wl) {
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

    void serialize_workload(const DMANNWorkload_NPU40_RESERVED& wl) {
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

public:
    DMACostSerializationWrap(CSVSerializer& ser, bool inhibit = false, size_t the_uid = 0)
            : CostSerializationWrap(ser, inhibit, the_uid)  // initialize the base class

    {
    }

    ~DMACostSerializationWrap() = default;

    void serializeDMAWorkload_closeLine(const DMADesc& workload) {
        if (!is_serialization_enabled())
            return;

        // do we need to post process?
        try {
            serializer.serialize(SerializableField<VPUDevice>{"device", workload.device});
            serialize_workload(workload);
            serializer.end();
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
        serializer.clean_buffers();
    }

    void serializeDMAWorkload(const DMADesc& workload) {
        if (!is_serialization_enabled())
            return;

        // do we need to post process?
        try {
            serializer.serialize(SerializableField<VPUDevice>{"device", workload.device});
            serialize_workload(workload);

        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
    }

    void serializeCycles(const CyclesInterfaceType cycles) {
        if (!is_serialization_enabled())
            return;

        try {
            if (!serializer.is_write_buffer_clean()) {
                serializer.serialize(SerializableField<decltype(cycles)>{"cycles", cycles});
                serializer.end();
            }
        } catch (const std::exception& e) {
            Logger::warning() << "Encountered invalid workload while serialization: " << e.what() << "\n";
            set_error();  // mark the error, so next operations will know something is not OK
            serializer.clean_buffers();
        }
        serializer.clean_buffers();
    }
};
}  // namespace VPUNN

#endif  // VPUNN_SERIALIZATION_WRAPPER_H
