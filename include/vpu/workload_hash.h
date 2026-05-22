// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_WORKLOAD_HASH_H
#define VPUNN_WORKLOAD_HASH_H

#include "vpu/dpu_workload.h"
#include "vpu/sep_mode.h"
#include "vpu/shave_workload.h"

namespace VPUNN {

/**
 * @brief Static utility class providing FNV-1a based hash functions for VPU workload components.
 *
 * WorkloadHash builds on top of PrimitiveHash to provide composite hashing for the
 * domain-specific data structures used to describe VPU workloads.
 *
 * Like PrimitiveHash, every method follows the incremental hashing pattern: it takes a
 * running hash state @p h and returns an updated hash after folding in all fields of the
 * given structure. Callers typically initialise @p h with @c fnv_offset_basis and chain
 * multiple calls to accumulate a full workload hash.
 *
 * The class is not instantiable; all methods are static.
 */
class WorkloadHash {
private:
    WorkloadHash() = default;  // prevent instantiation
public:
    /**
     * @brief Hash a VPUTensor by folding in its shape, data type, layout, and sparsity flag.
     *
     * Iterates over each dimension of the tensor's shape array, then hashes the data type
     * enum, the layout enum, and the sparsity boolean.
     *
     * @param h       Current running hash state.
     * @param tensor  The VPUTensor whose attributes are folded into the hash.
     * @return Updated hash after incorporating all tensor attributes.
     */
    static uint32_t hash_tensor(uint32_t h, const VPUTensor& tensor);

    /**
     * @brief Hash a HaloWorkload by folding in all six directional padding values for each
     *        of its four halo descriptors (input_0_halo, output_0_halo,
     *        output_0_halo_broadcast_cnt, output_0_inbound_halo).
     *
     * Each descriptor contributes top, bottom, left, right, front, and back values
     * (24 uint32_t fields in total).
     *
     * @param h     Current running hash state.
     * @param halo  The HaloWorkload to fold into the hash.
     * @return Updated hash after incorporating all halo fields.
     */
    static uint32_t hash_halo(uint32_t h, const HaloWorkload& halo);

    /**
     * @brief Hash a SEPModeInfo by folding in the sep_activators flag, storage element
     *        pointer dimensions, actual activator input dimensions, and the no_sparse_map flag.
     *
     * @param h    Current running hash state.
     * @param sep  The SEPModeInfo to fold into the hash.
     * @return Updated hash after incorporating all SEP fields.
     */
    static uint32_t hash_sep(uint32_t h, const SEPModeInfo& sep);

    /**
     * @brief Hash a SHAVE kernel parameter (variant<int, float, string, bool>).
     *
     * The variant's type index is hashed first so that different types holding the same
     * numeric value produce distinct hashes. The contained value is then dispatched to
     * the appropriate PrimitiveHash helper (hash_uint32, hash_float, hash_string, or
     * hash_bool).
     *
     * @param h  Current running hash state.
     * @param p  The Param variant to fold into the hash.
     * @return Updated hash after incorporating the type tag and contained value.
     */
    static uint32_t hash_param(uint32_t h, const SHAVEWorkload::Param& p);
};

} // namespace VPUNN

#endif // VPUNN_WORKLOAD_HASH_H
