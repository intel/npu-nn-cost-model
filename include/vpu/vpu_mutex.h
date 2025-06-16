// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_MUTEX_H
#define VPUNN_VPU_MUTEX_H

#include <mutex>

namespace VPUNN {

/**
 * @brief VPU_MutexAcces is a base class that provides a recursive mutex for thread-safe access to shared resources.
 * It is designed to be inherited by other classes that require synchronized access to shared data.
 *
 * Derive virtually from this class to ensure that all derived classes share the same mutex.
 * Intended to be used for thread level synchronization in the VPU NN library, hence the recursive mutex approach.
 * Should allow the thread to lock multiple times the mutex, like it is in a single thread context and no mutex is
 * present.
 *
 * The mutex is mutable, allowing const member functions to lock the mutex if needed.
 * This is useful for classes that need to provide thread-safe access to their members without modifying them.
 */
class VPU_MutexAcces {
protected:
    mutable std::recursive_mutex L1_mutex{};  ///< should be a base class, so all derived will have access to the same
                                            ///< mutex (DMA/DPU/SHAVE/DCIM)
};

}  // namespace VPUNN

#endif  // VPUNN_VPU_MUTEX_H
