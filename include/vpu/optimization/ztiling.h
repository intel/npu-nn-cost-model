// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_ZTILING_H
#define VPUNN_ZTILING_H

#include <list>
#include <set>
#include <string>
#include <vector>

#include "dimension_tiler.h"
#include "tiler.h"
#include "vpu/dpu_types.h"
#include "vpu/layer.h"
#include "vpu/layer_split_info.h"
#include "vpu/ranges.h"

namespace VPUNN {

/**
 * @brief A Tiler child class that implement the ZTiling algorithm
 *
 */
class ZTiling : public ITilerAlgorithm {
public:
    static constexpr bool force_LegacyZTiling{
    // this is used only in .cpp, cmake control should properly affect only  this component
#ifdef VPUNN_OPT_LEGACY_ZTILING
            true
#else
            false
#endif
    };
    /**
     * @brief Using the Tiler class constructor
     *
     */
    using ITilerAlgorithm::ITilerAlgorithm;

    /**
     * @brief Generate a valid pool of splits
     *
     * @param numDPU number of DPUs to optimize with
     * @param valid_execution_mode valid DPU ExecutionMode
     * @return std::vector<unsigned int>
     */
    std::set<unsigned int> generateSplitPool(const unsigned int numDPU,
                                             const ExecutionMode& valid_execution_mode) const override;

    /**
     * @brief Tile multiple workloads using the ZTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                        const unsigned int nWorkloads) override;

    std::string name() const override {
        return "ZTiling";
    }

protected:
    /**
     * @brief Split a DPULayer over the Z dimension, appending the result to the splitPool list
     *
     * HALO ASPECTS: all HW direction halos are passed down to split
     * Output_inbound halo for Z/K  direction technically should be adjusted to fit always the full layer memory, but
     * this is not really necessary for the DPUWorload level because is not used for NN, just for fit to memory check.
     * If the layer fits to memory the intra tile split will also fit. Decision(for now) is to pass and leave unaltered
     * also the output_inbound_halo Z direction.
     *
     * @param layer a DPULayer to be split
     * @param splitPool [out] the pool of valid split
     * @param mode the valid ExecutionMode
     * @param nWorkloads number of splits to apply
     * @param validZTiles valid Z dimension for the splits
     */
    void splitOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                    const unsigned int nWorkloads, const std::vector<unsigned int>& validZTiles);

    DPUWorkload createTileZ(const DPULayer& layer, const unsigned int channels, const unsigned int offset_channels);

    /// these operations do not require a split in Z (under some external conditions)
    //bool isNoSplitOperation(Operation op) const;

    void splitInNOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                       const ExecutionMode mode, const unsigned int nWorkloads, const SplitDimension& theSplitter);

private:
    // VPU ZTiling split constants
    static constexpr unsigned int DEFAULT_ZTILE_VALUE = 16;
    static constexpr unsigned int MIN_VALID_ZTILE_EXPONENT = 4;
    static constexpr unsigned int MAX_VALID_ZTILE_EXPONENT = 8;

    static SmartRanges getValidIntraTilesChannelsOptions(VPUDevice device) {
        // for NPU_RESERVED_1 and beyond we support more than 64 ch for operations like DW_CONV, AVGPOOL, MAXPOOL
        if (device >= VPUDevice::NPU_RESERVED_1)
            return SmartRanges{16, 256, 16, 32};
        return SmartRanges{16, 64, 16, 32};  // 16,32,64
    }

    bool isValidZ(unsigned int channels, const std::vector<unsigned int>& validZTiles);
};

}  // namespace VPUNN

#endif  // ZTILING_H
