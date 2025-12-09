// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/optimization/tiler.h"
#include "vpu/optimization/splits.h"

#include "vpu/optimization/dimension_tiler.h"

namespace VPUNN {

// VPU split constants
constexpr unsigned int DEFAULT_ZTILE_VALUE = 16;
constexpr unsigned int MIN_VALID_ZTILE_EXPONENT = 4;
constexpr unsigned int MAX_VALID_ZTILE_EXPONENT = 8;

/**
 * @brief Infers input shape from a DPU workload and its original layer. It modify the DPUWorkload
 *
 * @param wl [in, out] DPUWorkload that has to have the input computed based on output and kernel. This is a workload
 * part of the originalLayer but split on  one or more DPUs
 * @param originalLayer the DPULayer that is the source of the workloads split on one or more DPU's
 */
static void inferInputTensorShape(DPUWorkload& wl, const DPULayer& originalLayer) {
    // const auto input_width = (wl.outputs[0].width() - 1) * wl.strides[Dim::Grid::W] + wl.kernels[Dim::Grid::W] -
    //                         wl.padding[Dim::Padding::LEFT] - wl.padding[Dim::Padding::RIGHT];
    // const auto input_height = (wl.outputs[0].height() - 1) * wl.strides[Dim::Grid::H] + wl.kernels[Dim::Grid::H] -
    //                          wl.padding[Dim::Padding::TOP] - wl.padding[Dim::Padding::BOTTOM];

    const auto input_width = helper_input_dim(wl.outputs[0].width(), wl.kernels[Dim::Grid::W],
                                              wl.padding[Dim::Padding::LEFT] + wl.padding[Dim::Padding::RIGHT],
                                              wl.strides[Dim::Grid::W]);

    const auto input_height = helper_input_dim(wl.outputs[0].height(), wl.kernels[Dim::Grid::H],
                                               wl.padding[Dim::Padding::TOP] + wl.padding[Dim::Padding::BOTTOM],
                                               wl.strides[Dim::Grid::H]);

    // the workload can be a result of HW split or Z split
    // for HW split the input channels remain unaffected (operation irrelevant)
    //
    // for Z split: the output (of 1 DPU workload) has only a fraction of the original z. 
    // For the operations that use the full input channels in their kernel (e.g. CONV, CM_CONV)  
    // - the input's Z should be again the full original input channels, there was no split of input Z
    // For the operation that are not having a kernel with depth (e.g. ELEMENTWISE)
    // - the input's Z should be equal to output's Z  (as a more general rule)
    const auto input_channel =
            ((wl.op == Operation::CONVOLUTION) || (wl.op == Operation::CM_CONVOLUTION) || wl.output_autopad)  // kernels need all input Z
                    ? wl.inputs[0].z()  // use what the split left here by default (maybe original z, maybe a Z split of
                                        // inputs(future?))
                    : wl.outputs[0].z();  // for elementwise operations the in-out channels should match

    const auto input_batch = wl.outputs[0].batches();

    const auto inputTensor =
            VPUTensor({input_width, input_height, input_channel, input_batch}, originalLayer.inputs[0]);
    wl.inputs[0] = inputTensor;

    // sanitary limitation
    {  // limit halo height
        if (wl.halo.input_0_halo.top > (int)wl.inputs[0].height()) {
            wl.halo.input_0_halo.top = (int)wl.inputs[0].height();
        }
        if (wl.halo.input_0_halo.bottom > (int)wl.inputs[0].height()) {
            wl.halo.input_0_halo.bottom = (int)wl.inputs[0].height();
        }
    }
    {  // limit width
        if (wl.halo.input_0_halo.left > (int)wl.inputs[0].width()) {
            wl.halo.input_0_halo.left = (int)wl.inputs[0].width();
        }
        if (wl.halo.input_0_halo.right > (int)wl.inputs[0].width()) {
            wl.halo.input_0_halo.right = (int)wl.inputs[0].width();
        }
    }

    {  // limit halo channels/depth
        if (wl.halo.input_0_halo.front > (int)wl.inputs[0].channels()) {
            wl.halo.input_0_halo.front = (int)wl.inputs[0].channels();
        }
        if (wl.halo.input_0_halo.back > (int)wl.inputs[0].channels()) {
            wl.halo.input_0_halo.back = (int)wl.inputs[0].channels();
        }
    }
}

void ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(DPUWorkloadsWithCyclesSplit& workloads_split,
                                                          const ExecutionMode mode, const DPULayer& originalLayer) {
    for (auto& wl : workloads_split.workloads) {
        wl.execution_order = mode;
        inferInputTensorShape(wl, originalLayer);
    }
}

/**
 * @brief Generate splits values from a range
 *
 * @param maxSplitRange the maximum split range as a power of two
 * @param maxLimit the absolute maximum split value
 * @return std::vector<unsigned int>
 */
static std::vector<unsigned int> getSplitsFromRange(const unsigned int maxSplitRange, const unsigned int maxLimit) {
    std::vector<unsigned int> splits;
    for (unsigned int idx = 0; idx < std::log2(maxSplitRange); idx++) {
        auto powIdx = static_cast<unsigned int>(std::pow(2, idx));
        auto splitCandidate = maxSplitRange / powIdx;
        if (maxSplitRange % powIdx == 0 && splitCandidate <= maxLimit) {
            splits.push_back(splitCandidate);
        }
    }
    return splits;
}

/**
 * @brief Return true if the layer require a maximum size in Z
 * ALso NOT suitable for HW tiling!
 * CM_CONV is not allowed for split, (Hw due to this presence)??
 *
 * @param layer the DPULayer object to optimize
 */
static bool requireMaxZTile(const DPULayer& layer) {
    if (layer.device >= VPUDevice::VPU_2_7) {
        if (                                          /*layer.op == Operation::CM_CONVOLUTION || */
            layer.op == Operation::MAXPOOL ||         //
            layer.op == Operation::DW_CONVOLUTION ||  //
            layer.op == Operation::AVEPOOL            //
        ) {
            return true;
        }
    }
    return false;
}

std::list<DPUWorkloadsWithCyclesSplit> ITilerAlgorithm::split_tile_in_workloads(const ExecutionMode mode,
                                                                                const unsigned int nWorkloads) {
    std::list<DPUWorkloadsWithCyclesSplit> splitPool;
    // Optimized for 1 workloads  (todo Analyse if necessary,  what if it is not valid?)
    if (nWorkloads == 1) {
        DPUWorkloadsWithCyclesSplit workloads_split{{Cycles::NO_ERROR}, {layer_on_tile}};  // same as original
        ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode,
                                                             layer_on_tile);  // computes also input tensor
        splitPool.push_back(std::move(workloads_split));
    } else {
        tileMultipleWl(splitPool, mode, nWorkloads);
    }

    return splitPool;
}

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
                                             const ExecutionMode& valid_execution_mode) const override {
        if (numDPU == 1 && isNoSplitOperation(this->layer_on_tile.op)) {
            return {1U};  // why? CONV and ELEMwisae are not split?
        }

        // Enable ZTiling only for VPU2.0 in vector mode for non conv layers
        // This is a VPUX specific behavior we need to keep
        if ((this->layer_on_tile.device == VPUDevice::VPU_2_0) &&
            ((valid_execution_mode != ExecutionMode::VECTOR) &&
             (this->layer_on_tile.op != VPUNN::Operation::CONVOLUTION))) {
            return {1U};
        }

        // The max number of splits in the Z dimension
        std::vector<unsigned int> maxSplitsInZ;
        {
            std::vector<unsigned int> validZTiles;
            // 2^4 equals to the CMX word size in bytes,  2^8 is an up bound to limit the number of splits
            for (unsigned int i = MIN_VALID_ZTILE_EXPONENT; i < MAX_VALID_ZTILE_EXPONENT; ++i) {
                validZTiles.push_back(static_cast<unsigned int>(std::pow(2, i)));
                validZTiles.push_back(validZTiles.back() + DEFAULT_ZTILE_VALUE);
            }

            for (const auto& zTile : validZTiles) {
                maxSplitsInZ.push_back(static_cast<unsigned int>(
                        std::ceil(layer_on_tile.outputs[0].z() / static_cast<double>(zTile))));
            }
        }

        const auto maxZ = *std::max_element(maxSplitsInZ.begin(), maxSplitsInZ.end());
        const auto maxSplits = std::min(maxWorkloads, maxZ);

        std::set<unsigned int> dpuMulSplits;  //< unique values, sorted!
        for (auto i = numDPU; i < maxSplits + 1U; i += numDPU) {
            dpuMulSplits.insert(i);
        }
        for (auto splitsZ : maxSplitsInZ) {
            auto zRanges = getSplitsFromRange(splitsZ, maxSplits);
            dpuMulSplits.insert(zRanges.begin(), zRanges.end());
        }
        dpuMulSplits.insert(1U);

        return dpuMulSplits;
    }

    /**
     * @brief Tile multiple workloads using the ZTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                        const unsigned int nWorkloads) override {
        if (layer_on_tile.device < VPUDevice::NPU_5_0  //
            || (force_LegacyZTiling)                   // Every device runs as before
        ) {                                            // Some layers have a max size in Z by specification
            const std::vector<unsigned int> validZTiles{requireMaxZTile(layer_on_tile)
                                                                ? std::vector<unsigned int>({16, 32, 64})
                                                                  : std::vector<unsigned int>({})};
            splitOverZ(layer_on_tile, splitPool, mode, nWorkloads, validZTiles);
        } else {  // experimental for new devices
            const SplitDimension splitter{
                    requireMaxZTile(layer_on_tile) ? SmartRanges{16, 64, 16, 32}  // 16,32,64
                                                   : SmartRanges{1, 8192, 16}     // div 16
            };
            splitInNOverZ(layer_on_tile, splitPool, mode, nWorkloads, splitter);
        }
    }

    std::string name() const override {
        return "ZTiling";
    }

protected:
    /// these operations do not require a split in Z (under some external conditions)
    bool isNoSplitOperation(Operation op) const {
        return (op == Operation::CONVOLUTION        //
                || op == Operation::ELTWISE         //
                || op == Operation::CM_CONVOLUTION  //
                || op == Operation::LAYER_NORM      //
                || op == Operation::ELTWISE_MUL);
    }

    void splitInNOverZ(const DPULayer& layer, std::list<DPUWorkloadsWithCyclesSplit>& splitPool,
                       const ExecutionMode mode, const unsigned int nWorkloads, const SplitDimension& theSpliter) {
        SplitDimension::SplitContainer split_bins{};  // empty
        const int dimensionToSplit{static_cast<int>(layer.outputs[0].channels())};
        const bool split_status = theSpliter.divideBalanced(dimensionToSplit, nWorkloads, split_bins);

        if (!split_status) {
            return;  // no valid split
        }
        if (split_bins.size() != nWorkloads) {
            return;  // no valid split, ERROR in splitter
        }

        DPUWorkloadsWithCyclesSplit workloads_split;
        int offset_channels{0};  // used/incremented  in  loop
        for (const auto& split : split_bins) {
            const auto actual_channels{split};

            if (actual_channels <= 0) {
                return;  // invalid split
            }

            workloads_split.workloads.emplace_back(
                    createTileZ(layer, static_cast<unsigned>(actual_channels), static_cast<unsigned>(offset_channels)));
            workloads_split.cycles.emplace_back(Cycles::NO_ERROR);

            offset_channels += static_cast<unsigned>(actual_channels);  // for next iteration
        }

        ITilerAlgorithm::setWorkloadsModeAndInfereInputShape(workloads_split, mode,
                                                             layer);  // computes also input tensor
        splitPool.push_back(std::move(workloads_split));
    }
};

/**
 * @brief A Tiler child class that implement the HWTiling algorithm
 *
 */
class HWTiling : public ITilerAlgorithm {
public:
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
                                             const ExecutionMode& valid_execution_mode) const override {
        std::set<unsigned int> dpuMulSplits{1};  //< unique values, sorted!
        if (numDPU == 1) {
            return dpuMulSplits;
        }
        // Get the min grid size from valid MPE grid size
        const auto grid = mpe_mode_to_grid(valid_execution_mode);
        const unsigned int grid_x{grid[Dim::Grid::W]};
        const unsigned int grid_y{grid[Dim::Grid::H]};

        std::vector<unsigned int> maxSplitsInXY;
        maxSplitsInXY.push_back(static_cast<unsigned int>(std::ceil(layer_on_tile.outputs[0].y() / grid_x) *
                                                          std::ceil(layer_on_tile.outputs[0].x() / grid_y)));

        const auto maxXY = *std::max_element(maxSplitsInXY.begin(), maxSplitsInXY.end());
        const auto maxSplits = std::min(maxWorkloads, maxXY);

        for (auto i = numDPU; i < maxSplits + 1; i = i + numDPU) {
            dpuMulSplits.insert(static_cast<uint32_t>(i));
        }

        for (auto splitsXY : maxSplitsInXY) {
            const auto xyRanges = getSplitsFromRange(splitsXY, maxSplits);
            dpuMulSplits.insert(xyRanges.begin(), xyRanges.end());
        }

        return dpuMulSplits;
    }

    /**
     * @brief Tile multiple workloads using the ZTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    virtual void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                                const unsigned int nWorkloads) override {
        // Get each pair of factor of nWorkloads (largest, smallest)
        for (const auto& factor : getFactors(nWorkloads)) {
            // Map factor.first , factor.second -> width, height
            if (factor.first <= layer_on_tile.outputs[0].x() && factor.second <= layer_on_tile.outputs[0].y()) {
                tileOverHW(splitPool, factor.first, factor.second, mode);
            }
        }
    }

    std::string name() const override {
        return "HWTiling";
    }

protected:
    /**
     * @brief Tile a DPULayer over H and W
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param widthFactor the number of splits in the X dimension
     * @param heightFactor  the number of splits in the Y dimension
     * @param mode the selected ExecutionMode
     */
    void tileOverHW(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const unsigned int widthFactor,
                    const unsigned int heightFactor, const ExecutionMode mode) {
        splitOverHW(layer_on_tile, splitPool, widthFactor, heightFactor, mode);
    }

    /**
     * @brief Get the prime factor of N
     *
     * @param n a natural number
     * @return std::list<std::pair<unsigned int, unsigned int>>
     */
    std::list<std::pair<unsigned int, unsigned int>> getFactors(unsigned int n) {
        std::list<std::pair<unsigned int, unsigned int>> factors;
        for (unsigned int i = 1; i <= sqrt(n); i++) {
            if (n % i == 0) {
                factors.emplace_back(n / i, i);  // larger, smaller
                factors.emplace_back(i, n / i);  // smaller, larger
            }
        }
        return factors;
    }
};

/**
 * @brief A Tiler child class that implement the HTiling algorithm
 *
 */
class HTiling : public HWTiling {
public:
    /**
     * @brief Using the HWTiling class constructor
     *
     */
    using HWTiling::HWTiling;

    /**
     * @brief Tile multiple workloads using the ZTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                        const unsigned int nWorkloads) override {
        tileOverHW(splitPool, 1, nWorkloads, mode);
    }
    std::string name() const override {
        return "HTiling";
    }
};

/**
 * @brief A Tiler child class that implement the WTiling algorithm
 *
 */
class WTiling : public HWTiling {
public:
    /**
     * @brief Using the HWTiling class constructor
     *
     */
    using HWTiling::HWTiling;

    /**
     * @brief Tile multiple workloads using the ZTiling algorithm
     *
     * @param splitPool the pool of valid splits returned by this function
     * @param mode the MPE mode
     * @param nWorkloads the number of workloads to generate
     */
    void tileMultipleWl(std::list<DPUWorkloadsWithCyclesSplit>& splitPool, const ExecutionMode mode,
                        const unsigned int nWorkloads) override {
        tileOverHW(splitPool, nWorkloads, 1, mode);
    }
    std::string name() const override {
        return "WTiling";
    }
};

/**
 * @brief Return true if it is possible to tile over H and/or W dimension.
 * If number of DPU's (inside tile) is 1 and ZTiling is in list, than nothing else except ZTiling is permitted
 * If the operation requires a max CHannel size (like 16) than HW tiling is not allowed
 *
 * @param layer the DPULayer object to optimize
 * @param options that control the tiling
 * @returns true if the layer support HWTiling
 */
static bool isHWTilingAllowed(const DPULayer& layer, const SplitOptions& options) {
    const auto& strategies = options.availableStrategies;
    if (options.nDPU == 1 &&
        std::find(strategies.begin(), strategies.end(), VPUSplitStrategy::Z_TILING) != strategies.end()) {
        return false;  // WHY? ONly K tiling allowed on one DPU, no parallelism?
    }

    // If the layer require a max tile size in Z then only ZTiling is allowed
    return !requireMaxZTile(layer);
}

/**
 * @brief Get the Tiling Algorithms objects, These are intra tile algos (splitting to DPUWorkloads)
 */
TilingAlgorithmsContainer getTilingAlgorithms(const DPULayer& layer, const SplitOptions& options) {
    TilingAlgorithmsContainer algos;
    // @todo: re-factor this to be more explicit; Like pass 1 , filter out not allowed strategies, step 2 create Tiling
    // algorithms for allowed ones.

    for (auto strategy : options.availableStrategies) {
        switch (strategy) {
        case VPUSplitStrategy::Z_TILING:
            algos.push_back(std::make_unique<ZTiling>(layer, options.maxWorkloads));
            break;
        case VPUSplitStrategy::HW_TILING:
            if (isHWTilingAllowed(layer, options))  // Z Tiling will inhibit HW on 1 dpu
                algos.push_back(std::make_unique<HWTiling>(layer, options.maxWorkloads));
            break;
        case VPUSplitStrategy::H_TILING:
            if (isHWTilingAllowed(layer, options))
                algos.push_back(std::make_unique<HTiling>(layer, options.maxWorkloads));
            break;
        case VPUSplitStrategy::W_TILING:
            if (isHWTilingAllowed(layer, options))
                algos.push_back(std::make_unique<WTiling>(layer, options.maxWorkloads));
            break;
        default:
            continue;
        }
    }
    return algos;
}
}  // namespace VPUNN