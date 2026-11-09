// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/dma_cost_providers/dma_strided_math_cost_provider.h"
#include "vpu/dma_descriptor_transformers.h"
#include "vpu/dma_types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "logger/logger.h"
#include "vpu/dma_cost_providers/dma_theoretical_cost_provider.h"
#include "vpu_cost_model.h"
#include "vpu_dma_cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

/// Test fixture for DMA Strided Math Cost Model for NPU 6.x devices
class TestDMA_Strided_CostModelVPU6x : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

    // Default device for tests - can be changed in one place to affect all tests
    static constexpr VPUDevice default_device = VPUDevice::NPU_6_0;

    /// Convert a DMANNWorkload_NPU40_50 to an equivalent VPUDMADescriptor.
    /// Used to drive DMAStridedMathCostProvider_NI with the same logical transfer
    /// as the DMANNWorkload_NPU40_50 workload passed to DMAStridedMathCostProvider.
    static VPUDMADescriptor toVPUDMADescriptor(const DMANNWorkload_NPU40_50& wl) {
        return DMADescriptorTransformer::fromDMANNWorkload_NPU40_50_to_VPUDMADescriptor(wl);
    }

    /// Convert a MemoryDirection enum into the corresponding (input, output) MemoryLocation pair.
    static std::pair<MemoryLocation, MemoryLocation> directionToLocations(MemoryDirection dir) {
        switch (dir) {
        case MemoryDirection::CMX2CMX:
            return {MemoryLocation::CMX, MemoryLocation::CMX};
        case MemoryDirection::CMX2DDR:
            return {MemoryLocation::CMX, MemoryLocation::DRAM};
        case MemoryDirection::DDR2CMX:
            return {MemoryLocation::DRAM, MemoryLocation::CMX};
        case MemoryDirection::DDR2DDR:
            return {MemoryLocation::DRAM, MemoryLocation::DRAM};
        default:
            return {MemoryLocation::CMX, MemoryLocation::CMX};
        }
    }
};

/// Test basic functionality of DMA Strided Math Model
TEST_F(TestDMA_Strided_CostModelVPU6x, DMAStridedMath_theoretical_basic_Test) {
    auto mkWl_NPU40_50 = [](const VPUDevice dev, int size_bytes = 4864) {
        DMANNWorkload_NPU40_50 wl{dev};
        wl.src_width = size_bytes;
        wl.dst_width = size_bytes;
        wl.num_dim = 0;  // 1D transfer (no extra dimensions)
        wl.transfer_direction = MemoryDirection::CMX2DDR;
        return wl;
    };

    // Two provider instances — one per specialization of BaseDMAStridedCostProvider.
    // Both must produce identical cycle counts for the same logical transfer.
    DMAStridedMathCostProvider_NNIntf dma_cost_provider;       // BaseDMAStridedCostProvider<DMANNWorkload_NPU40_50>
    DMAStridedMathCostProvider_DescIntf dma_cost_provider_ni;  // BaseDMAStridedCostProvider<VPUDMADescriptor>

    {  // valid cases for NPU 5.0 and 6.0
        std::vector<VPUDevice> valid_npu40_50_devices = {
                VPUDevice::NPU_5_0,
                VPUDevice::NPU_5_0_W,
                VPUDevice::NPU_6_0,
        };

        for (const auto d : valid_npu40_50_devices) {
            const DMANNWorkload_NPU40_50 wl = mkWl_NPU40_50(d);
            const VPUDMADescriptor wl_ni = toVPUDMADescriptor(wl);

            std::string cost_source, cost_source_ni;
            auto dma_n = dma_cost_provider.get_cost(wl, &cost_source);
            auto dma_n_ni = dma_cost_provider_ni.get_cost(wl_ni, &cost_source_ni);

            EXPECT_FALSE(Cycles::isErrorCode(dma_n))
                    << "Device:" << static_cast<int>(d) << " Err::" << Cycles::toErrorText(dma_n);
            EXPECT_GT(dma_n, 0) << "Device:" << static_cast<int>(d) << " should return positive cycles";
            EXPECT_EQ(cost_source, "theoretical_strided") << "Cost source should be theoretical_strided";

            EXPECT_FALSE(Cycles::isErrorCode(dma_n_ni))
                    << "NI Device:" << static_cast<int>(d) << " Err::" << Cycles::toErrorText(dma_n_ni);
            EXPECT_EQ(cost_source_ni, "theoretical_strided") << "NI cost source should be theoretical_strided";
            EXPECT_EQ(dma_n, dma_n_ni) << "Both providers must return the same cycles for device:"
                                       << static_cast<int>(d);
        }
    }

    {  // Test with various transfer sizes
        const VPUDevice device = default_device;

        // Small transfer
        {
            auto wl = mkWl_NPU40_50(device, 256);
            auto cycles = dma_cost_provider.get_cost(wl);
            auto cycles_ni = dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl));
            EXPECT_FALSE(Cycles::isErrorCode(cycles)) << "Small transfer failed";
            EXPECT_GT(cycles, 0) << "Small transfer should have positive cycles";
            EXPECT_EQ(cycles, cycles_ni) << "Small transfer: both providers must agree";
        }

        // Medium transfer
        {
            auto wl = mkWl_NPU40_50(device, 4096);
            auto cycles = dma_cost_provider.get_cost(wl);
            auto cycles_ni = dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl));
            EXPECT_FALSE(Cycles::isErrorCode(cycles)) << "Medium transfer failed";
            EXPECT_GT(cycles, 0) << "Medium transfer should have positive cycles";
            EXPECT_EQ(cycles, cycles_ni) << "Medium transfer: both providers must agree";
        }

        // Large transfer
        {
            auto wl = mkWl_NPU40_50(device, 65536);
            auto cycles = dma_cost_provider.get_cost(wl);
            auto cycles_ni = dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl));
            EXPECT_FALSE(Cycles::isErrorCode(cycles)) << "Large transfer failed";
            EXPECT_GT(cycles, 0) << "Large transfer should have positive cycles";
            EXPECT_EQ(cycles, cycles_ni) << "Large transfer: both providers must agree";
        }
    }

    {                                                 // Test strided transfers (with gaps) on NPU 6.0
        const VPUDevice device = VPUDevice::NPU_6_0;  // NPU 6.0 has stride penalty enabled

        DMANNWorkload_NPU40_50 wl_strided{device};
        wl_strided.src_width = 64;
        wl_strided.dst_width = 64;
        wl_strided.num_dim = 1;
        wl_strided.e_dim[0].src_dim_size = 15;
        wl_strided.e_dim[0].dst_dim_size = 15;
        wl_strided.e_dim[0].src_stride = 128;  // Gap: stride > width (strided access)
        wl_strided.e_dim[0].dst_stride = 128;
        wl_strided.transfer_direction = MemoryDirection::CMX2DDR;

        auto cycles_strided = dma_cost_provider.get_cost(wl_strided);
        auto cycles_strided_ni = dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl_strided));
        EXPECT_FALSE(Cycles::isErrorCode(cycles_strided)) << "Strided transfer failed";
        EXPECT_GT(cycles_strided, 0) << "Strided transfer should have positive cycles";
        EXPECT_EQ(cycles_strided, cycles_strided_ni) << "Strided: both providers must agree";

        DMANNWorkload_NPU40_50 wl_contiguous{device};
        wl_contiguous.src_width = 64;
        wl_contiguous.dst_width = 64;
        wl_contiguous.num_dim = 1;
        wl_contiguous.e_dim[0].src_dim_size = 15;
        wl_contiguous.e_dim[0].dst_dim_size = 15;
        wl_contiguous.e_dim[0].src_stride = 64;  // Contiguous
        wl_contiguous.e_dim[0].dst_stride = 64;
        wl_contiguous.transfer_direction = MemoryDirection::CMX2DDR;

        auto cycles_contiguous = dma_cost_provider.get_cost(wl_contiguous);
        auto cycles_contiguous_ni = dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl_contiguous));
        EXPECT_FALSE(Cycles::isErrorCode(cycles_contiguous)) << "Contiguous transfer failed";
        EXPECT_EQ(cycles_contiguous, cycles_contiguous_ni) << "Contiguous: both providers must agree";

        EXPECT_GE(cycles_strided, cycles_contiguous) << "Strided transfer should be >= contiguous transfer on NPU 6.0";
    }

    {  // Test that stride penalty is not applied on default device
        const VPUDevice device = default_device;

        DMANNWorkload_NPU40_50 wl_strided{device};
        wl_strided.src_width = 64;
        wl_strided.dst_width = 64;
        wl_strided.num_dim = 1;
        wl_strided.e_dim[0].src_dim_size = 15;
        wl_strided.e_dim[0].dst_dim_size = 15;
        wl_strided.e_dim[0].src_stride = 128;  // Gap: stride > width
        wl_strided.e_dim[0].dst_stride = 128;

        auto cycles = dma_cost_provider.get_cost(wl_strided);
        auto cycles_ni = dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl_strided));
        EXPECT_FALSE(Cycles::isErrorCode(cycles)) << "Strided transfer failed on default device";
        EXPECT_GT(cycles, 0) << "Strided transfer should have positive cycles on default device";
        EXPECT_EQ(cycles, cycles_ni) << "No-penalty device: both providers must agree";
    }

    {  // invalid cases - invalid devices should return error or fallback gracefully
        std::vector<VPUDevice> invalid_dev{(VPUDevice)20, (VPUDevice)30};
        for (const auto d : invalid_dev) {
            const DMANNWorkload_NPU40_50 wl = mkWl_NPU40_50(d);
            EXPECT_NO_THROW(dma_cost_provider.get_cost(wl));
            EXPECT_NO_THROW(dma_cost_provider_ni.get_cost(toVPUDMADescriptor(wl)));
        }
    }
}

/// Test continuous DMA operations across different sizes and all direction combinations.
/// Compares three cost models: MATH_DMA (Strided Math), DMANN NN (NPU 5.0 proxy), Theoretical.
TEST_F(TestDMA_Strided_CostModelVPU6x, DISABLED_SWEEP_MATH_Continuous_DMA_NPU6) {
    class TestCase {
    public:
        const DMANNWorkload_NPU40_50 t_in;  // wl
        int t_bytes;                        // for reporting
        MemoryDirection t_dir;              // for reporting
    };

    // const VPUDevice device = VPUDevice::NPU_6_0;

    // Helper to create continuous (non-strided) workload
    auto mkWl = [](int bytes, MemoryDirection dir) {
        DMANNWorkload_NPU40_50 wl{VPUDevice::NPU_6_0};
        wl.src_width = bytes;
        wl.dst_width = bytes;
        wl.num_dim = 0;  // 1D transfer (continuous, no extra dimensions)
        wl.transfer_direction = dir;
        return wl;
    };

    // Transfer sizes to sweep - similar to SweepGT_DMATime_40
    const std::vector<int> dmaTxSizes{1,      2,      3,      4,       6,       8,       12,     16,     24,
                                      32,     48,     64,     96,      128,     192,     256,    384,    512,
                                      768,    1024,   1536,   2048,    3072,    4096,    6144,   8192,   12288,
                                      16384,  24576,  32768,  49152,   65536,   98304,   131072, 196608, 262144,
                                      393216, 524288, 786432, 1048576, 1572864, 2097152, 3145728};

    // All direction combinations
    const std::vector<MemoryDirection> directions{MemoryDirection::CMX2CMX, MemoryDirection::CMX2DDR,
                                                  MemoryDirection::DDR2CMX, MemoryDirection::DDR2DDR};

    // Build test cases for all combinations
    std::vector<TestCase> tc;
    tc.reserve(dmaTxSizes.size() * directions.size());

    for (const auto bytes : dmaTxSizes) {
        for (const auto& dir : directions) {
            tc.push_back({mkWl(bytes, dir), bytes, dir});
        }
    }

    // Initialize all cost providers
    DMAStridedMathCostProvider_DescIntf dma_strided_math;
    DMATheoreticalCostProvider_PTL dma_theoretical{HWCharacteristicsSuperSets::get_mainConfigurationRef()};
    // DMANN NN model — NPU 5.0 model used as proxy (no NPU 6.0 model available yet)
    DMACostModel<DMANNWorkload_NPU40_50> dma_nn(NPU_DMA_5_0_MODEL_PATH);
    const bool dma_nn_ok = dma_nn.nn_initialized();

    // Helper to convert DMANNWorkload_NPU40_50 to DMAWorkload for the theoretical provider
    auto to_dma_workload = [](const DMANNWorkload_NPU40_50& wl_nn) {
        int total_bytes = wl_nn.src_width;  // continuous 1D transfer
        const auto [in_loc, out_loc] = directionToLocations(wl_nn.transfer_direction);
        return DMAWorkload{wl_nn.device,
                           {VPUTensor(total_bytes, 1, 1, 1, DataType::UINT8)},
                           {VPUTensor(total_bytes, 1, 1, 1, DataType::UINT8)},
                           in_loc,
                           out_loc,
                           1};
    };

    // Helper: print a single cycles value right-aligned, or the error text
    auto print_cycles = [](CyclesInterfaceType c, int w) {
        if (Cycles::isErrorCode(c))
            utest_log::debug() << std::setw(w) << Cycles::toErrorText(c);
        else
            utest_log::debug() << std::setw(w) << c;
    };

    // Helper: direction name — uses the registered EnumMap from dma_types.h
    auto get_dir_name = [](MemoryDirection dir) -> const std::string& {
        return MemoryDirection_ToText.at(static_cast<int>(dir));
    };

    // -------------------------------------------------------------------------
    // Print header
    // -------------------------------------------------------------------------
    utest_log::debug() << "\n==================== DMA COST MODEL COMPARISON for NPU 6.0 ====================\n";
    utest_log::debug() << "Testing continuous (non-strided) DMA transfers\n";
    utest_log::debug() << "Comparing: MATH_DMA (Strided Math), DMANN NN (NPU5.0 proxy), Theoretical\n";
    utest_log::debug() << "DMANN NN model initialized: " << (dma_nn_ok ? "YES" : "NO (fallback to theoretical)") << "\n";
    utest_log::debug() << "Total test cases: " << tc.size() << "\n";

    // -------------------------------------------------------------------------
    // Detailed table — one block per direction, all sizes inside
    // Columns: BYTES | MATH_DMA | DMANN_NN | Theoretical | D%(M-NN) | D%(M-T)
    //   D%(M-NN) = (MATH - DMANN_NN)    / DMANN_NN    * 100  (NN is 2nd column)
    //   D%(M-T)  = (MATH - Theoretical) / Theoretical * 100  (T  is 3rd column)
    // -------------------------------------------------------------------------
    for (const auto& dir : directions) {
        utest_log::debug() << "\n--- Direction: " << get_dir_name(dir) << " ---\n";
        utest_log::debug() << std::setw(10) << "BYTES"
                          << ",  " << std::setw(12) << "MATH_DMA"
                          << ",  " << std::setw(12) << "DMANN_NN"
                          << ",  " << std::setw(12) << "Theoretical"
                          << ",  " << std::setw(12) << "D%(M-NN)"
                          << ",  " << std::setw(12) << "D%(M-T)"
                          << "\n";
        utest_log::debug() << "------------------------------------------------------------------------"
                             "--------------------\n";

        for (const auto& test : tc) {
            if (test.t_dir != dir)
                continue;

            const DMANNWorkload_NPU40_50& wl_in = test.t_in;

            auto cycles_math = dma_strided_math.get_cost(toVPUDMADescriptor(wl_in));
            auto cycles_nn = dma_nn.computeCycles(wl_in);
            auto cycles_t = dma_theoretical.DMATheoreticalCyclesPTL_ON(to_dma_workload(wl_in));

            // Validate
            EXPECT_FALSE(Cycles::isErrorCode(cycles_math))
                    << "MATH_DMA failed for bytes=" << test.t_bytes << " dir=" << static_cast<int>(test.t_dir);
            EXPECT_FALSE(Cycles::isErrorCode(cycles_t))
                    << "Theoretical failed for bytes=" << test.t_bytes << " dir=" << static_cast<int>(test.t_dir);
            if (!Cycles::isErrorCode(cycles_math)) {
                EXPECT_GT(cycles_math, 0) << "MATH_DMA > 0 for bytes=" << test.t_bytes;
            }
            if (!Cycles::isErrorCode(cycles_t)) {
                EXPECT_GT(cycles_t, 0) << "Theoretical > 0 for bytes=" << test.t_bytes;
            }

            // Deltas in the same order as data columns: M / NN / T
            const float dmnn = (!Cycles::isErrorCode(cycles_nn) && cycles_nn > 0)
                                       ? (((float)cycles_math - (float)cycles_nn) / (float)cycles_nn) * 100.0f
                                       : 0.0f;
            const float dmt =
                    (cycles_t > 0) ? (((float)cycles_math - (float)cycles_t) / (float)cycles_t) * 100.0f : 0.0f;

            utest_log::debug() << std::setw(10) << test.t_bytes << ",  ";
            print_cycles(cycles_math, 12);
            utest_log::debug() << ",  ";
            print_cycles(cycles_nn, 12);
            utest_log::debug() << ",  ";
            print_cycles(cycles_t, 12);
            utest_log::debug() << ",  ";
            utest_log::debug() << std::setw(11) << std::setprecision(2) << std::showpos << std::fixed << dmnn << "%,  "
                              << std::setw(11) << dmt << "%" << std::noshowpos << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // Summary table — one row per size, one column group per direction
    // Each cell: M / NN / T  (dMT% / dMNN%)
    // Fixed widths: M=7, NN=7, T=7, delta group=20  => cell total ~44 chars
    // -------------------------------------------------------------------------
    utest_log::debug() << "\n========================================================================\n";
    utest_log::debug() << "--- Summary Table (M=MATH / NN=DMANN_NN / T=Theoretical) ---\n";

    // Column cell width: M(7) + "/" + NN(7) + "/" + T(7) + "   " + delta(20) = 44 chars + ",  " separator
    constexpr int wM = 7;
    constexpr int wNN = 7;
    constexpr int wT = 7;
    constexpr int wD = 20;                                 // width for the delta group "(+xxx.x%/+xxx.x%)"
    constexpr int wCell = wM + 1 + wNN + 1 + wT + 3 + wD;  // 46

    auto col_header = [&](const char* dir_name) {
        std::ostringstream h;
        h << dir_name << "  M/NN/T   (dMNN%/dMT%)";  // delta order matches column order
        utest_log::debug() << std::setw(wCell) << std::left << h.str() << std::right;
    };

    utest_log::debug() << std::setw(10) << "BYTES" << ",  ";
    col_header("CMX2CMX");
    utest_log::debug() << ",  ";
    col_header("CMX2DDR");
    utest_log::debug() << ",  ";
    col_header("DDR2CMX");
    utest_log::debug() << ",  ";
    col_header("DDR2DDR");
    utest_log::debug() << "\n";

    std::string sep(10 + 2 + 4 * (wCell + 2), '-');
    utest_log::debug() << sep << "\n";

    for (const auto bytes : dmaTxSizes) {
        utest_log::debug() << std::setw(10) << bytes << ",  ";

        for (const auto& dir : directions) {
            auto wl = mkWl(bytes, dir);
            auto cycles_m = dma_strided_math.get_cost(toVPUDMADescriptor(wl));
            auto cycles_nn = dma_nn.computeCycles(wl);
            auto cycles_t = dma_theoretical.DMATheoreticalCyclesPTL_ON(to_dma_workload(wl));

            if (Cycles::isErrorCode(cycles_m) || Cycles::isErrorCode(cycles_t)) {
                utest_log::debug() << std::setw(wCell) << "ERROR";
            } else {
                float dmt = (cycles_t > 0) ? (((float)cycles_m - (float)cycles_t) / (float)cycles_t) * 100.0f : 0.0f;
                float dmnn = (!Cycles::isErrorCode(cycles_nn) && cycles_nn > 0)
                                     ? (((float)cycles_m - (float)cycles_nn) / (float)cycles_nn) * 100.0f
                                     : 0.0f;

                // Build delta string: "(+xxx%/+xxx%)" — no spaces inside, shorter
                std::ostringstream delta_ss;
                delta_ss << "(" << std::showpos << std::fixed << std::setprecision(0) << dmnn << "%/" << dmt << "%)";

                utest_log::debug() << std::noshowpos << std::setw(wM) << cycles_m << "/" << std::setw(wNN)
                                  << (Cycles::isErrorCode(cycles_nn) ? 0 : (int)cycles_nn) << "/" << std::setw(wT)
                                  << cycles_t << " "  // single space before parenthesis
                                  << std::setw(wD) << delta_ss.str();
            }
            utest_log::debug() << ",  ";
        }
        utest_log::debug() << "\n";
    }

    utest_log::debug() << "\n--- Legend ---\n";
    utest_log::debug() << "M     = MATH_DMA cycles  (1st column)\n";
    utest_log::debug() << "NN    = DMANN NN cycles  (2nd column, NPU 5.0 model used as proxy for NPU 6.0)\n";
    utest_log::debug() << "T     = Theoretical cycles  (3rd column)\n";
    utest_log::debug() << "Deltas follow the same left-to-right order as the data columns (M / NN / T):\n";
    utest_log::debug() << "dMNN% = (M - NN) / NN * 100  (signed: + means MATH slower than DMANN NN)\n";
    utest_log::debug() << "dMT%  = (M - T)  / T  * 100  (signed: + means MATH slower than Theoretical)\n";
    utest_log::debug() << "\n==================== END OF SWEEP ====================\n";
}

/// Test strided DMA operations for a fixed ~100k transfer size across all direction combinations.
/// Sweeps the innermost contiguous chunk size (powers of two, 1..tensorSize/2).
/// Each workload is 2D: innermost width = chunk, stride = chunk+64 (fixed gap), outer dim computed accordingly.
/// Only the summary table is printed.
TEST_F(TestDMA_Strided_CostModelVPU6x, DISABLED_SWEEP_MATH_Strided_100k_DMA_NPU6) {
    constexpr int tensorSize = 98304;  // ~100 kB logical transfer size

    // Generate contiguous chunk sizes: powers of two from 1 up to tensorSize/2
    // For chunks >= 1024, also insert the midpoint between each power of two and the next
    // (i.e. c * 3/2), giving one extra sample per octave in the larger-chunk range.
    // All midpoints evenly divide tensorSize because 98304 = 3 * 2^15.
    // The final entry is chunk = tensorSize: dim_size = 0, one single chunk, fully contiguous
    // (stride is irrelevant when outer dim_size == 0).
    std::vector<int> chunkSizes;
    for (int c = 1; c <= tensorSize / 2; c *= 2) {
        chunkSizes.push_back(c);
        if (c >= 256 && c < 1024) {
            // fraction-based intermediates (work for small powers of two)
            const int q4 = c / 4;
            for (const int mid : {c + q4, c + 2 * q4, c + 3 * q4}) {
                if (mid < tensorSize) {
                    chunkSizes.push_back(mid);
                }
            }
        } else if (c >= 1024) {
            // offset-based intermediates: this pattern creates points that are non-multiples of 256
            // and one exact multiple (3/2·c). With side-specific thresholds, this is only a DDR-side
            // divisibility reference, not a universal no-penalty indicator.
            const int half = c / 2;
            for (const int mid : {c + 128, c + half, c + half + 128}) {
                if (mid < tensorSize) {
                    chunkSizes.push_back(mid);
                }
            }
        }
    }
    chunkSizes.push_back(tensorSize);  // limit case: one contiguous chunk = full tensor, outer dim_size = 0

    // All direction combinations
    const std::vector<MemoryDirection> directions{MemoryDirection::CMX2CMX, MemoryDirection::CMX2DDR,
                                                  MemoryDirection::DDR2CMX, MemoryDirection::DDR2DDR};

    // Helper: build a 2D strided workload
    //   - innermost width = chunk bytes (contiguous)
    //   - stride = chunk + 64  (fixed 64-byte gap between chunks)
    //   - outer dim_size (0-based) = (tensorSize / chunk) - 1
    //   => total logical bytes read/written = chunk * (dim_size+1) = tensorSize
    auto mkStridedWl = [&](int chunk, MemoryDirection dir) {
        constexpr int gap = 64;
        const int localTensorSz = tensorSize;
        DMANNWorkload_NPU40_50 wl{VPUDevice::NPU_6_0};
        wl.src_width = chunk;
        wl.dst_width = chunk;
        wl.num_dim = 1;
        wl.e_dim[0].src_stride = chunk + gap;
        wl.e_dim[0].dst_stride = chunk + gap;
        // Floor division: dim_size = floor(tensorSize / chunk) - 1
        // For exact divisors this equals (tensorSize/chunk)-1; for non-exact divisors
        // the actual transferred bytes = chunk * (dim_size+1) < tensorSize but is close.
        wl.e_dim[0].src_dim_size = (localTensorSz / chunk) - 1;
        wl.e_dim[0].dst_dim_size = (localTensorSz / chunk) - 1;
        wl.transfer_direction = dir;
        return wl;
    };

    // Helper: convert to DMAWorkload for the theoretical provider.
    // Uses getReadBytes() so the logical size (tensorSize) is preserved regardless of strides.
    auto to_dma_workload_strided = [](const DMANNWorkload_NPU40_50& wl_nn) {
        const int total_bytes = wl_nn.getReadBytes();
        const auto [in_loc, out_loc] = directionToLocations(wl_nn.transfer_direction);
        return DMAWorkload{wl_nn.device,
                           {VPUTensor(total_bytes, 1, 1, 1, DataType::UINT8)},
                           {VPUTensor(total_bytes, 1, 1, 1, DataType::UINT8)},
                           in_loc,
                           out_loc,
                           1};
    };

    // Initialize all cost providers
    DMAStridedMathCostProvider_DescIntf dma_strided_math;
    DMATheoreticalCostProvider_PTL dma_theoretical{HWCharacteristicsSuperSets::get_mainConfigurationRef()};
    DMACostModel<DMANNWorkload_NPU40_50> dma_nn(NPU_DMA_5_0_MODEL_PATH);
    const bool dma_nn_ok = dma_nn.nn_initialized();

    // Helper: direction name
    auto get_dir_name = [](MemoryDirection dir) -> const std::string& {
        return MemoryDirection_ToText.at(static_cast<int>(dir));
    };

    // -------------------------------------------------------------------------
    // Print header
    // -------------------------------------------------------------------------
    utest_log::debug() << "\n==================== STRIDED DMA SWEEP ~100k for NPU 6.0 ====================\n";
    utest_log::debug() << "Logical transfer size: " << tensorSize << " bytes\n";
    utest_log::debug() << "Workload: 2D strided : innermost width = CHUNK, stride = CHUNK+64, outer dim fills tensor\n";
    utest_log::debug() << "Chunk sizes (powers of two, 1 .. tensorSize/2): " << chunkSizes.size() << " steps\n";
    utest_log::debug() << "DMANN NN model initialized: " << (dma_nn_ok ? "YES" : "NO (fallback to theoretical)") << "\n";
    utest_log::debug() << "Comparing: M=MATH_DMA / NN=DMANN_NN / T=Theoretical\n";

    // -------------------------------------------------------------------------
    // Summary table — one row per chunk size, one column group per direction
    // Each cell: M / NN / T   (dMNN% / dMT%)
    // -------------------------------------------------------------------------
    constexpr int wM = 7;                                  // max ~1001985 = 7 digits
    constexpr int wNN = 6;                                 // max ~364070  = 6 digits
    constexpr int wT = 5;                                  // max ~3274    = 4 digits
    constexpr int wD = 17;                                 // "(+36402%/+36402%)" = 17 chars worst case
    constexpr int wCell = wM + 1 + wNN + 1 + wT + 1 + wD;  // 38

    auto col_header_strided = [&](MemoryDirection dir) {
        std::ostringstream h;
        h << get_dir_name(dir) << " M/NN/T (dMNN%/dMT%)";
        utest_log::debug() << std::setw(wCell) << std::left << h.str() << std::right;
    };

    // Header row
    utest_log::debug() << "\n";
    utest_log::debug() << std::setw(8) << "CHUNK*" << std::setw(6) << "DIM" << std::setw(8) << "STRIDE" << std::setw(8)
                      << "BYTES"
                      << ",  ";
    for (const auto& dir : directions) {
        col_header_strided(dir);
        utest_log::debug() << ",  ";
    }
    utest_log::debug() << "\n";

    std::string sep(8 + 6 + 7 + 8 + 2 + 4 * (wCell + 2), '-');
    utest_log::debug() << sep << "\n";

    // Data rows
    for (const int chunk : chunkSizes) {
        const int dimSize = (tensorSize / chunk) - 1;  // 0-based, floor division
        const int stride = chunk + 64;
        const int actualBytes = chunk * (dimSize + 1);  // may be < tensorSize for non-divisors

        utest_log::debug() << std::setw(7) << chunk << (chunk % 256 != 0 ? "*" : " ") << std::setw(6) << (dimSize + 1)
                          << std::setw(7) << stride << std::setw(8) << actualBytes << ",  ";

        for (const auto& dir : directions) {
            auto wl_s = mkStridedWl(chunk, dir);
            auto cycles_m = dma_strided_math.get_cost(toVPUDMADescriptor(wl_s));
            auto cycles_nn = dma_nn.computeCycles(wl_s);
            auto cycles_t = dma_theoretical.DMATheoreticalCyclesPTL_ON(to_dma_workload_strided(wl_s));

            if (Cycles::isErrorCode(cycles_m) || Cycles::isErrorCode(cycles_t)) {
                utest_log::debug() << std::setw(wCell) << "ERROR";
            } else {
                float dmt = (cycles_t > 0) ? (((float)cycles_m - (float)cycles_t) / (float)cycles_t) * 100.0f : 0.0f;
                float dmnn = (!Cycles::isErrorCode(cycles_nn) && cycles_nn > 0)
                                     ? (((float)cycles_m - (float)cycles_nn) / (float)cycles_nn) * 100.0f
                                     : 0.0f;

                // Build delta string: "(+xxx%/+xxx%)" — no spaces inside, shorter
                std::ostringstream delta_ss;
                delta_ss << "(" << std::showpos << std::fixed << std::setprecision(0) << dmnn << "%/" << dmt << "%)";

                utest_log::debug() << std::noshowpos << std::setw(wM) << cycles_m << "/" << std::setw(wNN)
                                  << (Cycles::isErrorCode(cycles_nn) ? 0 : (int)cycles_nn) << "/" << std::setw(wT)
                                  << cycles_t << " "  // single space before parenthesis
                                  << std::setw(wD) << delta_ss.str();
            }
            utest_log::debug() << ",  ";
        }
        utest_log::debug() << "\n";
    }

    utest_log::debug() << "\n--- Legend ---\n";
    utest_log::debug() << "CHUNK  = innermost contiguous bytes (src_width = dst_width);  * = not a multiple of 256 "
                         "(DDR-side divisibility marker)\n";
    utest_log::debug() << "DIM    = number of outer repetitions  floor(tensorSize / CHUNK)\n";
    utest_log::debug() << "BYTES  = actual logical bytes transferred = CHUNK * DIM  (may be < " << tensorSize
                      << " for non-exact divisors)\n";
    utest_log::debug() << "STRIDE = CHUNK + 64  (64-byte gap; irrelevant when DIM=1)\n";
    utest_log::debug() << "Intermediate rows per octave:\n";
    utest_log::debug() << "  c < 1024  : 5/4·c*, 3/2·c, 7/4·c*  (fraction-based; *=non-multiple of 256)\n";
    utest_log::debug() << "  c >= 1024 : c+128*, 3/2·c, 3/2·c+128*  (offset-based: ±128 never multiples of 256)\n";
    utest_log::debug() << "              3/2·c is a multiple of 256 for all c>=1024 (DDR-side reference row)\n";
    utest_log::debug() << "  DIM=1 (last row): single chunk = full tensor, stride unused (contiguous limit case)\n";

    utest_log::debug() << "\n==================== END OF STRIDED SWEEP ====================\n";
}

/// Test a 2D strided workload (inspired by the DDR2DDR_209436 reference case) for NPU 6.0.
/// Source is CONTIGUOUS 1D: src_width = total_logical_bytes, src_dim_size=0 (single read, no outer repetition).
/// Destination is STRIDED:  dst_width = 2B (1 FLOAT16), dst_dim_size = outer_dim_size, dst_stride = 6B (4-byte gap).
/// Total logical bytes = 2 * 78336*2 = 313344 bytes (approx. 78k FLOAT16 elements).
/// Also compares against a fully contiguous transfer of the same logical size.
/// Compares DMANN NN (NPU5.0 proxy), MATH_DMA (Strided Math), Theoretical PTL, and Theoretical LNL Legacy across all
/// directions.
TEST_F(TestDMA_Strided_CostModelVPU6x, DISABLED_Strided2D_78k_FLOAT16_AllDirections_NPU6) {
    // Workload parameters (mirrors DDR2DDR_209436 but for NPU 6.0)
    constexpr int src_width_bytes = 2;             // 1 FLOAT16 element = 2 bytes
    constexpr int outer_dim_size = 78336 * 2 - 1;  // 0-based count (156671 repetitions)
    constexpr int dst_stride_bytes = 6;            // destination stride = element_size + 4-byte gap
    const int total_logical_bytes = src_width_bytes * (outer_dim_size + 1);  // = 313344 bytes

    // All direction combinations
    const std::vector<MemoryDirection> directions{MemoryDirection::CMX2CMX, MemoryDirection::CMX2DDR,
                                                  MemoryDirection::DDR2CMX, MemoryDirection::DDR2DDR};

    // Helper: direction name
    auto get_dir_name = [](MemoryDirection dir) -> const std::string& {
        return MemoryDirection_ToText.at(static_cast<int>(dir));
    };

    // Helper: build the 2D strided workload for a given direction.
    // Source is CONTIGUOUS 1D: src_width = all bytes, src_dim_size = 0 (no outer repetition on source).
    // Destination is STRIDED:  dst_width = 2B per element, dst_dim_size = outer_dim_size, dst_stride = 6B.
    auto mkStridedWl = [&](MemoryDirection dir) {
        DMANNWorkload_NPU40_50 wl{VPUDevice::NPU_6_0};
        wl.src_width = total_logical_bytes;         // entire source read as one contiguous chunk
        wl.dst_width = src_width_bytes;             // 2B per destination element
        wl.num_dim = 1;                             // one extra dimension (applies to dst side)
        wl.e_dim[0].src_dim_size = 0;               // source: no outer repetition (single 1D read)
        wl.e_dim[0].src_stride = 0;                 // source: no gap
        wl.e_dim[0].dst_dim_size = outer_dim_size;  // destination: outer_dim_size+1 repetitions
        wl.e_dim[0].dst_stride = dst_stride_bytes;  // destination: 6B stride (4-byte gap between writes)
        wl.transfer_direction = dir;
        return wl;
    };

    // Helper: build the contiguous (1D) workload with the same logical size
    auto mkContiguousWl = [&](MemoryDirection dir) {
        DMANNWorkload_NPU40_50 wl{VPUDevice::NPU_6_0};
        wl.src_width = total_logical_bytes;
        wl.dst_width = total_logical_bytes;
        wl.num_dim = 0;  // 1D - fully contiguous
        wl.transfer_direction = dir;
        return wl;
    };

    // Helper: convert DMANNWorkload_NPU40_50 to DMAWorkload for the theoretical providers.
    // Uses getReadBytes() to get the actual logical transfer size.
    auto to_dma_workload = [](const DMANNWorkload_NPU40_50& wl_nn) {
        const int total_bytes = wl_nn.getReadBytes();
        const auto [in_loc, out_loc] = directionToLocations(wl_nn.transfer_direction);
        return DMAWorkload{wl_nn.device,
                           {VPUTensor(total_bytes, 1, 1, 1, DataType::UINT8)},
                           {VPUTensor(total_bytes, 1, 1, 1, DataType::UINT8)},
                           in_loc,
                           out_loc,
                           1};
    };

    // Initialize cost providers
    DMAStridedMathCostProvider_DescIntf dma_strided_math;
    // Use PTL directly: DMATheoreticalCostProvider routes NPU6 workloads through DMAStridedMathCostProvider,
    // which would duplicate the MATH_DMA column. PTL gives an independent theoretical baseline.
    DMATheoreticalCostProvider_PTL dma_theoretical_ptl{HWCharacteristicsSuperSets::get_mainConfigurationRef()};
    // Legacy provider for additional reference
    DMATheoreticalCostProvider_LNL_Legacy dma_theoretical_lnl;
    DMACostModel<DMANNWorkload_NPU40_50> dma_nn(NPU_DMA_5_0_MODEL_PATH);
    const bool dma_nn_ok = dma_nn.nn_initialized();

    // Helper: print cycles or error text right-aligned
    auto print_c = [](CyclesInterfaceType c, int w) {
        if (Cycles::isErrorCode(c))
            utest_log::debug() << std::setw(w) << Cycles::toErrorText(c);
        else
            utest_log::debug() << std::setw(w) << c;
    };

    // -------------------------------------------------------------------------
    // Print header
    // -------------------------------------------------------------------------
    utest_log::debug() << "\n==================== Strided 2D ~78k FLOAT16 for NPU 6.0 ====================\n";
    utest_log::debug() << "Strided workload : num_dim=1\n";
    utest_log::debug() << "  SRC: src_width=" << total_logical_bytes << "B (all bytes, 1D contiguous read),"
                      << " src_dim_size=0, src_stride=0\n";
    utest_log::debug() << "  DST: dst_width=" << src_width_bytes << "B (1xFLOAT16 per write),"
                      << " dst_dim_size=" << outer_dim_size << " (0-based),"
                      << " dst_stride=" << dst_stride_bytes << "B (" << (dst_stride_bytes - src_width_bytes)
                      << "-byte gap)\n";
    utest_log::debug() << "Contiguous workload: src_width=" << total_logical_bytes << " bytes (1D, no stride)\n";
    utest_log::debug() << "Total logical bytes: " << total_logical_bytes << "  (same for both workloads)\n";
    utest_log::debug() << "DMANN NN model initialized: " << (dma_nn_ok ? "YES" : "NO (fallback to theoretical)") << "\n";
    std::cout
            << "Comparing: MATH_DMA (Strided Math), DMANN_NN (NPU5.0 proxy), Theoretical PTL, Theoretical LNL Legacy\n";

    // -------------------------------------------------------------------------
    // Table — two blocks: all STRIDED rows, then all CONTIGUOUS rows
    // Columns: DIRECTION | MATH_DMA | DMANN_NN | Th_PTL | Th_LNL | D%(M-NN) | D%(M-PTL) | D%(M-LNL)
    // -------------------------------------------------------------------------
    constexpr int wDir = 12;
    constexpr int wCyc = 12;
    constexpr int wDelta = 12;

    auto print_row = [&](const std::string& dir_label, const DMANNWorkload_NPU40_50& wl) {
        auto cycles_math = dma_strided_math.get_cost(toVPUDMADescriptor(wl));
        auto cycles_nn = dma_nn.computeCycles(wl);
        const auto dma_wl = to_dma_workload(wl);
        auto cycles_ptl = dma_theoretical_ptl.DMATheoreticalCyclesPTL_ON(dma_wl);
        auto cycles_lnl = dma_theoretical_lnl.DMATheoreticalCyclesLegacyLNL(dma_wl);

        // Validate
        EXPECT_FALSE(Cycles::isErrorCode(cycles_math)) << "MATH_DMA failed for dir=" << dir_label;
        EXPECT_FALSE(Cycles::isErrorCode(cycles_ptl)) << "Theoretical PTL failed for dir=" << dir_label;
        EXPECT_FALSE(Cycles::isErrorCode(cycles_lnl)) << "Theoretical LNL failed for dir=" << dir_label;
        if (!Cycles::isErrorCode(cycles_math)) {
            EXPECT_GT(cycles_math, 0) << "MATH_DMA > 0 for dir=" << dir_label;
        }
        if (!Cycles::isErrorCode(cycles_ptl)) {
            EXPECT_GT(cycles_ptl, 0) << "Th_PTL > 0 for dir=" << dir_label;
        }
        if (!Cycles::isErrorCode(cycles_lnl)) {
            EXPECT_GT(cycles_lnl, 0) << "Th_LNL > 0 for dir=" << dir_label;
        }

        const float dmnn = (!Cycles::isErrorCode(cycles_nn) && cycles_nn > 0)
                                   ? (((float)cycles_math - (float)cycles_nn) / (float)cycles_nn) * 100.0f
                                   : 0.0f;
        const float dmt_ptl = (!Cycles::isErrorCode(cycles_ptl) && cycles_ptl > 0)
                                      ? (((float)cycles_math - (float)cycles_ptl) / (float)cycles_ptl) * 100.0f
                                      : 0.0f;
        const float dmt_lnl = (!Cycles::isErrorCode(cycles_lnl) && cycles_lnl > 0)
                                      ? (((float)cycles_math - (float)cycles_lnl) / (float)cycles_lnl) * 100.0f
                                      : 0.0f;

        utest_log::debug() << std::setw(wDir) << dir_label << ",  ";
        print_c(cycles_math, wCyc);
        utest_log::debug() << ",  ";
        print_c(cycles_nn, wCyc);
        utest_log::debug() << ",  ";
        print_c(cycles_ptl, wCyc);
        utest_log::debug() << ",  ";
        print_c(cycles_lnl, wCyc);
        utest_log::debug() << ",  ";
        utest_log::debug() << std::setw(wDelta - 1) << std::setprecision(2) << std::showpos << std::fixed << dmnn
                          << "%,  " << std::setw(wDelta - 1) << dmt_ptl << "%,  " << std::setw(wDelta - 1) << dmt_lnl
                          << "%" << std::noshowpos << "\n";
    };

    utest_log::debug() << "\n";
    utest_log::debug() << std::setw(wDir) << "DIRECTION"
                      << ",  " << std::setw(wCyc) << "MATH_DMA"
                      << ",  " << std::setw(wCyc) << "DMANN_NN"
                      << ",  " << std::setw(wCyc) << "Th_PTL"
                      << ",  " << std::setw(wCyc) << "Th_LNL"
                      << ",  " << std::setw(wDelta - 1) << "D%(M-NN)"
                      << ",  " << std::setw(wDelta - 1) << "D%(M-PTL)"
                      << ",  " << std::setw(wDelta - 1) << "D%(M-LNL)"
                      << "\n";
    const std::string row_sep(wDir + 2 + 4 * (wCyc + 2) + 3 * (wDelta + 2), '-');
    utest_log::debug() << row_sep << "\n";

    // First block: all STRIDED rows
    utest_log::debug() << std::setw(wDir) << "--- STRIDED ---" << "\n";
    utest_log::debug() << row_sep << "\n";
    for (const auto& dir : directions) {
        print_row(get_dir_name(dir), mkStridedWl(dir));
    }

    // Second block: all CONTIGUOUS rows
    utest_log::debug() << row_sep << "\n";
    utest_log::debug() << std::setw(wDir) << "--- CONTIGUOUS ---" << "\n";
    utest_log::debug() << row_sep << "\n";
    for (const auto& dir : directions) {
        print_row(get_dir_name(dir), mkContiguousWl(dir));
    }
    utest_log::debug() << row_sep << "\n";

    utest_log::debug() << "\n--- Legend ---\n";
    utest_log::debug() << "STRIDED    : num_dim=1\n";
    utest_log::debug() << "  SRC: src_width=" << total_logical_bytes
                      << "B (all bytes, 1D contiguous), src_dim_size=0, src_stride=0\n";
    utest_log::debug() << "  DST: dst_width=" << src_width_bytes << "B, dst_dim_size=" << outer_dim_size
                      << " (0-based), dst_stride=" << dst_stride_bytes << "B (" << (dst_stride_bytes - src_width_bytes)
                      << "-byte gap)\n";
    utest_log::debug() << "CONTIGUOUS : src_width=" << total_logical_bytes << "B, num_dim=0 (1D, no stride)\n";
    utest_log::debug() << "Both workloads transfer " << total_logical_bytes << " logical bytes\n";
    utest_log::debug() << "MATH_DMA = DMAStridedMathCostProvider_DescIntf cycles\n";
    utest_log::debug() << "DMANN_NN = DMACostModel (NPU 5.0 model used as proxy for NPU 6.0) cycles\n";
    utest_log::debug() << "Th_PTL   = DMATheoreticalCostProvider_PTL cycles (independent baseline; no stride penalty)\n";
    utest_log::debug() << "Th_LNL   = DMATheoreticalCostProvider_LNL_Legacy cycles (legacy formula)\n";
    utest_log::debug() << "D%(M-NN)  = (MATH - NN)  / NN  * 100  (+ means MATH slower than NN)\n";
    utest_log::debug() << "D%(M-PTL) = (MATH - PTL) / PTL * 100  (+ means MATH slower than Theoretical PTL)\n";
    utest_log::debug() << "D%(M-LNL) = (MATH - LNL) / LNL * 100  (+ means MATH slower than Theoretical LNL Legacy)\n";
    utest_log::debug() << "\n==================== END ====================\n";
}

/// Test that VPUCostModel::DMA(const VPUDMADescriptor&) never throws when sanity checks fail.
///
/// Before the fix, a descriptor that triggered checkDescriptorSanity() would propagate an
/// std::exception all the way to the caller.  After the fix the exception is caught internally,
/// a warning is logged, and ERROR_INVALID_INPUT_CONFIGURATION is returned instead.
///
/// Cases verified:
///   (a) A valid descriptor → no throw, non-error cycles returned.
///   (b) Sub-byte source dtype (UINT4) → no throw, ERROR_INVALID_INPUT_CONFIGURATION.
///   (c) Sub-byte destination dtype (INT4) → no throw, ERROR_INVALID_INPUT_CONFIGURATION.
///   (d) Invalid device (VPUDevice::__size) → no throw, ERROR_INVALID_INPUT_CONFIGURATION.
///   (e) Invalid src.num_dims (> VPU_DMA_MAX_DIMS) → no throw, ERROR_INVALID_INPUT_CONFIGURATION.
///   (f) Unsupported memory-location combination → no throw, ERROR_INVALID_INPUT_CONFIGURATION.
TEST_F(TestDMA_Strided_CostModelVPU6x, DMA_Descriptor_SanityFail_DoesNotThrow) {
    // Cost model without an NN — DMA(VPUDMADescriptor) always uses the strided-math provider.
    VPUCostModel cm{VPUNN::VPUDevice::__size, "empty"};

    // Helper: build a minimal valid 1D contiguous descriptor for NPU_6_0 CMX→CMX.
    auto makeValidDesc = []() {
        VPUDMADescriptor desc;
        desc.device = VPUDevice::NPU_6_0;
        desc.src_location = MemoryLocation::CMX;
        desc.dst_location = MemoryLocation::CMX;
        desc.src.dtype = DataType::UINT8;
        desc.src.setDimension_OutermostFirst({{4096, 1}});
        desc.dst.dtype = DataType::UINT8;
        desc.dst.setDimension_OutermostFirst({{4096, 1}});
        return desc;
    };

    // (a) Valid descriptor — must not throw and must return a non-error cycle count.
    {
        const VPUDMADescriptor desc = makeValidDesc();
        CyclesInterfaceType result{0};
        EXPECT_NO_THROW(result = cm.DMA(desc)) << "Valid descriptor must not throw";
        EXPECT_FALSE(Cycles::isErrorCode(result))
                << "Valid descriptor must return non-error cycles, got: " << Cycles::toErrorText(result);
        EXPECT_GT(result, 0u) << "Valid descriptor must return positive cycles";
    }

    // (b) Sub-byte source dtype (UINT4) — sanity check rejects it.
    {
        VPUDMADescriptor desc = makeValidDesc();
        desc.src.dtype = DataType::UINT4;
        CyclesInterfaceType result{0};
        EXPECT_NO_THROW(result = cm.DMA(desc)) << "Sub-byte src dtype must not throw";
        EXPECT_EQ(result, Cycles::ERROR_INVALID_INPUT_CONFIGURATION)
                << "Sub-byte src dtype must return ERROR_INVALID_INPUT_CONFIGURATION";
    }

    // (c) Sub-byte destination dtype (INT4) — sanity check rejects it.
    {
        VPUDMADescriptor desc = makeValidDesc();
        desc.dst.dtype = DataType::INT4;
        CyclesInterfaceType result{0};
        EXPECT_NO_THROW(result = cm.DMA(desc)) << "Sub-byte dst dtype must not throw";
        EXPECT_EQ(result, Cycles::ERROR_INVALID_INPUT_CONFIGURATION)
                << "Sub-byte dst dtype must return ERROR_INVALID_INPUT_CONFIGURATION";
    }

    // (d) Invalid device (VPUDevice::__size) — sanity check rejects it.
    {
        VPUDMADescriptor desc = makeValidDesc();
        desc.device = VPUDevice::__size;
        CyclesInterfaceType result{0};
        EXPECT_NO_THROW(result = cm.DMA(desc)) << "Invalid device must not throw";
        EXPECT_EQ(result, Cycles::ERROR_INVALID_INPUT_CONFIGURATION)
                << "Invalid device must return ERROR_INVALID_INPUT_CONFIGURATION";
    }

    // (e) src.num_dims out of range (> VPU_DMA_MAX_DIMS) — sanity check rejects it.
    {
        VPUDMADescriptor desc = makeValidDesc();
        desc.src.num_dims = VPU_DMA_MAX_DIMS + 1;
        CyclesInterfaceType result{0};
        EXPECT_NO_THROW(result = cm.DMA(desc)) << "Out-of-range src.num_dims must not throw";
        EXPECT_EQ(result, Cycles::ERROR_INVALID_INPUT_CONFIGURATION)
                << "Out-of-range src.num_dims must return ERROR_INVALID_INPUT_CONFIGURATION";
    }

    // (f) Unsupported memory-location combination (DRAM src + DRAM-as-undefined dst alias).
    //     Use two values that don't map to any of the four supported MemoryDirection entries.
    {
        VPUDMADescriptor desc = makeValidDesc();
        // MemoryLocation::__size is outside the valid range and forces getDirection() == MemoryDirection::__size.
        desc.src_location = MemoryLocation::__size;
        CyclesInterfaceType result{0};
        EXPECT_NO_THROW(result = cm.DMA(desc)) << "Unsupported memory-location pair must not throw";
        EXPECT_EQ(result, Cycles::ERROR_INVALID_INPUT_CONFIGURATION)
                << "Unsupported memory-location pair must return ERROR_INVALID_INPUT_CONFIGURATION";
    }
}

}  // namespace VPUNN_unit_tests
