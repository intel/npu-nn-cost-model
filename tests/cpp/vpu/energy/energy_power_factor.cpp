// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.
#include "energy_power_factor.h"
#include "vpu/energy/power.h"

#include <map>
#include <string>

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

/**
 * @brief Test class for validating the structure and integrity of the Power Factor LUT (dev_lut) in VPUPowerFactorLUT
 * This class should be as general as possible, without relying on specific values in the LUT,
 * to ensure that the structure of the LUT is correct and contains the necessary entries for power factor resolution to
 * work properly.
 */
class TestVPUPowerFactorLUTStructure : public ::testing::Test, private VPUPowerFactorLUT {
protected:
    /// @brief Provide test access to the private dev_lut
    /// this access should be only given to TestVPUPowerFactorLUT to be tested
    /// because any other class should not test it for itself
    static const auto& get_dev_lut() {
        return VPUPowerFactorLUT::dev_lut;
    }
};

/**
 * @brief Validates that every engine in every device entry of dev_lut has INT8 defined with values
 *
 * The power factor resolution logic (resolve_power_factor) relies on INT8 being present
 * as the base reference and fallback for each engine. If INT8 is missing or has an empty
 * values_lut, power calculations will produce incorrect results or hit undefined behavior.
 *
 * For each device in dev_lut, for each engine defined in that device, this test checks:
 * 1. INT8 ComputePowerTypeClass exists in the engine's type class map
 * 2. The INT8 PowerFactor has a non-empty values_lut (at least one operation entry)
 * 3. Each operation entry in the values_lut has at least one channel-to-power-factor mapping
 *
 * SCL engine is mandatory for every device, and must have INT8 defined.
 * For DCIM engine, if defined, it must also have INT8 defined.
 */
TEST_F(TestVPUPowerFactorLUTStructure, EveryEngineHasINT8WithValues) {
    const auto& pf_dev_lut = get_dev_lut();

    // The LUT must not be empty - we expect at least the base devices
    ASSERT_FALSE(pf_dev_lut.empty()) << "dev_lut is empty, no devices configured";

    for (size_t dev_idx = 0; dev_idx < pf_dev_lut.size(); ++dev_idx) {
        const auto& device_lut = pf_dev_lut[dev_idx];
        ASSERT_FALSE(device_lut.devices.empty()) << "Device LUT entry [" << dev_idx << "] has an empty devices list";
        const std::string device_name = VPUDevice_ToText.at(static_cast<int>(device_lut.devices.front()));

        // Each device must have at least one engine defined
        ASSERT_FALSE(device_lut.factors.empty())
                << "Device [" << dev_idx << "] " << device_name << " has no engines defined";

        // SCL engine must always be present for every device
        const auto scl_it = device_lut.factors.find(VPUNN::MPEEngine::SCL);
        ASSERT_NE(scl_it, device_lut.factors.end()) << "Device [" << dev_idx << "] " << device_name
                                                    << ": SCL engine is missing (must be defined for every device)";

        for (const auto& [engine, type_class_map] : device_lut.factors) {
            const std::string engine_name = MPEEngine_ToText.at(static_cast<int>(engine));

            // INT8 must exist for this engine
            const auto int8_it = type_class_map.find(details::ComputePowerTypeClass::INT8);
            ASSERT_NE(int8_it, type_class_map.end()) << "Device [" << dev_idx << "] " << device_name << ", Engine "
                                                     << engine_name << ": INT8 ComputePowerTypeClass is missing";

            // INT8 values_lut must not be empty
            const auto& int8_values = int8_it->second.values_lut;
            EXPECT_FALSE(int8_values.empty()) << "Device [" << dev_idx << "] " << device_name << ", Engine "
                                              << engine_name << ": INT8 PowerFactor has empty values_lut";

            // Each operation entry in INT8 values_lut must have at least one channel mapping
            for (size_t op_idx = 0; op_idx < int8_values.size(); ++op_idx) {
                const auto& [operation, channel_map] = int8_values[op_idx];
                const std::string op_name = Operation_ToText.at(static_cast<int>(operation));

                EXPECT_FALSE(channel_map.empty())
                        << "Device [" << dev_idx << "] " << device_name << ", Engine " << engine_name << ", Operation "
                        << op_name << ": INT8 channel-to-power-factor map is empty";
            }
        }
    }
}

/**
 * @brief Validates that dev_lut contains exactly one entry per VPUDevice.
 *
 * Iterates over all VPUDevice enum values. For each device verifies
 * it appears exactly once in dev_lut. It will fail if there are any PowerLUT entries
 * has duplicated devices, and will print a warning if any device is not covered by any LUT entry (e.g. due to embargo
 * macro not enabled). Each device should be covered by exactly one LUT entry to ensure that power factor resolution
 * works correctly
 */
TEST_F(TestVPUPowerFactorLUTStructure, CheckUniqueDeviceUsedForPowerFactorLUT) {
    const auto& pf_dev_lut = get_dev_lut();
    ASSERT_FALSE(pf_dev_lut.empty()) << "dev_lut is empty, no devices configured";

    // Collect how many LUT entries cover each device
    std::map<VPUDevice, int> device_count;
    for (const auto& entry : pf_dev_lut) {
        for (const auto& dev : entry.devices) {
            device_count[dev]++;
        }
    }

    // Check every device in the VPUDevice enum
    for (int i = 0; i < static_cast<int>(VPUDevice::__size); ++i) {
        const auto device = static_cast<VPUDevice>(i);
        const auto& name = VPUDevice_ToText.at(i);
        const int count = device_count.count(device) ? device_count.at(device) : 0;

        if (count == 0) {
            // Device is absent: acceptable (e.g. embargo not enabled), but warn
            std::cout << "NOTE: Device " << name << " has no power LUT entry (embargo macro may be disabled)"
                      << std::endl;
        } else {
            // Device must be covered by exactly one LUT entry
            EXPECT_EQ(count, 1) << "Device " << name << " should appear in exactly one dev_lut entry, but found in "
                                << count;
        }
    }
}

}  // namespace VPUNN_unit_tests
