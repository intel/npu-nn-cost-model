// Copyright © 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef SHAVE_COLLECTION_H
#define SHAVE_COLLECTION_H

#include <map>
#include <memory>
#include <type_traits>

#include "elementwise.h"
#include "interface_shave_op_executor.h"
#include "shave_op_executors.h"
#include "vpu/cycles_interface_types.h"
#include "vpu/types.h"

namespace VPUNN {

/// @brief list of shaves attached to a device
/// must have access to the destructor of the ShaveOpExecutor.
///     owns the executer concrete instances (creation and destruction is in its responsibility)
class DeviceShaveContainer {
private:
    ///@brief in here we can delete a p because we are friends!
    static void deleter(ShaveOpExecutor* p) {
        delete p;
    }

    VPUDevice device;  ///< device intended for this container
    /// the map payload is a unique pointer with delete helper
    /// should always be moved otherwise the pointer will be deleted.
    using map_content_t = std::unique_ptr<ShaveOpExecutor, decltype(&DeviceShaveContainer::deleter)>;
    /// maps names of functions to their model instances (executor). Owns this instances, responsible with destruction
    std::map<std::string, map_content_t> map_shaves;

    /// @brief inserts an executor to the map, full transfer of ownership
    void addOp(map_content_t&& up) {
        if (up != nullptr) {
            // map[up->getName()] = std::move(up);  // overrides previous if existed
            // auto up{map_content_t(p, &deleter)};
            map_shaves.insert({up->getName(), std::move(up)});  // for results checking
        }
    }

public:
    DeviceShaveContainer(VPUDevice device): device(device) {
    }
    DeviceShaveContainer(const DeviceShaveContainer&) = delete;
    DeviceShaveContainer(DeviceShaveContainer&&) = default;

public:
    VPUDevice getDevice() const {
        return device;
    }
    bool existsShave(const std::string sw) const {
        const auto it = map_shaves.find(sw);
        return (it != map_shaves.end());
    }
    std::vector<std::string> getShaveList() const {
        std::vector<std::string> list;
        list.reserve(map_shaves.size());
        for (const auto& i : map_shaves) {
            list.push_back(i.first);
        }

        return list;
    }

    const ShaveOpExecutor& getShaveExecutor(const std::string& sw) const {
        if (existsShave(sw)) {
            return *(map_shaves.at(sw));
        }
        std::stringstream buffer;
        buffer << "[ERROR]:could not find the Shave function with name: " << sw
               << " in the shave list for Device: " << VPUDevice_ToText.at(static_cast<int>(device))
               << " File: " << __FILE__ << " Line: " << __LINE__;
        std::string details = buffer.str();
        throw std::out_of_range(details);
    }

protected:
    // next are helper functions to add a particular instance of concrete  executors, will be used by derived types.

    template <DataType dtype, unsigned int VectorSize, unsigned int UnrollSize, unsigned int DpuFreq,
              unsigned int ShvFreq>
    void Add(const std::string& name, float slope, float intercept, float offset_scalar, float offset_unroll) {
        auto p{new ShaveActivation1on1<dtype, VectorSize, UnrollSize, DpuFreq, ShvFreq>(name, slope, intercept,
                                                                                        offset_scalar, offset_unroll)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }

    //// just a mock
    // void Add(const std::string& name) {
    //     addOp(map_content_t((new ShaveOPMOckTest(name)), &DeviceShaveContainer::deleter));
    // }

    // Legacy modes
    template <typename KERNEL_NAME, unsigned int efficiencyX1000, unsigned int latency>
    void Add(const std::string& name) {
        auto p{new ShaveClassicLinear<KERNEL_NAME, efficiencyX1000, latency>(name)};
        addOp(map_content_t(p, &DeviceShaveContainer::deleter));
    }
};

class ShaveInstanceHolder_VPU27 : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_VPU27(): DeviceShaveContainer(VPUDevice::VPU_2_7) {
        populate();
    }

    void populate();  ///< to be implemented automatically
};

class ShaveInstanceHolder_VPU27CLassic : public DeviceShaveContainer {
public:
    using DeviceShaveContainer::getDevice;
    const DeviceShaveContainer& getContainer() const {
        return *this;
    }

    ShaveInstanceHolder_VPU27CLassic(): DeviceShaveContainer(VPUDevice::VPU_2_7) {
        populate();
    }

    void populate();  ///< to be implemented automatically
};

}  // namespace VPUNN
#endif