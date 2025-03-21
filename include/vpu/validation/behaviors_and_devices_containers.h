// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_VPU_BEHAVIORS_AND_DEVICES_CONTAINERS_H
#define VPUNN_VPU_BEHAVIORS_AND_DEVICES_CONTAINERS_H

#include "data_dpu_operation.h"
#include "interface_operations_behavior.h"
#include "interface_valid_values.h"
#include "vpu/types.h"

#include <sstream>

namespace VPUNN {

/// @brief basic implementation for IContainer_OperationsDynamicBehavior
/// maps operations to behaviors (provided as classed that implement IOperationDynamicConstraints interface at least)
/// the order  matters, the map is fixed to be in the following order:
/// CONVOLUTION, DW_CONVOLUTION, CM_CONVOLUTION, ELTWISE, MAXPOOL
///
/// @tparam TOperationsBehavior parameter pack providing the correctly ordered list of behaviors
template <class... TOperationsBehavior>
class Behaviours : public IContainer_OperationsDynamicBehavior {
private:
    using OpList = std::tuple<TOperationsBehavior...>;
    OpList op_list;

public:
    /// @brief provides the behavior associated with the desired operation
    ///
    /// @param op the desired operation, will throw if not supported
    /// @throws runtime_error if the operation is not supported or known
    /// @returns IOPerationsConstrainsto be used for this operation
    const IOperationDynamicConstraints& get_operation_specific_behaviour(const Operation op) const override {
        return get_operation_specific_<IOperationDynamicConstraints>(op);
    }

    /// @brief provides the behavior associated with the desired operation
    ///
    /// @param op the desired operation, will throw if not supported
    /// @throws runtime_error if the operation is not supported or known
    /// @returns Interface to be used for this operation
    template <class I>
    const I& get_operation_specific_(const Operation op) const {
        switch (op) {
        case Operation::CONVOLUTION:
            static_assert((std::tuple_size<OpList>::value) > 0, "check tuple");
            return std::get<0>(op_list);
            break;

        case Operation::DW_CONVOLUTION:
            static_assert((std::tuple_size<OpList>::value) > 1, "check tuple");
            return std::get<1>(op_list);
            break;

        case Operation::CM_CONVOLUTION:
            static_assert((std::tuple_size<OpList>::value) > 2, "check tuple");
            return std::get<2>(op_list);
            break;

        case Operation::ELTWISE:
            static_assert((std::tuple_size<OpList>::value) > 3, "check tuple");
            return std::get<3>(op_list);
            break;

        case Operation::MAXPOOL:
            static_assert((std::tuple_size<OpList>::value) > 4, "check tuple");
            return std::get<4>(op_list);
            break;

        case Operation::LAYER_NORM:
            static_assert((std::tuple_size<OpList>::value) > 5, "check tuple");
            return std::get<5>(op_list);
            break;

        case Operation::ELTWISE_MUL:
            static_assert((std::tuple_size<OpList>::value) > 6, "check tuple");
            return std::get<6>(op_list);
            break;

        default: {
            // should throw!
            std::stringstream buffer;
            std::string op_text{(mapToText<Operation>().find(static_cast<int>(op))) == mapToText<Operation>().cend()
                                        ? "UNKNOWN OPERATION VALUE"
                                        : (mapToText<Operation>().find(static_cast<int>(op)))->second};
            buffer << "[ERROR]OperationBehaviours::get_operation_specific_(), \n Operation not "
                      "supported: "
                   << static_cast<int>(op) << " : [ " << op_text << " ]  No behavior available! \n"
                   << " File: " << __FILE__ << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::runtime_error(details);
        }
        }
    }
};

/// Holds dynamic Behaviors (operations based) and Rules based on devices. Makes the connection between the device Rules
/// and the dynamic behaviors
/// The behaviors are common for all devices
template <class OpBehavior, class... DeviceValues>
class Behavior_Device_Mapping {
protected:
    const OpBehavior specific_behaviours{};  ///< known behaviors for each operation
protected:
    const std::tuple<DeviceValues...> specific_vv{
            DeviceValues(specific_behaviours)...  //
    };

    /// static rules (described by data) for each device
    const std::vector<const IDeviceValidValues*> validators_config{
            &(std::get<0>(specific_vv)),  //
            &(std::get<1>(specific_vv)),  //

            &(std::get<2>(specific_vv)),  //
           // &(std::get<3>(specific_vv))   //
    };

public:
    /// @brief true if the device is supported by this instance
    bool is_supported(VPUNN::VPUDevice device) const {
        bool found = false;
        for (const auto config : validators_config) {
            if (config != nullptr) {
                auto it = std::find(config->get_devices().cbegin(), config->get_devices().cend(), device);
                if (it != config->get_devices().cend()) {
                    found = true;
                    break;
                }
            }
        }
        return found;
    }

    /// @brief gets the data that contain valid values for this device. Throws if unknown device
    const IDeviceValidValues& get_config(VPUNN::VPUDevice device) const {
        for (const auto config : validators_config) {
            if (config != nullptr) {
                auto it = std::find(config->get_devices().cbegin(), config->get_devices().cend(), device);
                if (it != config->get_devices().cend()) {
                    return *config;
                    break;
                }
            }
        }

        {
            std::stringstream buffer;
            buffer << "[ERROR] NO configuration exist for device : " << (int)device << " [ "
                   << VPUDevice_ToText.at((int)device) << " ]"
                   << " File: " << __FILE__ << " Line: " << __LINE__;
            std::string details = buffer.str();
            throw std::runtime_error(details);
        }
    }
};

}  // namespace VPUNN

#endif  //
