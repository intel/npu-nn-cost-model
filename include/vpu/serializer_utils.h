// Copyright @ 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZER_UTILS
#define VPUNN_SERIALIZER_UTILS

#include <string>
#include<functional>

#include "sep_mode.h"

namespace VPUNN {
/// this is a function declaration with two parameters, used for example in serializer or deserializer for arguments
/// that have a member_map either for set value for a variable or get value for a variable
/// a member_map can have as a key value a lambda function defined like this above
///
/// @first parameter is a bool and represents either set_mode when true or get_mode when false
/// @second parameter is a string and represent the value we want to set for a variable when first parameter is true
/// (set_mode), when first parameter is false (get_mode) doesn't matter the value for the second parameter, we just
/// return the variable value, we don't set one
///
/// example of usage: let's assume that we have a lambda function f(bool set_mode, std::string s)
/// for set_mode the function call looks like this f(true, "5") in this case our variable value will be set to 5
/// for get_mode the function call looks like this f(false, "5") or f(false, ""), or f(false, "abc"), important is that
/// first argument should be false, second as you can see doesn't matter
using SetGet_MemberMapValues = std::function<VPUNN::DimType(bool, std::string)>; 

}  // namespace VPUNN

#endif  //