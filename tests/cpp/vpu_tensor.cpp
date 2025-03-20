// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "vpu/vpu_tensor.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "common_helpers.h"
#include "vpu/dpu_types.h"
#include "vpu/validation/interface_valid_values.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

//@todo: Better Packmodes implementation. INT4 support to be reviewed.

class VPUTensorTest : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }
    struct TestInput {
        std::array<unsigned int, 4> dim;  ///< WHCB
        DataType dtype;
    };

    template <typename T>
    struct TestExpectation {
        T value_expected;
    };

    template <typename T>
    struct TestCase {
        TestInput t_in;
        TestExpectation<T> t_exp;
        std::string t_info = "";
    };

private:
protected:
    static auto size_pk_0(const VPUTensor& tensor) {
        return tensor.size_packmode_0();
    }

    static auto size_pk_1(const VPUTensor& tensor) {
        return tensor.size_packmode_1();
    }

    static auto size_pk_2(const VPUTensor& tensor) {
        return tensor.size_packmode_2();
    }

    static auto size_pk_3(const VPUTensor& tensor) {
        return tensor.size_packmode_3();
    }

    static void throw_if_invalid(const VPUTensor& tensor) {
        tensor.throw_if_invalid();
    }

    static bool is_valid_pk_0(const VPUTensor& tensor) {
        return tensor.is_tensor_valid_packmode_0();
    }

    static bool is_valid_pk_1(const VPUTensor& tensor) {
        return tensor.is_tensor_valid_packmode_1();
    }

    static bool is_valid_pk_2(const VPUTensor& tensor) {
        return tensor.is_tensor_valid_packmode_2();
    }

    static bool is_valid_pk_3(const VPUTensor& tensor) {
        return tensor.is_tensor_valid_packmode_3();
    }

    static auto computeContigElemCntAndSize(const VPUTensor& tensor) {
        return tensor.computeContiguousElementCountAndSize();
    }

    static auto alignedSequencesSize(const VPUTensor& tensor, int dim) {
        return tensor.computeAlignedSequencesSize_B(dim);
    }

    static auto remainingElem(const VPUTensor& tensor, int dim) {
        std::pair<const int, const int> seq_info = tensor.computeContiguousElementCountAndSize();
        const int contiguousSeq_elm = seq_info.first;

        return tensor.sizeOfRemaininElem_B(dim, contiguousSeq_elm);
    }
};
TEST_F(VPUTensorTest, Tensor_size_packmode_0) {
    using TestsVector = std::vector<TestCase<int>>;
    const TestsVector tests = {
            // ZXY layout

            {{{1, 1, 1, 1}, DataType::INT8}, {1}, "TENSOR 1x1x1x1, INT8"},
            {{{1, 1, 2, 1}, DataType::FLOAT16}, {4}, "TENSOR 1x1x2x1, FLOAT16"},

            {{{2, 1, 2, 1}, DataType::INT8}, {4}, "TENSOR 2x1x2x1, INT8"},
            {{{2, 1, 2, 1}, DataType::FLOAT16}, {8}, "TENSOR 2x1x2x1, FLOAT16"},

            {{{1, 1, 1, 1}, DataType::INT32}, {4}, "TENSOR 1x1x1x1, INT32"},
            {{{1, 1, 1, 1}, DataType::FLOAT32}, {4}, "TENSOR 1x1x1x1, FLOAT32"},

            {{{2, 1, 2, 1}, DataType::INT32}, {16}, "TENSOR 2x1x2x1, INT32"},
            {{{2, 1, 2, 1}, DataType::FLOAT32}, {16}, "TENSOR 2x1x2x1, FLOAT32"},

            // INT4
            {{{3, 1, 2, 1}, DataType::INT4}, {3}, "TENSOR 3x1x2x1, INT4"},
            {{{3, 1, 3, 1}, DataType::INT4}, {3 * 2}, "TENSOR 3x1x3x1, INT4"},
            {{{1, 1, 2, 1}, DataType::INT4}, {1}, "TENSOR 1x1x2x1, INT4"},

            {{{3, 4, 1, 1}, DataType::INT4}, {12}, "TENSOR 3x4x1x1, INT4"},      // e (4 bits padding)
            {{{3, 4, 2, 1}, DataType::INT4}, {12}, "TENSOR 3x4x2x1, INT4"},      //  exact
            {{{3, 4, 3, 1}, DataType::INT4}, {12 * 2}, "TENSOR 3x4x3x1, INT4"},  //  (4 bits padding)

            // INT2
            {{{1, 1, 4, 1}, DataType::INT2}, {1}, "TENSOR 1x1x4x1, INT2"},       // exact
            {{{3, 4, 1, 1}, DataType::INT2}, {12}, "TENSOR 3x4x1x1, INT2"},      // (6 bits padding)
            {{{3, 4, 4, 1}, DataType::INT2}, {12}, "TENSOR 3x4x4x1, INT2"},      // exact
            {{{3, 4, 5, 1}, DataType::INT2}, {12 * 2}, "TENSOR 3x4x5x1, INT2"},  // (2 bits padding)

            // INT1
            {{{1, 1, 1, 1}, DataType::INT1}, {1}, "TENSOR 1x1x1x1, INT1"},
            {{{1, 1, 8, 1}, DataType::INT1}, {1}, "TENSOR 1x1x8x1, INT1"},  // exact

            {{{1, 1, 2, 1}, DataType::INT1}, {1}, "TENSOR 1x1x2x1, INT1"},
            {{{1, 2, 2, 1}, DataType::INT1}, {2}, "TENSOR 1x2x2x1, INT1"},

            {{{3, 5, 1, 1}, DataType::INT1}, {15}, "TENSOR 3x5x1x1, INT1"},
            {{{3, 5, 8, 1}, DataType::INT1}, {15}, "TENSOR 3x5x8x1, INT1"},      // exact
            {{{3, 5, 9, 1}, DataType::INT1}, {15 * 2}, "TENSOR 3x5x9x1, INT1"},  // (1 bit padding)

    };

    for (const auto& t : tests) {
        VPUTensor tensor;
        EXPECT_NO_THROW(tensor = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3], t.t_in.dtype))
                << t.t_info << "\n";
        EXPECT_EQ(size_pk_0(tensor), t.t_exp.value_expected) << t.t_info << "\n" << tensor << "\n";
    }
}

TEST_F(VPUTensorTest, Tensor_size_packmode_1) {
    using TestsVector = std::vector<TestCase<int>>;
    const TestsVector tests = {
            // ZXY layout
            {{{1, 1, 1, 1}, DataType::INT8}, {1}, "TENSOR 1x1x1x1, INT8"},
            {{{1, 1, 2, 1}, DataType::FLOAT16}, {4}, "TENSOR 1x1x2x1, FLOAT16"},
            {{{1, 1, 1, 1}, DataType::INT32}, {4}, "TENSOR 1x1x1x1, INT32"},
            {{{1, 1, 2, 1}, DataType::FLOAT32}, {8}, "TENSOR 1x1x2x1, FLOAT32"},

            // INT4
            {{{3, 1, 2, 1}, DataType::INT4}, {3}, "TENSOR 3x1x2x1, INT4"},
            {{{3, 1, 3, 1}, DataType::INT4}, {5}, "TENSOR 3x1x3x1, INT4"},
            {{{1, 1, 2, 1}, DataType::INT4}, {1}, "TENSOR 1x1x2x1, INT4"},

            {{{3, 4, 1, 1}, DataType::INT4}, {6}, "TENSOR 3x4x1x1, INT4"},
            {{{3, 4, 2, 1}, DataType::INT4}, {12}, "TENSOR 3x4x2x1, INT4"},
            {{{3, 4, 3, 1}, DataType::INT4}, {18}, "TENSOR 3x4x3x1, INT4"},

            // INT2
            {{{1, 1, 4, 1}, DataType::INT2}, {1}, "TENSOR 1x1x4x1, INT2"},
            {{{3, 4, 1, 1}, DataType::INT2}, {3}, "TENSOR 3x4x1x1, INT2"},
            {{{3, 4, 4, 1}, DataType::INT2}, {12}, "TENSOR 3x4x4x1, INT2"},
            {{{3, 4, 5, 1}, DataType::INT2}, {15}, "TENSOR 3x4x5x1, INT2"},

            // INT1
            {{{1, 1, 1, 1}, DataType::INT1}, {1}, "TENSOR 1x1x1x1, INT1"},
            {{{1, 1, 8, 1}, DataType::INT1}, {1}, "TENSOR 1x1x8x1, INT1"},

            {{{1, 1, 2, 1}, DataType::INT1}, {1}, "TENSOR 1x1x2x1, INT1"},
            {{{1, 2, 2, 1}, DataType::INT1}, {1}, "TENSOR 1x2x2x1, INT1"},

            {{{3, 5, 1, 1}, DataType::INT1}, {2}, "TENSOR 3x5x1x1, INT1"},
            {{{3, 5, 8, 1}, DataType::INT1}, {15}, "TENSOR 3x5x8x1, INT1"},
            {{{3, 5, 9, 1}, DataType::INT1}, {17}, "TENSOR 3x5x9x1, INT1"},
    };

    for (const auto& t : tests) {
        VPUTensor tensor;
        EXPECT_NO_THROW(tensor = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3], t.t_in.dtype))
                << t.t_info << "\n";

        EXPECT_EQ(size_pk_1(tensor), t.t_exp.value_expected) << t.t_info << "\n";
    }
}

// right now we can't have any specific test case for packmode2 because all types we have are submultiples of 1 or 2
// bytes
TEST_F(VPUTensorTest, Tensor_size_packmode_2) {
    using TestsVector = std::vector<TestCase<int>>;
    const TestsVector tests = {
            // ZXY layout
            {{{1, 1, 1, 1}, DataType::INT8}, {1}, "TENSOR 1x1x1x1, INT8"},
            {{{1, 1, 2, 1}, DataType::FLOAT16}, {4}, "TENSOR 1x1x2x1, FLOAT16"},
            {{{1, 1, 1, 1}, DataType::INT32}, {4}, "TENSOR 1x1x1x1, INT32"},
            {{{1, 1, 2, 1}, DataType::FLOAT32}, {8}, "TENSOR 1x1x2x1, FLOAT32"},

            // INT4
            {{{3, 1, 2, 1}, DataType::INT4}, {3}, "TENSOR 3x1x2x1, INT4"},
            {{{3, 1, 3, 1}, DataType::INT4}, {6}, "TENSOR 3x1x3x1, INT4"},
            {{{1, 1, 2, 1}, DataType::INT4}, {1}, "TENSOR 1x1x2x1, INT4"},

            {{{3, 4, 1, 1}, DataType::INT4}, {12}, "TENSOR 3x4x1x1, INT4"},
            {{{3, 4, 2, 1}, DataType::INT4}, {12}, "TENSOR 3x4x2x1, INT4"},
            {{{3, 4, 3, 1}, DataType::INT4}, {24}, "TENSOR 3x4x3x1, INT4"},

            // INT2
            {{{1, 1, 4, 1}, DataType::INT2}, {1}, "TENSOR 1x1x4x1, INT2"},
            {{{3, 4, 1, 1}, DataType::INT2}, {12}, "TENSOR 3x4x1x1, INT2"},
            {{{3, 4, 4, 1}, DataType::INT2}, {12}, "TENSOR 3x4x4x1, INT2"},
            {{{3, 4, 5, 1}, DataType::INT2}, {24}, "TENSOR 3x4x5x1, INT2"},

            // INT1
            {{{1, 1, 1, 1}, DataType::INT1}, {1}, "TENSOR 1x1x1x1, INT1"},
            {{{1, 1, 8, 1}, DataType::INT1}, {1}, "TENSOR 1x1x8x1, INT1"},

            {{{1, 1, 2, 1}, DataType::INT1}, {1}, "TENSOR 1x1x2x1, INT1"},
            {{{1, 2, 2, 1}, DataType::INT1}, {2}, "TENSOR 1x2x2x1, INT1"},

            {{{3, 5, 1, 1}, DataType::INT1}, {15}, "TENSOR 3x5x1x1, INT1"},
            {{{3, 5, 8, 1}, DataType::INT1}, {15}, "TENSOR 3x5x8x1, INT1"},
            {{{3, 5, 9, 1}, DataType::INT1}, {30}, "TENSOR 3x5x9x1, INT1"},

    };

    for (const auto& t : tests) {
        VPUTensor tensor;
        EXPECT_NO_THROW(tensor = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3], t.t_in.dtype))
                << t.t_info << "\n";

        EXPECT_EQ(size_pk_2(tensor), t.t_exp.value_expected) << t.t_info << "\n";
    }
}

TEST_F(VPUTensorTest, Tensor_size_packmode_3) {
    using TestsVector = std::vector<TestCase<int>>;
    const TestsVector tests = {
            // ZXY layout
            {{{1, 1, 1, 1}, DataType::INT8}, {1}, "TENSOR 1x1x1x1, INT8"},
            {{{1, 1, 2, 1}, DataType::FLOAT16}, {4}, "TENSOR 1x1x2x1, FLOAT16"},
            {{{1, 1, 1, 1}, DataType::INT32}, {4}, "TENSOR 1x1x1x1, INT32"},
            {{{1, 1, 2, 1}, DataType::FLOAT32}, {8}, "TENSOR 1x1x2x1, FLOAT32"},

            // INT4
            {{{3, 1, 2, 1}, DataType::INT4}, {3}, "TENSOR 3x1x2x1, INT4"},
            {{{3, 1, 3, 1}, DataType::INT4}, {5}, "TENSOR 3x1x3x1, INT4"},
            {{{1, 1, 2, 1}, DataType::INT4}, {1}, "TENSOR 1x1x2x1, INT4"},

            {{{3, 4, 1, 1}, DataType::INT4}, {6}, "TENSOR 3x4x1x1, INT4"},
            {{{3, 4, 2, 1}, DataType::INT4}, {12}, "TENSOR 3x4x2x1, INT4"},
            {{{3, 4, 3, 1}, DataType::INT4}, {18}, "TENSOR 3x4x3x1, INT4"},

            // INT2
            {{{1, 1, 4, 1}, DataType::INT2}, {1}, "TENSOR 1x1x4x1, INT2"},
            {{{3, 4, 1, 1}, DataType::INT2}, {3}, "TENSOR 3x4x1x1, INT2"},
            {{{3, 4, 4, 1}, DataType::INT2}, {12}, "TENSOR 3x4x4x1, INT2"},
            {{{3, 4, 5, 1}, DataType::INT2}, {15}, "TENSOR 3x4x5x1, INT2"},

            // INT1
            {{{1, 1, 1, 1}, DataType::INT1}, {1}, "TENSOR 1x1x1x1, INT1"},
            {{{1, 1, 8, 1}, DataType::INT1}, {1}, "TENSOR 1x1x8x1, INT1"},

            {{{1, 1, 2, 1}, DataType::INT1}, {1}, "TENSOR 1x1x2x1, INT1"},
            {{{1, 2, 2, 1}, DataType::INT1}, {1}, "TENSOR 1x2x2x1, INT1"},

            {{{3, 5, 1, 1}, DataType::INT1}, {2}, "TENSOR 3x5x1x1, INT1"},
            {{{3, 5, 8, 1}, DataType::INT1}, {15}, "TENSOR 3x5x8x1, INT1"},
            {{{3, 5, 9, 1}, DataType::INT1}, {17}, "TENSOR 3x5x9x1, INT1"},

    };

    for (const auto& t : tests) {
        VPUTensor tensor;
        EXPECT_NO_THROW(tensor = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3], t.t_in.dtype))
                << t.t_info << "\n";

        EXPECT_EQ(size_pk_3(tensor), t.t_exp.value_expected) << t.t_info << "\n";
    }
}

// OLD STYLE OF SIZE COMPUTATION -> NOW TEST FAIL
// TEST_F(VPUTensorTest, Tensor_size) {
//     using TestsVector = std::vector<TestCase<int>>;
//     const TestsVector tests = {
//             // ZXY layout
//             {{{1, 1, 1, 1}, DataType::INT8}, {1}, "TENSOR 1x1x1x1, INT8"},
//             {{{1, 1, 2, 1}, DataType::FLOAT16}, {4}, "TENSOR 1x1x2x1, FLOAT16"},
//
//             {{{2, 1, 2, 1}, DataType::INT8}, {4}, "TENSOR 2x1x2x1, INT8"},
//             {{{2, 1, 2, 1}, DataType::FLOAT16}, {8}, "TENSOR 2x1x2x1, FLOAT16"},
//
//             {{{3, 1, 2, 1}, DataType::INT4}, {6}, "TENSOR 3x1x2x1, INT4"},
//             {{{1, 1, 2, 1}, DataType::INT4}, {2}, "TENSOR 1x1x2x1, INT4"},
//
//             {{{1, 1, 4, 1}, DataType::INT2}, {4}, "TENSOR 1x1x4x1, INT2"},
//
//     };
//
//     for (const auto& t : tests) {
//         VPUTensor tensor;
//         EXPECT_NO_THROW(tensor = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3], t.t_in.dtype))
//                 << t.t_info << "\n";
//         EXPECT_EQ(tensor.size(), t.t_exp.value_expected) << t.t_info << "\n";
//     }
// }

TEST_F(VPUTensorTest, Tensor_constructor) {  // tests are for pk 3 only!
    using TestsVector = std::vector<TestCase<bool>>;
    const TestsVector tests = {
            {{{1U, 1U, 0U, 1U}, DataType::INT8}, {false}, "Zero, INT8, no throw"},

            {{{1U, 1U, 2U, 1U}, DataType::INT8}, {false}, "Tensor 1x1x2x1, INT8, no throw"},
            {{{1U, 1U, 2U, 1U}, DataType::FLOAT16}, {false}, "Tensor 1x1x2x1, FLOAT16, no throw"},
            {{{1U, 1U, 2U, 1U}, DataType::FLOAT32}, {false}, "Tensor 1x1x2x1, FLOAT32, no throw"},

            {{{2U, 1U, 2U, 1U}, DataType::INT8}, {false}, "Tensor 2x1x2x1, INT8, no throw"},
            {{{2U, 1U, 2U, 1U}, DataType::FLOAT16}, {false}, "Tensor 2x1x2x1, FLOAT16, no throw"},
            {{{2U, 1U, 2U, 1U}, DataType::FLOAT32}, {false}, "Tensor 2x1x2x1, FLOAT32, no throw"},

            {{{1U, 1U, 1U, 1U}, DataType::INT4}, {true}, "Tensor 1x1x1x1, INT4, throws"},
            {{{1U, 1U, 2U, 1U}, DataType::INT4}, {false}, "Tensor 1x1x2x1, INT4, no throw"},
            {{{3U, 1U, 2U, 1U}, DataType::INT4}, {false}, "Tensor 3x1x2x1, INT4, no throw"},
            {{{1U, 1U, 3U, 1U}, DataType::INT4}, {true}, "Tensor 1x1x3x1, INT4, throws"},

            {{{1U, 1U, 1U, 1U}, DataType::INT2}, {true}, "Tensor 1x1x1x1, INT4, throws"},
            {{{1U, 1U, 2U, 1U}, DataType::INT2}, {true}, "Tensor 1x1x2x1, INT4, throws"},
            {{{1U, 1U, 3U, 1U}, DataType::INT2}, {true}, "Tensor 1x1x3x1, INT4, throws"},
            {{{1U, 1U, 4U, 1U}, DataType::INT2}, {false}, "Tensor 1x1x4x1, INT4, no throw"},

            {{{1U, 1U, 1U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 2U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 3U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 4U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 5U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 6U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 7U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},
            {{{1U, 1U, 8U, 1U}, DataType::INT1}, {false}, ", INT1, OK"},
            {{{1U, 1U, 9U, 1U}, DataType::INT1}, {true}, ", INT1, throws"},

    };
    auto test_if_constructor_throws = [](const TestsVector& tests) {
        for (const auto& t : tests) {
            VPUTensor tens_copy{};  // copy (=) operator does not  invoke constructor
            if (t.t_exp.value_expected) {
                EXPECT_NO_THROW(tens_copy = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3],
                                                      t.t_in.dtype));
                EXPECT_THROW(throw_if_invalid(tens_copy), std::invalid_argument);

            } else {
                EXPECT_NO_THROW(tens_copy = VPUTensor(t.t_in.dim[0], t.t_in.dim[1], t.t_in.dim[2], t.t_in.dim[3],
                                                      t.t_in.dtype));
                EXPECT_NO_THROW(throw_if_invalid(tens_copy));
            }
        }
    };
    test_if_constructor_throws(tests);
}

// test for function computeContiguousElementCountAndSize()
TEST_F(VPUTensorTest, Contiguous_Element_Count_And_Size_Test) {
    using TestsVector = std::vector<TestCase<std::pair<const int, const int>>>;

    auto test_message = [](VPUTensor tensor) {
        // clang-format off
        std::string message = "Tensor shape: {" +
                              std::to_string(tensor.get_shape()[0]) + ", " +
                              std::to_string(tensor.get_shape()[1]) + ", " +
                              std::to_string(tensor.get_shape()[2]) + ", " +
                              std::to_string(tensor.get_shape()[3]) + "} " +
                              DataType_ToText.at(static_cast<int>(tensor.get_dtype())) +
                              Layout_ToText.at(static_cast<int>(tensor.get_layout()))
                              +"\n";

        // clang-format on

        return message;
    };

    const TestsVector tests = {
            // ZXY layout

            {{{112, 32, 121, 1}, DataType::INT8}, {std::make_pair(1, 1)}},
            {{{45, 56, 22, 1}, DataType::FLOAT16}, {std::make_pair(1, 2)}},

            {{{203, 133, 24, 1}, DataType::INT4}, {std::make_pair(2, 1)}},
            {{{62, 167, 222, 1}, DataType::INT2}, {std::make_pair(4, 1)}},
            {{{132, 41, 21, 1}, DataType::INT1}, {std::make_pair(8, 1)}},

            {{{112, 32, 121, 1}, DataType::INT32}, {std::make_pair(1, 4)}},
            {{{45, 56, 22, 1}, DataType::FLOAT32}, {std::make_pair(1, 4)}},
    };

    auto lambda = [test_message](const TestsVector& tests) {
        for (const auto& t : tests) {
            VPUTensor tensor{t.t_in.dim, t.t_in.dtype};
            std::cout << test_message(tensor);

            std::pair<const int, const int> result = computeContigElemCntAndSize(tensor);

            EXPECT_EQ(result.first, t.t_exp.value_expected.first);
            EXPECT_EQ(result.second, t.t_exp.value_expected.second);
        }
    };
    lambda(tests);
}

// test for computeAlignedSequencesSize_B()
TEST_F(VPUTensorTest, Aligned_Sequences_Size_Test) {
    using TestsVector = std::vector<TestCase<std::array<int, 3>>>;

    auto test_message = [](VPUTensor tensor) {
        // clang-format off
        std::string message = "Tensor shape: {" +
                              std::to_string(tensor.get_shape()[0]) + ", " +
                              std::to_string(tensor.get_shape()[1]) + ", " +
                              std::to_string(tensor.get_shape()[2]) + ", " +
                              std::to_string(tensor.get_shape()[3]) + "} " +
                              DataType_ToText.at(static_cast<int>(tensor.get_dtype())) +
                              Layout_ToText.at(static_cast<int>(tensor.get_layout()))
                              +"\n";

        // clang-format on

        return message;
    };

    const TestsVector tests = {

            // expected value is an array containing the size in bytes of the total number of sequences,
            // when innermost dimension (which is the parameter of the function computeAlignedSequencesSize_B() )
            //  is first W, then H, then C
            {{{112, 32, 121, 1}, DataType::INT8}, {{112, 32, 121}}},
            {{{45, 56, 22, 1}, DataType::FLOAT16}, {{90, 112, 44}}},

            {{{203, 133, 24, 1}, DataType::INT4}, {{101, 66, 12}}},
            {{{62, 167, 222, 1}, DataType::INT2}, {{15, 41, 55}}},
            {{{132, 41, 21, 1}, DataType::INT1}, {{16, 5, 2}}},

            {{{112, 32, 121, 1}, DataType::INT32}, {{112 * 4, 32 * 4, 121 * 4}}},
            {{{45, 56, 22, 1}, DataType::FLOAT32}, {{45 * 4, 56 * 4, 22 * 4}}},
    };

    auto lambda = [test_message](const TestsVector& tests) {
        for (const auto& t : tests) {
            VPUTensor tensor{t.t_in.dim, t.t_in.dtype};
            std::cout << test_message(tensor);

            for (int i = 0; i < 3; i++) {
                int result = alignedSequencesSize(tensor, tensor.get_shape()[i]);

                EXPECT_EQ(result, t.t_exp.value_expected[i]);
            }
        }
    };
    lambda(tests);
}

// test for sizeOfRemaininElem_B
TEST_F(VPUTensorTest, RemainingElem_Test) {
    using TestsVector = std::vector<TestCase<std::array<int, 3>>>;

    auto test_message = [](VPUTensor tensor) {
        // clang-format off
        std::string message = "Tensor shape: {" +
                              std::to_string(tensor.get_shape()[0]) + ", " +
                              std::to_string(tensor.get_shape()[1]) + ", " +
                              std::to_string(tensor.get_shape()[2]) + ", " +
                              std::to_string(tensor.get_shape()[3]) + "} " +
                              DataType_ToText.at(static_cast<int>(tensor.get_dtype())) +
                              Layout_ToText.at(static_cast<int>(tensor.get_layout()))
                              +"\n";

        // clang-format on

        return message;
    };

    const TestsVector tests = {

            // expected value is an array containing the size in bytes of the total number of sequences,
            // when innermost dimension (which is the parameter of the function sizeOfRemaininElem_B() )
            //  is first W, then H, then C
            {{{112, 32, 121, 1}, DataType::INT8}, {{0, 0, 0}}},   // number of remaining elem: 0, 0, 0
            {{{45, 56, 22, 1}, DataType::FLOAT16}, {{0, 0, 0}}},  // number of remaining elem: 0, 0, 0

            {{{203, 133, 24, 1}, DataType::INT4}, {{1, 1, 0}}},  // number of remaining elem: 1, 1, 0
            {{{62, 167, 222, 1}, DataType::INT2}, {{1, 1, 1}}},  // number of remaining elem: 2, 3, 1
            {{{132, 41, 21, 1}, DataType::INT1}, {{1, 1, 1}}},   // number of remaining elem: 4, 1, 5

            {{{112, 32, 121, 1}, DataType::INT32}, {{0, 0, 0}}},  // number of remaining elem: 0, 0, 0
            {{{45, 56, 22, 1}, DataType::FLOAT32}, {{0, 0, 0}}},  // number of remaining elem: 0, 0, 0
    };

    auto lambda = [test_message](const TestsVector& tests) {
        for (const auto& t : tests) {
            VPUTensor tensor{t.t_in.dim, t.t_in.dtype};
            std::cout << test_message(tensor);

            for (int i = 0; i < 3; i++) {
                int result = remainingElem(tensor, tensor.get_shape()[i]);

                EXPECT_EQ(result, t.t_exp.value_expected[i]);
            }
        }
    };
    lambda(tests);
}
}  // namespace VPUNN_unit_tests