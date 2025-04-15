// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include "core/tensors.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestTensor : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Test cases covering the creation of the Tensor via its simple constructor, no init value
TEST_F(TestTensor, CreationNoInit) {
    struct Test {
        std::vector<unsigned int> dimensions;
        int expected_size;
    };
    std::vector<Test> test_vector{
            {{3U, 5U, 7U}, 3 * 5 * 7},
            {{1U, 9U, 8U}, 1 * 9 * 8},
            {{1U, 5U, 6U, 7U}, 1 * 5 * 6 * 7},
    };

    for (const auto& tst : test_vector) {
        VPUNN::Tensor<float> t(tst.dimensions);
        EXPECT_EQ(t.size(), tst.expected_size) << "size is not matching";

        const auto& t_dim = t.shape();
        ASSERT_EQ(t_dim.size(), tst.dimensions.size()) << "tensor dimensionality is not matching";
        ASSERT_EQ(t_dim, tst.dimensions);

        EXPECT_NE(t.data(), nullptr) << "Internal data must exist";
        EXPECT_EQ(t.data(), t.c_ptr());
    }
}
/// Test cases covering the creation of the Tensor and initialization of data
TEST_F(TestTensor, CreationwithInit) {
    struct Test {
        std::vector<unsigned int> dimensions;
        int expected_size;
        float init_val = 0.0;
    };
    std::vector<Test> test_vector{
            {{2U, 5U, 7U}, 2 * 5 * 7},
            {{3U, 9U, 8U}, 3 * 9 * 8, -2.77F},
            {{8U, 1U, 1U, 7U}, 8 * 1 * 1 * 7, 4.332F},
    };

    for (const auto& tst : test_vector) {
        VPUNN::Tensor<float> t(tst.dimensions, tst.init_val);
        EXPECT_EQ(t.size(), tst.expected_size) << "size is not matching";

        const auto& t_dim = t.shape();
        ASSERT_EQ(t_dim.size(), tst.dimensions.size()) << "tensor dimensionality is not matching";
        ASSERT_EQ(t_dim, tst.dimensions);

        EXPECT_NE(t.data(), nullptr) << "Internal data must exist";
        EXPECT_EQ(t.data(), t.c_ptr());

        for (int i = 0; i < t.size(); ++i) {
            ASSERT_FLOAT_EQ(t[i], tst.init_val) << "index = " << i;
        }
    }
}

/// Test cases covering the creation of the Tensor with external memory allocated
TEST_F(TestTensor, CreationExternalData) {
    struct Test {
        std::vector<unsigned int> dimensions;
        int expected_size;
    };
    std::vector<Test> test_vector{
            {{3U, 5U, 7U}, 3 * 5 * 7},
            {{1U, 9U, 8U}, 1 * 9 * 8},
            {{1U, 5U, 6U, 7U}, 1 * 5 * 6 * 7},
    };

    for (const auto& tst : test_vector) {
        float* memory = new float[tst.expected_size];

        VPUNN::Tensor<float> t(memory, tst.dimensions);
        EXPECT_EQ(t.size(), tst.expected_size) << "size is not matching";

        const auto& t_dim = t.shape();
        ASSERT_EQ(t_dim.size(), tst.dimensions.size()) << "tensor dimensionality is not matching";
        ASSERT_EQ(t_dim, tst.dimensions);

        EXPECT_NE(t.data(), nullptr) << "Internal data must exist";
        EXPECT_EQ(t.data(), memory);
    }

    // special test to check that memory is deallocated
    {
        auto& tst = test_vector[0];
        float* memory = new float[tst.expected_size];
        {
            float* raw{memory};
            float base{1000.7899345F};  // just an offset
            for (int i = 0; i < tst.expected_size; ++i) {
                raw[i] = base + ((float)(i+100) * (float)tst.expected_size)/(float)tst.expected_size;
            }
        }

        VPUNN::Tensor<float>* pt =
                new VPUNN::Tensor<float>(memory, tst.dimensions);  // now pt is considered the owner of memory
        VPUNN::Tensor<float>& t = *pt;
        EXPECT_EQ(t.size(), tst.expected_size) << "size is not matching";
        ASSERT_EQ(t.data(), memory);

        auto data_vector = t.data_vector();
        EXPECT_EQ(t.size(), data_vector.size()) << "size is not matching";
        for (unsigned int idx = 0; idx < data_vector.size(); idx++) {
            ASSERT_EQ(*(t.data() + idx), data_vector[idx]);
        }

        // delete the memory, implies the pt will have an invalid pointer that will try to deallocate(2nd time) at
        // destruction
        delete[] memory;

        // now delete the tensor, it will crash if tries to delete the memory second time
#ifdef _WINDOWS  // only on windows it crashes
        ASSERT_DEATH_IF_SUPPORTED(delete pt, "");
#else   // on linux (non windows) cannot make it crash and Gtest to catch it
        // EXPECT_EXIT(delete pt, testing::ExitedWithCode(0), "");
        // EXPECT_DEATH_IF_SUPPORTED(delete pt, "");
        // delete pt;
        // EXPECT_NO_THROW(delete pt);
#endif  // _WINDOWS
        /* coverity[leaked_storage] */
    }
}

/// Tests the copy operations
TEST_F(TestTensor, ObjectCopyConstructor) {
    {
        std::vector<unsigned int> dimensions{3U, 5U, 7U};  // must be non zero (well formed)
        Tensor<float> t1(dimensions);
        {
            float* raw{t1.data()};
            float base{7.7899345F};  // just an offset
            for (int i = 0; i < t1.size(); ++i) {
                raw[i] = base + (float)i;
            }
        }

        // use a copy constructor
        Tensor<float> t2_cc{t1};
        ASSERT_EQ(t2_cc.size(), t1.size());
        ASSERT_EQ(t2_cc.shape(), t1.shape());
        EXPECT_NE(t2_cc.data(), nullptr) << "Internal data must exist";
        EXPECT_NE(t2_cc.data(), t1.data()) << "Internal pointers must be different";

        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_EQ(t2_cc.data()[i], t1.data()[i]) << "content not matching at index: " << i << std::endl;
        }
    }

    // todo: add here more tests.
}

/// Tests the = operator
TEST_F(TestTensor, ObjectAsignementOperator) {
    {
        std::vector<unsigned int> dim1{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t1(dim1, 22.7F);        // init

        std::vector<unsigned int> dim2{2U, 2U, 5U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t2(dim2, 50.0F);        // init

        t2 = t1;
        ASSERT_EQ(t2.size(), t1.size());
        ASSERT_EQ(t2.shape(), t1.shape());
        EXPECT_NE(t2.data(), nullptr) << "Internal data must exist";
        EXPECT_NE(t2.data(), t1.data()) << "Internal pointers must be different";

        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_EQ(t2.data()[i], t1.data()[i]) << "content not matching at index: " << i << std::endl;
        }
    }
    {                                                // exact size match, no realloc
        std::vector<unsigned int> dim1{4U, 4U, 1U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t1(dim1, 22.7F);        // init

        std::vector<unsigned int> dim2{4U, 4U, 1U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t2(dim2, 50.0F);        // init

        t2 = t1;
        ASSERT_EQ(t2.size(), t1.size());
        ASSERT_EQ(t2.shape(), t1.shape());
        EXPECT_NE(t2.data(), nullptr) << "Internal data must exist";
        EXPECT_NE(t2.data(), t1.data()) << "Internal pointers must be different";

        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_EQ(t2.data()[i], t1.data()[i]) << "content not matching at index: " << i << std::endl;
        }
    }
}
/// Tests scenarios for assign method
TEST_F(TestTensor, AssignMethod) {
    {                                                      // size is the same
        const std::vector<unsigned int> dim1{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t1(dim1, 21.7F);              // init

        std::array<float, 3 * 3 * 2> memory;
        std::fill(memory.begin(), memory.end(), 101.0F);

        EXPECT_NO_THROW(t1.assign(memory.data(), static_cast<unsigned int>(memory.size() * sizeof(memory[0]))));

        ASSERT_EQ(t1.size(), memory.size());
        int32_t size_from_shape = std::accumulate(t1.shape().begin(), t1.shape().end(), 1, std::multiplies<int>());

        ASSERT_EQ(t1.size(), size_from_shape) << "size and shape information are not consistent";
        EXPECT_NE(t1.data(), nullptr) << "Internal data must exist";
        EXPECT_NE(t1.data(), memory.data()) << "Internal pointers must be different";

        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_EQ(t1[i], memory[i]) << "content not matching at index: " << i << std::endl;
        }
    }

    {                                                      // size increases
        const std::vector<unsigned int> dim1{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t1(dim1, 22.7F);              // init

        std::array<float, 20> memory;
        std::fill(memory.begin(), memory.end(), 100.0F);

        ASSERT_THROW(t1.assign(memory.data(), static_cast<unsigned int>(memory.size() * sizeof(memory[0]))),
                     std::runtime_error);

        // ASSERT_EQ(t1.size(), memory.size());
        int32_t size_from_shape = std::accumulate(t1.shape().begin(), t1.shape().end(), 1, std::multiplies<int>());

        ASSERT_EQ(t1.size(), size_from_shape) << "size and shape information are not consistent";
        EXPECT_NE(t1.data(), nullptr) << "Internal data must exist";
        EXPECT_NE(t1.data(), memory.data()) << "Internal pointers must be different";

        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_FLOAT_EQ(t1[i], 22.7F) << "content not matching at index: " << i << std::endl;
        }
    }
}
/// Tests scenarios for assign method with bad inputs
TEST_F(TestTensor, AssignMethodPathologicalInput) {
    {                                                      // pathologic dimension/size
        const std::vector<unsigned int> dim1{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t1(dim1, 23.7F);              // init

        std::array<float, 3U * 3U * 2U> memory;
        std::fill(memory.begin(), memory.end(), 103.0F);

        unsigned int specified_size = static_cast<unsigned int>(memory.size() * sizeof(memory[0])) - 1U;  // bad

        unsigned int initial_size = t1.size();
        float initial_fill = t1[0];
        ASSERT_THROW(t1.assign(memory.data(), specified_size), std::runtime_error);

        ASSERT_EQ(t1.size(), initial_size);
        int32_t size_from_shape = std::accumulate(t1.shape().begin(), t1.shape().end(), 1, std::multiplies<int>());

        ASSERT_EQ(t1.size(), size_from_shape) << "size and shape information are not consistent";
        EXPECT_NE(t1.data(), nullptr) << "Internal data must exist";
        EXPECT_NE(t1.data(), memory.data()) << "Internal pointers must be different, nothing was set";

        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_EQ(t1[i], initial_fill) << "content not matching at index: " << i << std::endl;
        }
    }
}

/// Tests scenarios for move constructor
TEST_F(TestTensor, MoveConstructor) {
    {                                                      // move constructor simple case
        const std::vector<unsigned int> dim1{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> t0(dim1, 23.7F);              // init

        float* initial_memory = t0.data();
        unsigned int initial_size = t0.size();
        float initial_fill = t0[0];

        VPUNN::Tensor<float> t1(std::move(t0));

        ASSERT_EQ(t1.size(), initial_size);
        int32_t size_from_shape = std::accumulate(t1.shape().begin(), t1.shape().end(), 1, std::multiplies<int>());

        ASSERT_EQ(t1.size(), size_from_shape) << "size and shape information are not consistent";
        EXPECT_NE(t1.data(), nullptr) << "Internal data must exist";
        EXPECT_EQ(t1.data(), initial_memory) << "Internal pointers must be different, nothing was set";
        for (auto i = 0; i < t1.size(); ++i) {
            EXPECT_EQ(t1[i], initial_fill) << "content not matching at index: " << i << std::endl;
        }

        // check also the source of move, should be undefined but usable
        /* coverity[use_after_move] */
        EXPECT_EQ(t0.size(), 0) << "moved obj should be empty";
        EXPECT_EQ(t0.data(), nullptr) << "not data expected in moved object";
        EXPECT_EQ(t0.shape().size(), 0) << "no dimensions in an empty moved object";
    }
}

/// Tests scenarios for move assignment
TEST_F(TestTensor, MoveAssignement) {
    {                                                       // move assignment simple
        const std::vector<unsigned int> dim_d{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> td(dim_d, 22.7F);              // init

        float* d_initial_memory = td.data();
        unsigned int d_initial_size = td.size();
        float d_initial_fill = td[0];

        const std::vector<unsigned int> dim_s{2U, 2U, 5U};  // must be non zero (well formed)
        {  // put source in a sub block so we see also the post destruction behavior
            VPUNN::Tensor<float> ts(dim_s, 50.0F);  // init
            float* initial_memory = ts.data();
            unsigned int initial_size = ts.size();
            float initial_fill = ts[0];

            td = std::move(ts);
            ASSERT_EQ(td.size(), initial_size);
            ASSERT_EQ(td.shape(), dim_s);
            EXPECT_NE(td.data(), nullptr) << "Internal data must exist";
            EXPECT_EQ(td.data(), initial_memory) << "Internal pointer is from moved object";

            for (auto i = 0; i < td.size(); ++i) {
                EXPECT_EQ(td[i], initial_fill) << "content not matching at index: " << i << std::endl;
            }

            // in our impl ts is consistent and is from td (this check might change if we change the impl). ts is moved
            // but stable
            // coverity[use_after_move]
            ASSERT_EQ(ts.size(), d_initial_size);
            ASSERT_EQ(ts.shape(), dim_d);
            EXPECT_EQ(ts.data(), d_initial_memory) << "Internal pointer is from moved object";
            EXPECT_EQ(ts[0], d_initial_fill);

        }  // here the ts is destroyed , this should have no influence on td

        ASSERT_EQ(td.shape(), dim_s);
        EXPECT_EQ(td[0], 50.F);

    }  // td is destroyed

    {                                                      // move assignment self, very odd case
        const std::vector<unsigned int> dim1{3U, 3U, 2U};  // must be non zero (well formed)
        VPUNN::Tensor<float> td(dim1, 22.7F);              // init

        float* d_initial_memory = td.data();
        unsigned int d_initial_size = td.size();
        float d_initial_fill = td[0];

        // td = std::move(td);//since it is not working due to -Wself-move
        auto surrogate_td{std::move(td)};
        td = std::move(surrogate_td);

        ASSERT_EQ(td.size(), d_initial_size);
        ASSERT_EQ(td.shape(), dim1);
        EXPECT_NE(td.data(), nullptr) << "Internal data must exist";
        EXPECT_EQ(td.data(), d_initial_memory) << "no change in data pointer";

        for (auto i = 0; i < td.size(); ++i) {
            EXPECT_EQ(td[i], d_initial_fill) << "content not matching at index: " << i << std::endl;
        }

    }  // td is destroyed, no throw here
}

}  // namespace VPUNN_unit_tests
