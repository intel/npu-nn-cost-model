// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.
#include <vpu/cycles_interface_types.h>

#include <gtest/gtest.h>
#include <sstream>  // for error formating
#include "common_helpers.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class TestCyclesInterfaceType : public ::testing::Test {
public:
protected:
    void SetUp() override {
    }

private:
};

/// Some compilers do not support constexpr where references to them are kind of assumed to be needed (GCC/LLVM)
/// some have no problems (MSVC)
TEST_F(TestCyclesInterfaceType, BasicOps) {
    EXPECT_EQ(V(Cycles::NO_ERROR), 0);

    auto mm = std::max(V(Cycles::ERROR_INPUT_TOO_BIG), V(Cycles::ERROR_INVALID_INPUT_CONFIGURATION));
    EXPECT_EQ(mm, V(Cycles::ERROR_INPUT_TOO_BIG));

    EXPECT_NE(V(Cycles::ERROR_INVALID_LAYER_CONFIGURATION), V(Cycles::ERROR_INPUT_TOO_BIG));

    // const VPUNN::CyclesInterfaceType& ref = VPUNN::Cycles::ERROR_INPUT_TOO_BIG;

    //    EXPECT_EQ(VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION, VPUNN::Cycles::ERROR_INPUT_TOO_BIG);

    //  const VPUNN::CyclesInterfaceType* ptr = &VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION;

    // EXPECT_EQ(*ptr, VPUNN::Cycles::ERROR_INPUT_TOO_BIG);
    // EXPECT_EQ(ptr, nullptr);
    // EXPECT_EQ(&VPUNN::Cycles::ERROR_INPUT_TOO_BIG, nullptr);

    //    EXPECT_EQ(ptr, &VPUNN::Cycles::ERROR_INPUT_TOO_BIG);
    //   EXPECT_EQ(ptr, &VPUNN::Cycles::ERROR_INVALID_LAYER_CONFIGURATION);
}

class CycleAdderCheckerTest : public ::testing::Test {
protected:
    static constexpr unsigned int NO_ERROR_EXPECTED{VPUNN::Cycles::NO_ERROR};

    struct TestInput {
        CyclesInterfaceType in0{NO_ERROR_EXPECTED};
        CyclesInterfaceType in1{NO_ERROR_EXPECTED};
    };

    struct TestExpectation {
        CyclesInterfaceType cycl_expected{NO_ERROR_EXPECTED};
    };

    struct TestCase {
        TestInput t_in;
        TestExpectation t_exp;
        const std::string test_case = "";
    };

    using TestsVector = std::vector<TestCase>;

    void SetUp() override {
    }

    void checkerTest(const TestInput& t_in, const bool& t_exp, const std::string& test_case = "") {
        std::string t_header{"** Test Case: " + test_case + "\n"};
        auto input_0 = t_in.in0;
        std::cout << t_header << std::endl;

        EXPECT_EQ(Cycles::isErrorCode(input_0), t_exp)
                << "Expected result: " << t_exp << ", instead received: " << Cycles::isErrorCode(input_0);
    }

    void doBasicTest(const TestInput& t_in, const TestExpectation& t_exp, const std::string& test_case = "") {
        std::string t_header{"** Test Case: " + test_case + "\n"};

        const auto result_cycle = t_exp.cycl_expected;
        auto input_0 = t_in.in0;
        auto input_1 = t_in.in1;

        std::cout << t_header << std::endl;

        EXPECT_EQ(Cycles::cost_adder(input_0, input_1), result_cycle)
                << "Expected result: " << result_cycle
                << ", instead received: " << Cycles::cost_adder(input_0, input_1);

        std::cout << "------------------------------------------------------------------------" << std::endl;
    }

    void executeTests(const TestsVector& tests) {
        int test_index = 0;
        for (const auto& t : tests) {
            std::stringstream buffer;
            buffer << test_index << " : " << t.test_case;
            const std::string test_case_info = buffer.str();

            doBasicTest(t.t_in, t.t_exp, test_case_info);

            ++test_index;
        }
    }
};

// precondition: the implementation and tests were designed for CyclesInterfaceType as unsigned int on 32 bits
static_assert(std::is_same<CyclesInterfaceType, unsigned int>::value, "CyclesInterfaceType must be uint32");
static_assert(std::is_same<CyclesInterfaceType, std::uint32_t>::value, "CyclesInterfaceType must be uint32");

/// @brief Basic test to check the sums that are supported or not
TEST_F(CycleAdderCheckerTest, BasicAdderCases) {
    const std::vector<TestCase> tests{
            {{0, 0}, {0}, "Simple adding case 0-0"},
            {{1, 0}, {1}, "Simple adding case 1-0"},
            {{0, 1}, {1}, "Simple adding case 0-1"},
            {{1, 1}, {2}, "Simple adding case 1-1"},

            {{(std::numeric_limits<CyclesInterfaceType>::max() / 2),
              (std::numeric_limits<CyclesInterfaceType>::max() / 2 - 1)},
             {Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE},
             "Error zone adding"},
            {{Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE, 1},
             {Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE},
             "Error added with normal cycles gives same error"},
            {{Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE, 1000},
             {Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE},
             "Overflow with one operand as error code"},
            {{Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE, Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE},
             {Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE},
             "Adding 2 error codes will give error code"},
            {{Cycles::ERROR_INVALID_LAYER_CONFIGURATION, 10},
             {Cycles::ERROR_INVALID_LAYER_CONFIGURATION},
             "Propagate the first error encountered"},
            {{10, Cycles::ERROR_INVALID_INPUT_DEVICE},
             {Cycles::ERROR_INVALID_INPUT_DEVICE},
             "Propagate the first error encountered"},

            {{1, Cycles::START_ERROR_RANGE}, {Cycles::ERROR_CUMULATED_CYCLES_TOO_LARGE}, "Being above limit"},
            {{1, (Cycles::START_ERROR_RANGE - 1)}, {Cycles::START_ERROR_RANGE}, "Being at the limit"},
            {{1, (Cycles::START_ERROR_RANGE - 2)}, {Cycles::START_ERROR_RANGE - 1}, "Under the limit"},

            {{std::numeric_limits<CyclesInterfaceType>::max(), std::numeric_limits<CyclesInterfaceType>::max()},
             {std::numeric_limits<CyclesInterfaceType>::max()},
             "max term with max"},

            {{std::numeric_limits<CyclesInterfaceType>::max(), 0},
             {std::numeric_limits<CyclesInterfaceType>::max()},
             "max term with zero"},

    };

    executeTests(tests);
}
TEST_F(CycleAdderCheckerTest, ErrorDetection) {
    struct CheckTests {
        TestInput t_in;
        bool t_exp;
        const std::string test_case = "";
    };

    const std::vector<CheckTests> tests{{{0, 0}, false, "zero"},
                                        {{Cycles::START_ERROR_RANGE - 1, 0}, false, "before start range  case!"},
                                        {{Cycles::START_ERROR_RANGE, 0}, false, "start range case!"},
                                        {{Cycles::START_ERROR_RANGE + 1, 0}, true, "after start range case!"},
                                        {{std::numeric_limits<CyclesInterfaceType>::max(), 0}, true, "max case!"}};

    for (const auto& t : tests) {
        checkerTest(t.t_in, t.t_exp, t.test_case);
    }
}

// test what instance is used
TEST_F(CycleAdderCheckerTest, DpuScheduleSpecialisationTest) {
    {  // testing what happens for long

        std::vector<long long> cost_vector = {Cycles::ERROR_INVALID_INPUT_DEVICE};
        long long overhead = 1;
        long long result = VPUNN::dpu_schedule(1, cost_vector, overhead);

        EXPECT_EQ(result, V(Cycles::ERROR_INVALID_INPUT_DEVICE) + 1);
    }
    {  // testing what happens for CycleInterfaceType
        std::vector<CyclesInterfaceType> cost_vector = {Cycles::ERROR_INVALID_INPUT_DEVICE};
        CyclesInterfaceType overhead = 1;
        CyclesInterfaceType result = VPUNN::dpu_schedule(1, cost_vector, overhead);

        EXPECT_EQ(result, V(Cycles::ERROR_INVALID_INPUT_DEVICE));
    }
    {  // testing what happens for CycleInterfaceType
        std::vector<unsigned int> cost_vector = {(unsigned int)Cycles::ERROR_INVALID_INPUT_DEVICE};
        unsigned int overhead = 1;
        unsigned int result = VPUNN::dpu_schedule(1, cost_vector, overhead);

        EXPECT_EQ(result, (unsigned int)V(Cycles::ERROR_INVALID_INPUT_DEVICE));
    }
    {  // testing what happens for unsigned int
        std::vector<unsigned int> cost_vector = {(unsigned int)Cycles::ERROR_INVALID_INPUT_DEVICE};
        unsigned int overhead = 1;
        unsigned int result = VPUNN::dpu_schedule<unsigned int>(1, cost_vector, overhead);

        EXPECT_EQ(result, (unsigned int)V(Cycles::ERROR_INVALID_INPUT_DEVICE));
    }
}

/// dpu_schedule pipelining mechanism checked
TEST_F(CycleAdderCheckerTest, DpuScheduleFunctionalTest) {
    {  // 1 processor
        const std::vector<CyclesInterfaceType> cost_vector = {10000, 20000, 15000};
        const CyclesInterfaceType overhead = 100;
        const CyclesInterfaceType result = VPUNN::dpu_schedule(1, cost_vector, overhead);

        EXPECT_EQ(result, 10000 + overhead + 20000 + overhead + 15000 + overhead);
    }
    {  // 2 processors
        const std::vector<CyclesInterfaceType> cost_vector = {10000, 20000, 15000};
        const CyclesInterfaceType overhead = 100;
        const CyclesInterfaceType result = VPUNN::dpu_schedule(2, cost_vector, overhead);

        EXPECT_EQ(result, 10000 + overhead + 15000 + overhead);
    }
    {  // 4 processors
        const std::vector<CyclesInterfaceType> cost_vector = {10000, 20000, 15000, 45000, 30000,
                                                              10000, 2000,  23500, 37200};
        const CyclesInterfaceType overhead = 1000;
        const CyclesInterfaceType result = VPUNN::dpu_schedule(4, cost_vector, overhead);

        EXPECT_EQ(result, 15000 + overhead + 10000 + overhead + 37200 + overhead);
    }
}

TEST_F(CycleAdderCheckerTest, CastingTesting) {
    {
        long long maxV = std::numeric_limits<CyclesInterfaceType>::max();
        long long int_test = maxV + 1;
        double float_test = std::numeric_limits<CyclesInterfaceType>::max() + 1.1;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(float_test))
                << "Float number exceeding the upper limit";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(int_test))
                << "Int number exceeding the upper limit";
    }
    {
        long long int_test = -1;
        double float_test = -1.1;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(float_test))
                << "Float number exceeding the lower limit";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(int_test))
                << "Int number exceeding the lower limit";
    }
    {
        long long int_test = 1000;
        double float_test = 1000.1;
        EXPECT_EQ(1001, Cycles::toCycleInterfaceType(float_test)) << "Float number in range";
        EXPECT_EQ(1000, Cycles::toCycleInterfaceType(int_test)) << "Int number exceeding in range";
    }
    {
        long long int_test = V(Cycles::ERROR_TILE_OUTPUT);
        double float_test = V(Cycles::ERROR_TILE_OUTPUT) + 0.1;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(float_test))
                << "Float number in error range";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(int_test))
                << "Int number in error range";
    }
    {
        int int_test = -1;
        float float_test = -1.0;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(float_test))
                << "Float number in error range";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(int_test))
                << "Int number in error range";
    }
    {
        int int_test = -10000;
        float float_test = -100000.0;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(float_test))
                << "Float number in error range";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(int_test))
                << "Int number in error range";
    }
    {
        char pozSigned = 100;
        char negSigned = -100;
        unsigned char pozUnsigned = 200;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negSigned))
                << "Negative signed char";
        EXPECT_EQ(100, Cycles::toCycleInterfaceType(pozSigned)) << "Positive signed char";
        EXPECT_EQ(200, Cycles::toCycleInterfaceType(pozUnsigned)) << "Unsigned char";
    }
    {
        short int pozSigned = 30000;
        short int negSigned = -30000;
        unsigned short int pozUnsigned = 65000;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negSigned))
                << "Negative signed short int";
        EXPECT_EQ(30000, Cycles::toCycleInterfaceType(pozSigned)) << "Positive signed short int";
        EXPECT_EQ(65000, Cycles::toCycleInterfaceType(pozUnsigned)) << "Unsigned short int";
    }
    {
        const long int pozSigned{2000000000};
        const long int negSigned{-2000000000};

        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negSigned))
                << "Negative signed long int";

        EXPECT_EQ(2000000000, Cycles::toCycleInterfaceType(pozSigned)) << "Positive signed long int :" << pozSigned;
        EXPECT_FALSE(pozSigned < 0) << "subzero Positive signed long int?";
        EXPECT_EQ(2000000000L, pozSigned) << "is expected? Positive signed long int";
    }
    {
        long long int pozSigned = 4000000000;
        signed long long int negSigned = -400000000;
        unsigned long long int pozUnsigned = 4000000000;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negSigned))
                << "Negative signed long long";
        EXPECT_EQ(4000000000, Cycles::toCycleInterfaceType(pozSigned)) << "Positive signed long long";
        EXPECT_EQ(4000000000, Cycles::toCycleInterfaceType(pozUnsigned)) << "Unsigned long long";

        long long int signedOverLimit = 40000000000;
        unsigned long long int unsignedOverLimit = 40000000000;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(signedOverLimit))
                << "Negative signed long long over limit";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(unsignedOverLimit))
                << "Negative signed leng long over limit";
    }
    {
        float negFloat = -100000.1f;
        float pozFloat = 100000.1f;
        float pozFloatOverLimit = 5000000000.2f;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negFloat))
                << "Negative float";
        EXPECT_EQ(100001, Cycles::toCycleInterfaceType(pozFloat)) << "Pozitive float";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(pozFloatOverLimit))
                << "Pozitive float over limit";
    }
    {
        double negdouble = -100000.1;
        double pozdouble = 100000.1;
        double pozdoubleOverLimit = 5000000000.2;
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negdouble))
                << "Negative double";
        EXPECT_EQ(100001, Cycles::toCycleInterfaceType(pozdouble)) << "Pozitive double";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(pozdoubleOverLimit))
                << "Pozitive double over limit";
    }
    {
        long double negdouble = static_cast<long double>(-100000.1);
        long double pozdouble = static_cast<long double>(100000.1);
        long double pozdoubleOverLimit = static_cast<long double>(5000000000.2);
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(negdouble))
                << "Negative double";
        EXPECT_EQ(100001, Cycles::toCycleInterfaceType(pozdouble)) << "Pozitive double";
        EXPECT_EQ(V(Cycles::ERROR_INVALID_CONVERSION_TO_CYCLES), Cycles::toCycleInterfaceType(pozdoubleOverLimit))
                << "Pozitive double over limit";
    }
}
TEST_F(CycleAdderCheckerTest, CastingTestingSpecial) {
    // the following 2 (commented out) are failing on Florins machine when building in release VS2022 (only).
    //{
    //    const long int pozSigned{1L};

    //    const auto cyc{Cycles::toCycleInterfaceType<long int>(pozSigned)};
    //    EXPECT_EQ(1L, cyc) << "Cycles :" << cyc << " Error meaning: " << Cycles::toErrorText(cyc)
    //                                << ", POZ: " << pozSigned << " Type : " << typeid(pozSigned).name();
    //}

    //{
    //    const int pozSigned{1};

    //    const auto cyc{Cycles::toCycleInterfaceType<int>(pozSigned)};
    //    EXPECT_EQ(1, cyc) << "Cycles :" << cyc << " Error meaning: " << Cycles::toErrorText(cyc)
    //                               << ", POZ: " << pozSigned << " Type : " << typeid(pozSigned).name();
    //}

    {
        constexpr long int pozSigned{2000000000L};

        constexpr auto cyc{Cycles::toCycleInterfaceType(pozSigned)};
        EXPECT_EQ(2000000000L, cyc) << "constexpr Cycles :" << cyc << " Error meaning: " << Cycles::toErrorText(cyc)
                                    << ", POZ: " << pozSigned << " Type : " << typeid(pozSigned).name();
    }

    {
        constexpr int pozSigned{2000000000};

        constexpr auto cyc{Cycles::toCycleInterfaceType(pozSigned)};
        EXPECT_EQ(2000000000, cyc) << "constexpr Cycles :" << cyc << " Error meaning: " << Cycles::toErrorText(cyc)
                                   << ", POZ: " << pozSigned << " Type : " << typeid(pozSigned).name();
    }
}
TEST_F(CycleAdderCheckerTest, CyclesTypeCastCheck) {
    {
        // testing the overload for the CyclesInterfaceType
        const CyclesInterfaceType noErrorValue = 1000;
        const CyclesInterfaceType errorValue = V(Cycles::ERROR_TILE_OUTPUT);
        EXPECT_EQ(1000, Cycles::toCycleInterfaceType(noErrorValue)) << "Number expected to remain the same";
        EXPECT_EQ(V(Cycles::ERROR_TILE_OUTPUT), Cycles::toCycleInterfaceType(errorValue))
                << "Error expected to be kept if in the error range";
    }
    {
        // since CyclesInterfaceType is an alias for uint_32 we should get the same numbers
        const uint32_t noErrorValue = 1000;
        const uint32_t errorValue = V(Cycles::ERROR_TILE_OUTPUT);
        EXPECT_EQ(1000, Cycles::toCycleInterfaceType(noErrorValue)) << "Number expected to remain the same";
        EXPECT_EQ(V(Cycles::ERROR_TILE_OUTPUT), Cycles::toCycleInterfaceType(errorValue))
                << "Error expected to be kept if in the error range";
    }
}

}  // namespace VPUNN_unit_tests