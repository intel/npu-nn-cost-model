// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/cache.h"
#include "core/persistent_cache.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <random>
#include <vector>
#include "core/serializer.h"
#include "vpu/compatibility/types11.h"
#include "vpu/compatibility/types14.h"
#include "vpu_cost_model.h"

/// @brief namespace for Unit tests of the C++ library
namespace VPUNN_unit_tests {
using namespace VPUNN;

class VPUNNCacheTest : public testing::Test {
public:
protected:
    void SetUp() override {
    }
    VPUNN::VPUCostModel small_cache_model{std::string(""), false, 10};
    VPUNN::VPUCostModel no_cache_model{std::string(""), false, 0};

    using DPU_LRU_Cache = LRUCache<std::vector<float>, float>;  ///< DPU Cache, ex LRUCache of float
};
// Demonstrate some basic assertions.
TEST_F(VPUNNCacheTest, BasicAssertions) {
    VPUNN::DPUWorkload wl = {
            VPUNN::VPUDevice::VPU_2_7,
            VPUNN::Operation::CONVOLUTION,
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // input dimensions
            {VPUNN::VPUTensor(56, 56, 16, 1, VPUNN::DataType::UINT8)},  // output dimensions
            {3, 3},                                                     // kernels
            {1, 1},                                                     // strides
            {1, 1},                                                     // padding
            VPUNN::ExecutionMode::CUBOID_16x16                          // execution mode
    };

    auto dpu_cycles = small_cache_model.DPU(wl);

    for (auto idx = 0; idx < 100; idx++) {
        // Testing caching
        EXPECT_EQ(dpu_cycles, small_cache_model.DPU(wl));
        // Testing correctness
        EXPECT_EQ(dpu_cycles, no_cache_model.DPU(wl));
    }
}
// Demonstrate some basic assertions.
TEST_F(VPUNNCacheTest, CacheBasicTest) {
    DPU_LRU_Cache cache(1 /*, 0*/, "");
    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> v1(100), v2(100);
    for (auto idx = 0; idx < 100; idx++) {
        // Generate a random vector and val
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> rand_gen(-1.0f, 1.0f);
        auto random_float = [&]() {
            return rand_gen(gen);
        };
        std::generate(v1.begin(), v1.end(), random_float);
        std::generate(v2.begin(), v2.end(), random_float);
        auto val1 = random_float();
        auto val2 = random_float();

        // Testing that there is no vector
        EXPECT_FALSE(cache.get(v1));
        EXPECT_FALSE(cache.get(v2));

        cache.add(v1, val1);

        EXPECT_EQ(*cache.get(v1), val1);
        EXPECT_FALSE(cache.get(v2));

        cache.add(v2, val2);
        EXPECT_FALSE(cache.get(v1));
        EXPECT_EQ(*cache.get(v2), val2);
    }
}

TEST_F(VPUNNCacheTest, CacheBasicTest_ext) {
    DPU_LRU_Cache cache(1 /*, 0*/, "");  // one position
    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> v1(100), v2(100);
    // for (auto idx = 0; idx < 100; idx++) {
    // Generate a random vector and val
    {
        std::fill(v1.begin(), v1.end(), 1.0f);
        std::fill(v2.begin(), v2.end(), 2.0f);
    }
    auto val1 = 101.0f;
    auto val2 = 102.0f;

    // Testing that there is no vector
    EXPECT_FALSE(cache.get(v1));
    EXPECT_FALSE(cache.get(v2));

    EXPECT_NO_THROW(cache.add(v1, val1));
    EXPECT_NO_THROW(cache.add(v1, val1));//second add 

    EXPECT_NO_THROW(cache.get(v1));

    EXPECT_EQ(*cache.get(v1), val1);
    EXPECT_FALSE(cache.get(v2));

    EXPECT_NO_THROW(cache.add(v2, val2));
    EXPECT_FALSE(cache.get(v1));
    EXPECT_EQ(*cache.get(v2), val2);
    //}
}

//------

class VPUNNCachePreloadedTest : public testing::Test {
public:
protected:
    void SetUp() override {
    }

    static auto mkhsh(const std::vector<float>& desc) {
        return NNDescriptor(desc).hash();
    }

    const DPUWorkload wl_conv1 = {
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                       // kernels
            {1, 1},                                       // strides
            {1, 1, 1, 1},                                 // padding
            ExecutionMode::CUBOID_16x16,                  // execution mode
            ActivationFunction::NONE,                     ///< operation activation function
            0.0f,                                         ///< activation sparsity
            0.0f,                                         ///< weight sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},         // input
            {Swizzling::KEY_5},                           // output
    };
    const DPUWorkload wl_conv2 = {
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 57, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(56, 57, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                         // kernels
            {1, 1},                                         // strides
            {1, 1, 1, 1},                                   // padding
            ExecutionMode::CUBOID_8x16,                     // execution mode
            ActivationFunction::NONE,                       ///< operation activation function
            0.0f,                                           ///< activation sparsity
            0.0f,                                           ///< weight sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},           // input
            {Swizzling::KEY_0},                             // output
    };
    const DPUWorkload wl_conv3 = {
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(59, 24, 16, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(59, 24, 16, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                         // kernels
            {1, 1},                                         // strides
            {1, 1, 1, 1},                                   // padding
            ExecutionMode::CUBOID_4x16,                     // execution mode
            ActivationFunction::NONE,                       ///< operation activation function
            0.0f,                                           ///< activation sparsity
            0.0f,                                           ///< weight sparsity
            {Swizzling::KEY_5, Swizzling::KEY_0},           // input
            {Swizzling::KEY_0},                             // output
    };

    std::string clean_csv_exension(const std::string& s) {
        std::filesystem::path csv_normalizer{s};
        csv_normalizer.replace_extension();
        return csv_normalizer.string();
    }

    template <class NNAdapter>
    bool isCompatibleWithTrainedSpace(const DPUWorkload& wl) {
        // is compatible with what the NN can predict? Otherwise frofiling is for an out of scope WL, that
        // will replace a good one with the same descriptor (eg swizzling)
        const std::tuple<Swizzling, Swizzling, Swizzling> initial_swizz{wl.input_swizzling[0], wl.input_swizzling[1],
                                                                        wl.output_swizzling[0]};

        const auto swizz = NNAdapter::establishUniqueSwizzling(wl.input_swizzling[0], wl.input_swizzling[1],
                                                               wl.output_swizzling[0], wl.op);

        if (initial_swizz != swizz) {
            // look at maxpool special condition
            if (NNAdapter::weights_always_zero.count(wl.op) != 0) {  // not influencing runtime
                if ((std::get<0>(initial_swizz) == std::get<0>(swizz)) &&
                    (std::get<2>(initial_swizz) == std::get<2>(swizz))) {
                    // weigths do not matrter(always 0), only in /out swizzling influences
                    return true;
                }
            }
            return false;
        }

        return true;
    }

    static float convertGTFromCSV(const std::string& gt) {
        float gt_value{0.0f};
        try {
            gt_value = std::stof(gt);  // the same op as in descriptor transformers
        } catch (const std::exception&) {
            gt_value = -1.0f;  // negative
        }

        return gt_value;
    }

    const std::string gtSPRUN_tag{"gt_superrun"};

    const std::string cache_file_40{
            (std::filesystem::path{VPU_4_0_MODEL_PATH}).replace_extension("cache_bin").string()};
    const VPUCostModel model_40{VPU_4_0_MODEL_PATH};  // default 4 model
    Preprocessing_Interface4011<float> pp_4011;       // used interface where we have cache

    const std::string cache_file_41{
            (std::filesystem::path{VPU_4_1_MODEL_PATH}).replace_extension("cache_bin").string()};
    const VPUCostModel model_41{VPU_4_1_MODEL_PATH};  // default 4 model
    Preprocessing_Interface4111<float> pp_4111;       // used interface where we have cache

    const size_t size_of_descriptor_INterface11{pp_4011.output_size()};  // same for all 11

#ifdef INTEL_EMBARGO_NPU5
    const std::string cache_file_51{(std::filesystem::path{NPU_5_0_MODEL_PATH}).replace_extension("cachebin").string()};
    const VPUCostModel model_51{NPU_5_0_MODEL_PATH};
    Preprocessing_Interface14<float> pp_5014;  // used interface where we have cache
    const size_t size_of_descriptor_INterface14{pp_5014.output_size()};
#endif  // INTEL_EMBARGO_NPU5

    //////
    const std::string csv_file_OneCache{"c:\\gitwrk\\CM_Profilings\\UTests\\Cache_UT\\"
                                        "all_cache_misses_iter2_profiled.csv"};
    const std::string csv_cache_misses{csv_file_OneCache};

    // const std::string csv_file_OneCache{"c:\\Users\\fistoc\\OneDrive - Intel Corporation"
    //                                     "\\Models_logs\\logs\\abedekarvpunn-1.6.8-tag6-9Sept\\all_mexp_9_oct\\Cache\\"
    //                                     "cache_luca.csv"};

    // const std::string csv_file_OneCache{"c:\\Users\\fistoc\\OneDrive - Intel Corporation"
    //                                     "\\Models_logs\\logs\\abedekarvpunn-1.6.8-tag6-9Sept\\all_mexp_29_oct\\Cache_1\\"
    //                                     "l1_workloads_unique_all_mexp_29_oct_profiled.csv"};

    // const std::string csv_cache_misses{"c:\\Users\\fistoc\\OneDrive - Intel Corporation"
    //                                    "\\Models_logs\\logs\\abedekarvpunn-1.6.8-tag6-9Sept\\all_mexp_29_oct\\Cache_1\\"
    //                                    "cache_misses37144.csv"};

    // const std::string csv_file_OneCache{"c:\\Users\\fistoc\\OneDrive - Intel Corporation"
    //                                     "\\Models_logs\\logs\\SJvpunn-1.6.8-post8-without-extra-swizzling\\Cache_3\\"
    //                                     //"0_l1_workloads_unique_all_mexp_29_oct.csv"};
    //                                     //"1_l1_unique_mexp_procyon_profiled.csv"};
    //                                     "2_all_cache_misses_partly_profiled.csv"};

    const std::string mult_csv_folder_OneCache{
            //        "/home/abalanes/source/vpux-plugin/thirdparty/vpucostmodel/build/tests/cpp/"
            "/home/epele/mlir_serializer/cache_13/"
            //"c:\\Users\\fistoc\\OneDrive - Intel Corporation"
            //"\\Models_logs\\logs\\SJvpunn-1.6.8-post8-without-extra-swizzling\\cache_12_NN41\\"
    };  // 3, 4, 5.....

    const std::string cache_input_normal{
            //        "/home/abalanes/source/vpux-plugin/thirdparty/vpucostmodel/models/vpu_4_0.cache_bin"
            "/home/epele/mlir_serializer/cache_13/vpu_4_0.cache_bin"
            // mult_csv_folder_OneCache + "vpu_4_1.cache_bin"
    };

    const std::vector<std::string> multiple_csv_files{

            // cache 3
            // "0_l1_workloads_unique_all_mexp_29_oct.csv",  //
            // "1_l1_unique_mexp_procyon_profiled.csv",      //
            // "2_all_cache_misses_partly_profiled.csv",

            // cache 4
            //"l1_workloads_unique_all_mexp_29_oct.csv",  // first
            //"l1_unique_mexp_procyon.csv",
            //"all_cache_misses.csv",
            //"all_cache_misses_iter_1.csv",

            // cache5
            //"all_cache_misses_iter2_profiled.csv",

            // cache7
            // "all_cache_misses_iter_3_augmented_profiled.csv",

            // cache8
            //"all_cache_misses_iter_4_augmented_profiled.csv",

            // cache9
            //"all_cache_misses_iter5_augmented_profiled.csv",

            // // cache10
            // "all_cache_misses_iter6_augmented_profiled.csv",

            // cache11
            // "all_cache_misses_iter7_augmented_profiled.csv",

            // cache12
            //"all_cache_misses_iter8_augmented_profiled.csv",

            // cache13
            "all_cache_misses_iter9_mixed_swizz_profiling.csv",
    };

    const float four_bilion{4000000000.f};
    const float cache_delta_tolerance{100.f};

    struct OneCacheConfig {  // One Cache settings, affects all the "OneCache" tests

        const VPUCostModel& sanitizerModel /*{model_40}*/;
        const std::string NN_CM_to_use /*{VPU_4_0_MODEL_PATH}*/;
        const std::string csv_file /*{csv_file_OneCache}*/;
        const std::string cache_file_input /*{cache_input_normal}*/;
    };

    struct OneCacheNormal : public OneCacheConfig {  // One Cache settings, affects all the "OneCache" tests
        using pp_adapt_forWrite = NN40InputAdapter;
        using pp_adapt_forRead = NN40InputAdapter;

        Preprocessing_Interface4011<float>& preprop_forWrite /*{pp_4011}*/;
        Preprocessing_Interface4011<float>& preprop_forRead /*{pp_4011}*/;
    };
    OneCacheNormal normalConfig{
            {model_40, VPU_4_0_MODEL_PATH, csv_file_OneCache, cache_input_normal},
            pp_4011,
            pp_4011,
    };  // normal config

    // configuration affecting TRansparentSwizzling
    struct OneCacheTRansparentSwizzling :
            public OneCacheConfig {  // One Cache settings, affects all the "OneCache" tests
        using pp_adapt_forWrite = NN41InputAdapter;
        using pp_adapt_forRead = NN41InputAdapter;

        Preprocessing_Interface4111<float>& preprop_forWrite /*{pp_4111}*/;
        Preprocessing_Interface4111<float>& preprop_forRead /*{pp_4111}*/;
    };

    OneCacheTRansparentSwizzling transparent_SwizzConfig{
            {model_41, VPU_4_1_MODEL_PATH, csv_file_OneCache, cache_input_normal},
            pp_4111,
            pp_4111,
    };  // transparent swizzling config
};

TEST_F(VPUNNCachePreloadedTest, EmptyBasicTest) {
    FixedCache empty_cache{/*size_of_descriptor_INterface11,*/ ""};
    const auto& theMap{empty_cache.getMap()};
    EXPECT_EQ(pp_4011.output_size(), 93);
    EXPECT_EQ(theMap.size(), 0);
}

TEST_F(VPUNNCachePreloadedTest, FolderBasicTest) {
    std::string model4_folder{VPU_4_0_MODEL_PATH};
    std::cout << "Model 4 folder is : " << model4_folder << std::endl;

    auto pathC = std::filesystem::current_path();  // getting path
    std::cout << "Current path is : " << pathC << std::endl;

    std::filesystem::path pathM = model4_folder;
    std::cout << "PathMOdel: " << pathM << " Has FilenameL " << pathM.has_filename() << std::endl;
    // pathM.remove_filename();
    // std::cout << "PathMOdelFOlder: " << pathM << " Has Filename: " << pathM.has_filename() << std::endl;
    // pathM /= "cache.bin";
    pathM.replace_extension("cache_bin");
    std::cout << "PathModelCache: " << pathM << " Has Filename: " << pathM.has_filename()
              << " Exists: " << std::filesystem::exists(pathM) << std::endl;

    const auto cache_file{pathM.string()};

    ASSERT_TRUE(std::filesystem::exists(cache_file)) << cache_file;
    ASSERT_TRUE(std::filesystem::exists(cache_file_40)) << cache_file_40;
}

TEST_F(VPUNNCachePreloadedTest, SmokeBasicTest) {
    auto& preprop_now = pp_4011;  // irelevant since we create our own cache
    FixedCache the_cache{/*size_of_descriptor_INterface11,*/ ""};
    auto pathC = std::filesystem::current_path();  // getting path
    std::cout << "Current path is : " << pathC << std::endl;

    const std::string test_cache_file{"test_cache.bin"};

    // DPUWorkload wl;  // a wl

    // generate a descriptor for the wl
    const std::vector<float> desc1{preprop_now.transformSingle(wl_conv1)};
    const std::vector<float> desc2{preprop_now.transformSingle(wl_conv2)};
    const std::vector<float> desc3{preprop_now.transformSingle(wl_conv3)};

    ASSERT_EQ(desc1.size(), size_of_descriptor_INterface11);
    ASSERT_EQ(desc2.size(), size_of_descriptor_INterface11);
    ASSERT_EQ(desc3.size(), size_of_descriptor_INterface11);

    {
        the_cache.insert(mkhsh(desc1), 1.0f);
        ASSERT_TRUE(the_cache.contains(mkhsh(desc1)));
        ASSERT_FALSE(the_cache.contains(mkhsh(desc2)));
        ASSERT_FALSE(the_cache.contains(mkhsh(desc3)));

        the_cache.insert(mkhsh(desc2), 2.0f);
        ASSERT_TRUE(the_cache.contains(mkhsh(desc1)));
        ASSERT_TRUE(the_cache.contains(mkhsh(desc2)));

        // EXPECT_TRUE(the_cache.get_pointer(mkhsh(desc1)) != nullptr);
        EXPECT_TRUE(the_cache.get(mkhsh(desc1)).has_value());

        EXPECT_EQ(*the_cache.get(mkhsh(desc1)), 1.0f);
        EXPECT_EQ(*the_cache.get(mkhsh(desc2)), 2.0f);

        std::filesystem::path fileOne{pathC /= test_cache_file};
        std::cout << "Current test cache is : " << fileOne << std::endl;
        std::cout << "File exists?: " << std::filesystem::exists(fileOne) << std::endl;

        if (std::filesystem::exists(fileOne)) {
            // delete it
            std::filesystem::remove(fileOne);
        }

        EXPECT_TRUE(the_cache.write_cache(test_cache_file));
    }

    {
        const FixedCache read_cache{test_cache_file};
        // read_cache.deserializeCacheFromFile("test_cache.bin");

        ASSERT_TRUE(read_cache.contains(mkhsh(desc1)));
        ASSERT_TRUE(read_cache.contains(mkhsh(desc2)));

        EXPECT_EQ(*read_cache.get(mkhsh(desc1)), 1.0f);
        EXPECT_EQ(*read_cache.get(mkhsh(desc2)), 2.0f);
    }

    // check also  reading with the data pointer interface

    {  // open the file in binary mode and reads its full content in a vector
        std::ifstream file(test_cache_file, std::ios::binary);
        // check file is open
        ASSERT_TRUE(file.is_open()) << "Error opening file for binary read: " << test_cache_file << std::endl;
        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});
        file.close();
        ASSERT_GE(buffer.size(), 68)  // just a value to check if the file is not empty and has 2 values
                << "Empty file?: " << test_cache_file << ", Size: " << buffer.size() << std::endl;

        {
            // get the raw data from the vector
            const char* data = buffer.data();
            const size_t size = buffer.size();

            const FixedCache read_cache{data, size};

            ASSERT_TRUE(read_cache.contains(mkhsh(desc1)));
            ASSERT_TRUE(read_cache.contains(mkhsh(desc2)));
            ASSERT_FALSE(read_cache.contains(mkhsh(desc3)));

            EXPECT_EQ(*read_cache.get(mkhsh(desc1)), 1.0f);
            EXPECT_EQ(*read_cache.get(mkhsh(desc2)), 2.0f);

            const auto& cacheStats{read_cache.getCounter()};
            // print cache statistics
            std::cout << "\nCache statistics for Binary load section: " << cacheStats.printString() << std::endl;
            cacheStats.reset();
            std::cout << "\nCache statistics After Reset " << cacheStats.printString() << std::endl;
        }
    }
    // EXPECT_TRUE(false);
}

#ifdef INTEL_EMBARGO_NPU5
TEST_F(VPUNNCachePreloadedTest, DISABLED_SearchTimeTest) {
    auto& preprop_now = pp_5014;
    FixedCache the_cache(cache_file_51);
    std::cout << "Cache size: " << the_cache.getMap().size() << std::endl;

    // generate a descriptor for the wl
    const std::vector<float> desc1{preprop_now.transformSingle(wl_conv1)};
    const std::vector<float> desc2{preprop_now.transformSingle(wl_conv2)};
    const std::vector<float> desc3{preprop_now.transformSingle(wl_conv3)};

    ASSERT_EQ(desc1.size(), size_of_descriptor_INterface14);
    ASSERT_EQ(desc2.size(), size_of_descriptor_INterface14);
    ASSERT_EQ(desc3.size(), size_of_descriptor_INterface14);

    {
        the_cache.insert(mkhsh(desc1), 1.0f);
        ASSERT_TRUE(the_cache.contains(mkhsh(desc1)));
        ASSERT_FALSE(the_cache.contains(mkhsh(desc2)));
        ASSERT_FALSE(the_cache.contains(mkhsh(desc3)));

        the_cache.insert(mkhsh(desc2), 2.0f);
        ASSERT_TRUE(the_cache.contains(mkhsh(desc1)));
        ASSERT_TRUE(the_cache.contains(mkhsh(desc2)));

        // EXPECT_TRUE(the_cache.get_pointer(mkhsh(desc1)) != nullptr);
        EXPECT_TRUE(the_cache.get(mkhsh(desc1)).has_value());

        EXPECT_EQ(*the_cache.get(mkhsh(desc1)), 1.0f);
        EXPECT_EQ(*the_cache.get(mkhsh(desc2)), 2.0f);
    }

    const int num_iterations{10000};

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        the_cache.get(fnv1a_hash(desc1));
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    auto avg_duration = static_cast<double>(duration) / num_iterations;
    std::cout << "Average search time for " << num_iterations << " iterations using new direct hash: " << avg_duration
              << " microseconds" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        the_cache.get(fnv1a_hash(vec2int_str(desc1)));
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    avg_duration = static_cast<double>(duration) / num_iterations;
    std::cout << "Average search time for " << num_iterations
              << " iterations using old vec2int and hash: " << avg_duration << " microseconds" << std::endl;

    const auto hash1 = fnv1a_hash(desc1);
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        the_cache.get(hash1);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    avg_duration = static_cast<double>(duration) / num_iterations;
    std::cout << "Average search time for " << num_iterations << " iterations precomputed hash: " << avg_duration
              << " microseconds" << std::endl;

    EXPECT_TRUE(false);
}

TEST_F(VPUNNCachePreloadedTest, SparsityLevelsTest) {
    auto& preprop_now = pp_5014;
    FixedCache the_cache(cache_file_51);
    std::cout << "Cache size: " << the_cache.getMap().size() << std::endl;

    const DPUWorkload wl_1 = {
            VPUDevice::NPU_5_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                       // kernels
            {1, 1},                                       // strides
            {1, 1, 1, 1},                                 // padding
            ExecutionMode::CUBOID_16x16,                  // execution mode
            ActivationFunction::NONE,                     ///< operation activation function
            0.0f,                                         ///< activation sparsity
            0.0f,                                         ///< weight sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},         // input
            {Swizzling::KEY_5},                           // output
    };

    const DPUWorkload wl_1_sparse = {
            VPUDevice::NPU_5_0,
            Operation::CONVOLUTION,
            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // input dimensions
            {VPUTensor(56, 56, 16, 1, DataType::UINT8)},  // output dimensions
            {3, 3},                                       // kernels
            {1, 1},                                       // strides
            {1, 1, 1, 1},                                 // padding
            ExecutionMode::CUBOID_16x16,                  // execution mode
            ActivationFunction::NONE,                     ///< operation activation function
            0.5f,                                         ///< activation sparsity
            0.5f,                                         ///< weight sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},         // input
            {Swizzling::KEY_5},                           // output
    };

    // generate a descriptor for the wl
    const std::vector<float> desc1{preprop_now.transformSingle(wl_1)};
    const std::vector<float> desc1_sparse{preprop_now.transformSingle(wl_1_sparse)};

    const auto wl_1_hash = mkhsh(desc1);
    const auto wl_1_sparse_hash = mkhsh(desc1_sparse);

    ASSERT_NE(wl_1_hash, wl_1_sparse_hash);
}
#endif  // INTEL_EMBARGO_NPU5

// Foirethis ro work we need a Good cache file for DPU in the models files
TEST_F(VPUNNCachePreloadedTest, DISABLED_SmokeSparsityFloatBasicTest_40) {
    auto& preprop_now = pp_4011;  // OLD, must associate with the cache you want to use now!

    std::filesystem::path pathNN = normalConfig.NN_CM_to_use;  // teh vpunn
    std::filesystem::path cache_file_now{pathNN.replace_extension("cache_bin")};
    std::cout << "Current test cache is : " << cache_file_now << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_now) << std::endl << std::endl;

    const std::string test_cache_file{cache_file_now.string()};

    const FixedCache the_cache{test_cache_file};
    EXPECT_TRUE(the_cache.getMap().size() > 0);

    // DPUWorkload wl;  // a wl

    const float f0{0.925926f};
    const std::string f0_str{"0.925926"};
    EXPECT_EQ(std::to_string(f0), f0_str);

    const float f1{0.925926030f};
    const float f2{0.925926089f};

    EXPECT_EQ(f0, f1);
    EXPECT_NE(f0, f2);

    EXPECT_EQ(std::to_string(f1), std::to_string(f0));
    EXPECT_EQ(std::to_string(f2), std::to_string(f0));

    const float f1_norm{std::stof(std::to_string(f1))};
    const float f2_norm{std::stof(std::to_string(f1))};

    EXPECT_EQ(f0, f1_norm);
    EXPECT_EQ(f0, f2_norm);
    EXPECT_EQ(f2_norm, f1_norm);

    const DPUWorkload convSparse1 = {
            VPUDevice::VPU_4_0,
            Operation::CONVOLUTION,
            {VPUTensor(500, 17, 48, 1, DataType::FLOAT16)},  // input dimensions
            {VPUTensor(500, 16, 32, 1, DataType::FLOAT16)},  // output dimensions
            {3, 3},                                          // kernels
            {1, 1},                                          // strides
            {1, 0, 1, 1},                                    // padding TBLR
            ExecutionMode::CUBOID_8x16,                      // execution mode
            ActivationFunction::NONE,                        ///< operation activation function
            0.0f,                                            ///< activation sparsity
            f1,                                              ///< weight sparsity
            {Swizzling::KEY_5, Swizzling::KEY_5},            // input
            {Swizzling::KEY_5},                              // output
            1,                                               // owt
            {0, 0, 0, 0},                                    // offs
            ISIStrategy::CLUSTERING,                         // strategy
            true,                                            // sparsity enabled

    };
    DPUWorkload convSparse2{convSparse1};
    convSparse2.weight_sparsity = f2;

    std::cout << "WTS sparsity  values: " << convSparse1.weight_sparsity << " , " << convSparse2.weight_sparsity
              << std::endl;

    ASSERT_NE(convSparse1.weight_sparsity, convSparse2.weight_sparsity);

    // generate a descriptor for the wl
    const std::vector<float> desc1{preprop_now.transformSingle(convSparse1)};
    const std::vector<float> desc2{preprop_now.transformSingle(convSparse2)};

    ASSERT_EQ(desc1.size(), size_of_descriptor_INterface11);
    ASSERT_EQ(desc2.size(), size_of_descriptor_INterface11);

    {
        EXPECT_TRUE(the_cache.contains(mkhsh(desc1))) << "FLoat 1 does not hit";
        EXPECT_TRUE(the_cache.contains(mkhsh(desc2))) << "FLoat 2 does not hit";

        EXPECT_EQ(*the_cache.get(mkhsh(desc1)), *the_cache.get(mkhsh(desc2)));

        const float v1 = *the_cache.get(mkhsh(desc1));
        const float v2 = *the_cache.get(mkhsh(desc2));

        std::cout << "Cache values: " << v1 << " , " << v2 << std::endl;
    }

    // EXPECT_TRUE(false);
}

// generate a cache froma csv.
// can also append to existing cache
TEST_F(VPUNNCachePreloadedTest, DISABLED_CsvGenerate_OneCache) {
    OneCacheNormal& test_config{normalConfig};
    FixedCache the_cache{/*test_config.preprop_forWrite.output_size(),*/ ""};

    Serializer<FileFormat::CSV> serializer_IN{true};                                         // force enable
    serializer_IN.initialize(clean_csv_exension(test_config.csv_file), FileMode::READONLY);  // open file, basic fields
    EXPECT_TRUE(serializer_IN.is_initialized());

    DPUOperation wl_buff;
    SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
    Series readFieldsAll{};

    std::cout << " \n initial Cache contains  :" << the_cache.getMap().size()
              << " entries, CSV file: " << test_config.csv_file << " \n\n";

    bool allGood{true};
    // iterate in CSV
    int i = 1;
    std::cout << "Pos starts from " << i << " \n";
    while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
        auto dpu_wl = wl_buff.clone_as_DPUWorkload();
        const std::string gt{gt_SPRUN_buff.value};
        const float gt_value{(float)std::atof(gt.c_str())};
        if (gt.length() <= 0) {
            std::cout << "NO GT at pos: " << i << std::endl;
        } else {
            EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i << std::endl;
            if (gt_value > 0 && gt_value < four_bilion) {
                SanityReport sanityInfo{};
                // sanitize with the same NN version as fro what you want to generate
                const auto sane = test_config.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

                if (sane) {
                    // is compatible with what the NN can predict? Otherwise profiling is for an out of scope WL, that
                    // will replace a good one with the same descriptor (eg swizzling)
                    const auto isRelevant = isCompatibleWithTrainedSpace<
                            std::remove_reference_t<decltype(test_config)>::pp_adapt_forWrite>(dpu_wl);  // maybe not?
                    // we shuld trandorm loosly, let it pass, even if is not the ione that will be searched in reality
                    // we should generate also teh one that is in reality in case runtime is teh same(int/float fro wts)
                    const std::vector<float> descriptorNN{
                            test_config.preprop_forWrite.transformSingle(dpu_wl)};  // a specific processor
                    const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));
                    if (cache_hit) {
                        const auto prevVal = the_cache.get(mkhsh(descriptorNN)).value();
                        const auto delta = abs(gt_value - prevVal);
                        std::cout << "Cache already contains value at pos: " << i << " , New value: " << gt_value
                                  << " , Prev  value: " << prevVal << ", Delta: " << delta
                                  << "  WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                  << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                  << std::endl;

                        allGood = false;
                    }
                    if (isRelevant) {
                        if (!cache_hit) {
                            the_cache.insert(mkhsh(descriptorNN), gt_value);
                        } else {
                            std::cout << "   ------hit! value not set: " << gt_value << std::endl;
                        }
                    } else {
                        std::cout << "WL NOT RELEVANT value at pos: " << i << " ,Not adding to cache "
                                  << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                  << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                  << std::endl;
                        allGood = false;
                    }
                } else {
                    const auto err{sanityInfo.value()};
                    std::cout << "Sanity failed at pos: " << i << "  Code: " << err
                              << " meaning: " << Cycles::toErrorText(err) << std::endl;
                }
            } else {
                std::cout << "Negative or zero, or big GT at pos: " << i << " , GT: " << gt_value << std::endl;
                allGood = false;
            }
        }

        ++i;  // finally
    }

    std::cout << "\n Collected :" << i << " workloads from csv \n";
    std::cout << " Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    std::filesystem::path csv_file_now{test_config.csv_file};
    std::filesystem::path cache_file_now{csv_file_now.replace_extension("cache_bin")};
    std::cout << "Current test cache is : " << cache_file_now << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_now) << std::endl << std::endl;

    // ASSERT_FALSE(std::filesystem::exists(cache_file_now)) << "ABORT: Cannot overwrite: " << cache_file_now;
    // bool const appendIfExists{false};
    EXPECT_TRUE(the_cache.write_cache(cache_file_now.string() /*appendIfExists*/));

    ASSERT_TRUE(allGood);
}

// read a cache and add to it the csv file data
TEST_F(VPUNNCachePreloadedTest, DISABLED_CsvGenerateExtented_OneCache) {
    std::filesystem::path cache_file_input{normalConfig.cache_file_input};
    cache_file_input.replace_extension("cache_bin");

    std::cout << "Current test cache is : " << cache_file_input << ", CSV file: " << cache_file_input << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_input) << std::endl << std::endl;

    ASSERT_TRUE(std::filesystem::exists(cache_file_input))
            << "ABORT: Cannot read NON existent cache: " << cache_file_input;

    FixedCache the_cache{""};
    EXPECT_TRUE(the_cache.read_cache(cache_file_input.string()));

    Serializer<FileFormat::CSV> serializer_IN{true};                                          // force enable
    serializer_IN.initialize(clean_csv_exension(normalConfig.csv_file), FileMode::READONLY);  // open file, basic fields
    EXPECT_TRUE(serializer_IN.is_initialized());

    DPUOperation wl_buff;
    SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
    Series readFieldsAll{};

    std::cout << " \n initial Cache contains  :" << the_cache.getMap().size()
              << " entries, CSV file: " << normalConfig.csv_file << " \n\n";

    bool allGood{true};
    // iterate in CSV
    int i = 1;
    std::cout << "Pos starts from " << i << " \n";
    while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
        auto dpu_wl = wl_buff.clone_as_DPUWorkload();
        const std::string gt{gt_SPRUN_buff.value};
        const float gt_value{(float)std::atof(gt.c_str())};
        if (gt.length() <= 0) {
            std::cout << "NO GT at pos: " << i << std::endl;
        } else {
            EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i << std::endl;
            if (gt_value > 0 && gt_value < four_bilion) {
                SanityReport sanityInfo{};
                // sanitize with the same NN version as fro what you want to generate
                const auto sane = normalConfig.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

                if (sane) {
                    // is compatible with what the NN can predict? Otherwise profiling is for an out of scope WL, that
                    // will replace a good one with the same descriptor (eg swizzling)
                    const auto isRelevant = isCompatibleWithTrainedSpace<decltype(normalConfig)::pp_adapt_forWrite>(
                            dpu_wl);  // maybe not?
                    // we shuld trandorm loosly, let it pass, even if is not the ione that will be searched in reality
                    // we should generate also teh one that is in reality in case runtime is teh same(int/float fro wts)
                    const std::vector<float> descriptorNN{
                            normalConfig.preprop_forWrite.transformSingle(dpu_wl)};  // a specific processor
                    const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));
                    if (cache_hit) {
                        const auto prevVal = the_cache.get(mkhsh(descriptorNN)).value();
                        const auto delta = abs(gt_value - prevVal);
                        std::cout << "Cache already contains value at pos: " << i << " , New value: " << gt_value
                                  << " , Prev  value: " << prevVal << ", Delta: " << delta << ".             "
                                  << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0])) << std::endl;
                        allGood = false;
                    }
                    if (isRelevant) {
                        if (!cache_hit) {
                            the_cache.insert(mkhsh(descriptorNN), gt_value);
                        } else {
                            std::cout << "   ------hit! value not set: " << gt_value << std::endl;
                        }
                    } else {
                        std::cout << "WL NOT RELEVANT value at pos: " << i << " ,Not adding to cache "
                                  << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0])) << std::endl;
                        allGood = false;
                    }
                } else {
                    const auto err{sanityInfo.value()};
                    std::cout << "Sanity failed at pos: " << i << "  Code: " << err
                              << " meaning: " << Cycles::toErrorText(err) << std::endl;
                }
            } else {
                std::cout << "Negative or zero, or big GT at pos: " << i << " , GT: " << gt_value << std::endl;
                allGood = false;
            }
        }

        ++i;  // finally
    }

    std::cout << "\n Collected :" << i << " workloads from csv \n";
    std::cout << " Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    std::filesystem::path csv_file_now{normalConfig.csv_file};
    std::filesystem::path cache_file_output{csv_file_now.replace_extension("cache_bin")};
    std::cout << "Current test cache is : " << cache_file_output << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_output) << std::endl << std::endl;

    // ASSERT_FALSE(std::filesystem::exists(cache_file_now)) << "ABORT: Cannot overwrite: " << cache_file_now;
    // bool const appendIfExists{false};
    EXPECT_TRUE(the_cache.write_cache(cache_file_output.string() /*, appendIfExists*/));

    ASSERT_TRUE(allGood);
}

/// read a cache, and add to it a list of CSV files. first data in cache stays in cache (no replace/no update policy)
// use this one(check paths first) for generating after each iteration . (swizzling 555 filtered/normal NN)
TEST_F(VPUNNCachePreloadedTest, DISABLED_Multiple_CsvGenExtented_OneCache_ACTIVE_ITERATION) {
    OneCacheNormal& test_config{normalConfig};
    std::filesystem::path cache_file_input{test_config.cache_file_input};
    cache_file_input.replace_extension("cache_bin");

    std::cout << "Current test cache is : " << cache_file_input << ", CSV file: " << cache_file_input << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_input) << std::endl << std::endl;

    ASSERT_TRUE(std::filesystem::exists(cache_file_input))
            << "ABORT: Cannot read NON existent cache: " << cache_file_input;

    FixedCache the_cache{""};
    EXPECT_TRUE(the_cache.read_cache(cache_file_input.string()));

    std::cout << " \n ***** Initial Cache contains  :" << the_cache.getMap().size() << std::endl;

    bool allGood{true};
    for (const auto& one_csv : multiple_csv_files) {
        const std::string i_csv_file = mult_csv_folder_OneCache + one_csv;

        Serializer<FileFormat::CSV> serializer_IN{true};                               // force enable
        serializer_IN.initialize(clean_csv_exension(i_csv_file), FileMode::READONLY);  // open file, basic fields
        EXPECT_TRUE(serializer_IN.is_initialized());

        DPUOperation wl_buff;
        SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
        Series readFieldsAll{};

        std::cout << " \n initial Cache contains  :" << the_cache.getMap().size()
                  << " entries, Before reading CSV file: " << i_csv_file << " \n\n";

        // iterate in CSV
        int i = 1;
        std::cout << "Pos starts from " << i << " \n";
        while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
            auto dpu_wl = wl_buff.clone_as_DPUWorkload();
            const std::string gt{gt_SPRUN_buff.value};

            const float gt_value{convertGTFromCSV(gt)};  // the same op as in descriptor transformers

            SanityReport sanityInfo{};
            // sanitize with the same NN version as fro what you want to generate
            const auto sane = test_config.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

            // is compatible with what the NN can predict? Otherwise profiling is for an out of scope WL,
            // that will replace a good one with the same descriptor (eg swizzling)
            const auto isRelevant =
                    isCompatibleWithTrainedSpace<std::remove_reference_t<decltype(test_config)>::pp_adapt_forWrite>(
                            dpu_wl);  // maybe not

            const std::vector<float> descriptorNN{
                    test_config.preprop_forWrite.transformSingle(dpu_wl)};  // a specific processor
            const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));

            if (gt.length() <= 0) {
                std::cout << "NO GT at pos: " << i << ", gt: #" << gt << "#"
                          << ", sane: " << sane << ", Relevant:" << isRelevant << ", Cache hit: " << cache_hit
                          << std::endl;
            } else {
                EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i
                                          << ", sane: " << sane << ", Relevant:" << isRelevant
                                          << ", Cache hit: " << cache_hit << ", gt: #" << gt << "#" << std::endl;
                if (gt_value > 0 && gt_value < four_bilion) {
                    if (sane) {
                        if (cache_hit) {
                            const auto prevVal = the_cache.get(mkhsh(descriptorNN)).value();
                            const auto delta = abs(gt_value - prevVal);
                            std::cout << "Cache already contains value at pos: " << i << " , New value: " << gt_value
                                      << " , Prev  value: " << prevVal << ", Delta: " << delta << ".             "
                                      << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                      << "Swizzling :"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                      << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                      << std::endl;
                            ;
                            allGood = false;
                        }
                        if (isRelevant) {
                            if (!cache_hit) {
                                the_cache.insert(mkhsh(descriptorNN), gt_value);
                            } else {
                                std::cout << "   ------hit! value not set: " << gt_value << std::endl;
                            }
                        } else {
                            std::cout << "WL NOT RELEVANT value at pos: " << i << " ,Not adding to cache "
                                      << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                      << "Swizzling :"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                      << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                      << ", sane: " << sane << ", Relevant:" << isRelevant
                                      << ", Cache hit: " << cache_hit << std::endl;
                            ;
                            allGood = false;
                        }
                    } else {
                        const auto err{sanityInfo.value()};
                        std::cout << "Sanity failed at pos: " << i << "  Code: " << err
                                  << " meaning: " << Cycles::toErrorText(err) << ", sane: " << sane
                                  << ", Relevant:" << isRelevant << ", Cache hit: " << cache_hit << std::endl;
                    }
                } else {
                    std::cout << "Negative or zero, or big GT at pos: " << i << " , GT: " << gt_value
                              << ", sane: " << sane << ", Relevant:" << isRelevant << ", Cache hit: " << cache_hit
                              << ", gt: #" << gt << "#" << std::endl;
                    allGood = false;
                }
            }

            ++i;  // finally
        }

        std::cout << "\n Collected :" << i << " workloads from csv: " << i_csv_file << " \n";
        std::cout << " Cache contains now  :" << the_cache.getMap().size() << " entries \n\n";
    }  // csv iteration

    std::cout << "**** Final Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    // std::filesystem::path csv_file_now{csv_file};
    std::filesystem::path cache_file_output{std::move(cache_file_input)};
    cache_file_output.replace_filename(cache_file_output.stem() += "_output");
    cache_file_output.replace_extension("cache_bin");
    std::cout << "Current test cache is : " << cache_file_output << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_output) << std::endl << std::endl;

    ASSERT_FALSE(std::filesystem::exists(cache_file_output)) << "ABORT: Cannot overwrite: " << cache_file_output;
    /*bool const appendIfExists{false};*/
    EXPECT_TRUE(the_cache.write_cache(cache_file_output.string() /*, appendIfExists*/));

    // ASSERT_TRUE(false);
    EXPECT_TRUE(allGood) << "\n\nREAD the LOG!\n\n";
}

/// read a cache, and add to it a list of CSV files. first data in cache stays in cache (no replace/no update policy)
// use this one(check paths first) for generating after each iteration . (swizzling is transparent, except maxpooling)
TEST_F(VPUNNCachePreloadedTest, DISABLED_SWIZZ_transparent_Multiple_CsvGenExtented_OneCache_ACTIVE_ITERATION) {
    OneCacheTRansparentSwizzling& test_config{transparent_SwizzConfig};
    std::filesystem::path cache_file_input{test_config.cache_file_input};
    cache_file_input.replace_extension("cache_bin");

    std::cout << "Current test cache is : " << cache_file_input << ", CSV file: " << cache_file_input << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_input) << std::endl << std::endl;

    ASSERT_TRUE(std::filesystem::exists(cache_file_input))
            << "ABORT: Cannot read NON existent cache: " << cache_file_input;

    FixedCache the_cache{""};
    EXPECT_TRUE(the_cache.read_cache(cache_file_input.string()));

    std::cout << " \n ***** Initial Cache contains  :" << the_cache.getMap().size() << std::endl;

    bool allGood{true};
    for (const auto& one_csv : multiple_csv_files) {
        const std::string i_csv_file = mult_csv_folder_OneCache + one_csv;

        Serializer<FileFormat::CSV> serializer_IN{true};                               // force enable
        serializer_IN.initialize(clean_csv_exension(i_csv_file), FileMode::READONLY);  // open file, basic fields
        EXPECT_TRUE(serializer_IN.is_initialized());

        DPUOperation wl_buff;
        SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
        Series readFieldsAll{};

        std::cout << " \n initial Cache contains  :" << the_cache.getMap().size()
                  << " entries, Before reading CSV file: " << i_csv_file << " \n\n";

        // iterate in CSV
        int i = 1;
        std::cout << "Pos starts from " << i << " \n";
        while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
            auto dpu_wl = wl_buff.clone_as_DPUWorkload();

            const std::string gt{gt_SPRUN_buff.value};

            const float gt_value{convertGTFromCSV(gt)};  // the same op as in descriptor transformers

            SanityReport sanityInfo{};
            // sanitize with the same NN version as fro what you want to generate
            const auto sane = test_config.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

            // is compatible with what the NN can predict? Otherwise profiling is for an out of scope WL,
            // that will replace a good one with the same descriptor (eg swizzling)
            const auto isRelevant =
                    isCompatibleWithTrainedSpace<std::remove_reference_t<decltype(test_config)>::pp_adapt_forWrite>(
                            dpu_wl);  // maybe not

            const std::vector<float> descriptorNN{
                    test_config.preprop_forWrite.transformSingle(dpu_wl)};  // a specific processor
            const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));

            if (gt.length() <= 0) {
                std::cout << "NO GT at pos: " << i << ", gt: #" << gt << "#"
                          << ", sane: " << sane << ", Relevant:" << isRelevant << ", Cache hit: " << cache_hit
                          << std::endl;
            } else {
                EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i
                                          << ", sane: " << sane << ", Relevant:" << isRelevant
                                          << ", Cache hit: " << cache_hit << ", gt: #" << gt << "#" << std::endl;
                if (gt_value > 0 && gt_value < four_bilion) {
                    if (sane) {
                        if (cache_hit) {
                            const auto prevVal = the_cache.get(mkhsh(descriptorNN)).value();
                            const auto delta = abs(gt_value - prevVal);
                            std::cout << "Cache already contains value at pos: " << i << " , New value: " << gt_value
                                      << " , Prev  value: " << prevVal << ", Delta: " << delta << ".             "
                                      << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                      << "Swizzling :"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                      << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                      << std::endl;
                            ;
                            allGood = false;
                        }
                        if (isRelevant) {
                            if (!cache_hit) {
                                the_cache.insert(mkhsh(descriptorNN), gt_value);
                            } else {
                                std::cout << "   ------hit! value not set: " << gt_value << std::endl;
                            }
                        } else {
                            std::cout << "WL NOT RELEVANT value at pos: " << i << " ,Not adding to cache "
                                      << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                      << "Swizzling :"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                      << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                      << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                      << ", sane: " << sane << ", Relevant:" << isRelevant
                                      << ", Cache hit: " << cache_hit << std::endl;
                            ;
                            allGood = false;
                        }
                    } else {
                        const auto err{sanityInfo.value()};
                        std::cout << "Sanity failed at pos: " << i << "  Code: " << err
                                  << " meaning: " << Cycles::toErrorText(err) << ", sane: " << sane
                                  << ", Relevant:" << isRelevant << ", Cache hit: " << cache_hit << std::endl;
                    }
                } else {
                    std::cout << "Negative or zero, or big GT at pos: " << i << " , GT: " << gt_value
                              << ", sane: " << sane << ", Relevant:" << isRelevant << ", Cache hit: " << cache_hit
                              << ", gt: #" << gt << "#" << std::endl;
                    allGood = false;
                }
            }

            ++i;  // finally
        }

        std::cout << "\n Collected :" << i << " workloads from csv: " << i_csv_file << " \n";
        std::cout << " Cache contains now  :" << the_cache.getMap().size() << " entries \n\n";
    }  // csv iteration

    std::cout << "**** Final Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    // std::filesystem::path csv_file_now{csv_file};
    std::filesystem::path cache_file_output{std::move(cache_file_input)};
    cache_file_output.replace_filename(cache_file_output.stem() += "_output");
    cache_file_output.replace_extension("cache_bin");
    std::cout << "Current test cache is : " << cache_file_output << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_output) << std::endl << std::endl;

    ASSERT_FALSE(std::filesystem::exists(cache_file_output)) << "ABORT: Cannot overwrite: " << cache_file_output;
    // bool const appendIfExists{false};
    EXPECT_TRUE(the_cache.write_cache(cache_file_output.string() /*, appendIfExists*/));

    // ASSERT_TRUE(false);
    EXPECT_TRUE(allGood) << "\n\nREAD the LOG!\n\n";
}

// reads the CSV and sees if the cache contains the values from csv
TEST_F(VPUNNCachePreloadedTest, DISABLED_CsvVerifyContent_OneCache) {
    std::filesystem::path csv_file_now{normalConfig.csv_file};
    std::filesystem::path cache_file_now{csv_file_now.replace_extension("cache_bin")};
    std::cout << "Current test cache is : " << cache_file_now << ", CSV file: " << normalConfig.csv_file << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_now) << std::endl << std::endl;

    ASSERT_TRUE(std::filesystem::exists(cache_file_now)) << "ABORT: Cannot read NON existent cache: " << cache_file_now;

    FixedCache the_cache{""};
    EXPECT_TRUE(the_cache.read_cache(cache_file_now.string()));

    Serializer<FileFormat::CSV> serializer_IN{true};                                          // force enable
    serializer_IN.initialize(clean_csv_exension(normalConfig.csv_file), FileMode::READONLY);  // open file, basic fields
    EXPECT_TRUE(serializer_IN.is_initialized());

    DPUOperation wl_buff;
    SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
    Series readFieldsAll{};

    std::cout << " \n initial Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    bool allGood{true};

    // iterate in CSV
    int i = 1;
    std::cout << "Pos starts from " << i << " \n";
    while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
        auto dpu_wl = wl_buff.clone_as_DPUWorkload();
        const std::string gt{gt_SPRUN_buff.value};
        const float gt_value{(float)std::atof(gt.c_str())};
        if (gt.length() <= 0) {
            std::cout << "NO GT at pos: " << i << std::endl;
        } else {
            EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i << std::endl;
            if ((gt_value) > 0 && (gt_value < four_bilion)) {
                SanityReport sanityInfo{};
                // sanitize with the same NN version as fro what you want to generate
                const auto sane = normalConfig.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

                if (sane) {
                    const auto isRelevant = isCompatibleWithTrainedSpace<decltype(normalConfig)::pp_adapt_forRead>(
                            dpu_wl);  // maybe not
                    const std::vector<float> descriptorNN{
                            normalConfig.preprop_forRead.transformSingle(dpu_wl)};  // a specific processor

                    if (isRelevant) {
                        const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));
                        EXPECT_TRUE(cache_hit) << "Cache miss at pos: " << i << std::endl;
                        if (!cache_hit) {
                            std::cout << "Cache miss at pos: " << i << std::endl;
                            allGood = false;
                        } else {
                            // check hit is the gt match
                            const float cache_value = the_cache.get(mkhsh(descriptorNN)).value();
                            EXPECT_NEAR(cache_value, gt_value, cache_delta_tolerance)  // maybe csv has multiple GTs
                                    << "Cache value mismatch at pos: " << i << ", GT :" << gt_value
                                    << " ,Cache: " << cache_value << std::endl;
                        }
                    } else {
                        std::cout << "WL NOT RELEVANT value at pos: " << i << " ,Not checking  cache "
                                  << "WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0])) << std::endl;
                    }

                } else {
                    const auto err{sanityInfo.value()};
                    std::cout << "Sanity failed at pos: " << i << "  Code: " << err
                              << " meaning: " << Cycles::toErrorText(err) << std::endl;
                }
            } else {
                std::cout << "Negative or zero, or big GT at pos: " << i << " , GT: " << gt_value << std::endl;
            }
        }

        ++i;  // finally
    }

    std::cout << "\n Processed :" << i << " workloads from csv \n";
    std::cout << " Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    EXPECT_TRUE(allGood);
}

// checks that the COstMOdel gets the values from the cache (the paired cache). Looks to all in csv and expect the
// runtime is from cache
TEST_F(VPUNNCachePreloadedTest, DISABLED_CsvVerifyPairedCacheHit_OneCache) {
    std::filesystem::path csv_file_now{normalConfig.csv_file};
    std::filesystem::path cache_file_now{csv_file_now.replace_extension("cache_bin")};
    std::cout << "Current test cache is : " << cache_file_now << ", CSV file: " << normalConfig.csv_file << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_now) << std::endl << std::endl;

    ASSERT_TRUE(std::filesystem::exists(cache_file_now)) << "ABORT: Cannot read NON existent cache: " << cache_file_now;

    VPUCostModel model_active_cache{cache_file_now.string()};

    FixedCache the_cache{""};
    EXPECT_TRUE(the_cache.read_cache(cache_file_now.string()));

    Serializer<FileFormat::CSV> serializer_IN{true};                                          // force enable
    serializer_IN.initialize(clean_csv_exension(normalConfig.csv_file), FileMode::READONLY);  // open file, basic fields
    EXPECT_TRUE(serializer_IN.is_initialized());

    DPUOperation wl_buff;
    SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
    Series readFieldsAll{};

    std::cout << " \n initial Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    bool allGood{true};

    // iterate in CSV
    int i = 1;
    std::cout << "Pos starts from " << i << " \n";
    while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
        auto dpu_wl = wl_buff.clone_as_DPUWorkload();
        const std::string gt{gt_SPRUN_buff.value};
        const float gt_value{(float)std::atof(gt.c_str())};
        if (gt.length() <= 0) {
            std::cout << "NO GT at pos: " << i << std::endl;
        } else {
            EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i << std::endl;
            if ((gt_value) > 0 && (gt_value < four_bilion)) {
                auto wl_model = dpu_wl;
                SanityReport sanityInfo{};
                // sanitize with the same NN version as fro what you want to generate
                const auto sane = normalConfig.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

                std::string model_run_info{};
                const auto cycleTime{model_active_cache.DPU(std::move(wl_model), model_run_info)};

                if (sane) {
                    const auto isRelevant =
                            isCompatibleWithTrainedSpace<decltype(normalConfig)::pp_adapt_forRead>(dpu_wl);
                    const std::vector<float> descriptorNN{
                            normalConfig.preprop_forRead.transformSingle(dpu_wl)};  // a specific processor
                    const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));
                    if (isRelevant) {
                        EXPECT_TRUE(cache_hit) << "Reference Cache miss at pos: " << i << std::endl;
                        if (!cache_hit) {
                            std::cout << "Reference Cache miss at pos: " << i << std::endl;
                            allGood = false;
                        } else {  // check that cache hit is the same as the model
                            const float cache_value = the_cache.get(mkhsh(descriptorNN)).value();
                            const auto delta = abs(cache_value - (float)cycleTime);

                            if (delta > cache_delta_tolerance) {
                                std::cout << "Model cycles Mismatch cache value  at pos: " << i
                                          << ", MOdel:" << cycleTime << " ,Cache: " << cache_value
                                          << ", Delta: " << (long int)delta << std::endl;
                                allGood = false;
                            }
                        }
                    } else {
                        std::cout << "WL NOT RELEVANT value at pos: " << i
                                  << " ,Not checking! cache hit:  " << std::to_string(cache_hit)
                                  << "  WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0])) << std::endl;
                    }

                } else {
                    const auto err{sanityInfo.value()};
                    std::cout << "Sanity (csv WL) failed at pos: " << i << "  Code: " << err
                              << " meaning: " << Cycles::toErrorText(err) << std::endl;
                    allGood = false;
                }
            } else {
                std::cout << "Negative or zero GT at pos: " << i << " , GT: " << gt_value << std::endl;
                allGood = false;
            }
        }

        ++i;  // finally
    }

    std::cout << "\n Processed :" << i << " workloads from csv \n";
    std::cout << " Cache contains  :" << the_cache.getMap().size() << " entries \n\n";

    EXPECT_TRUE(allGood);
}

TEST_F(VPUNNCachePreloadedTest, DISABLED_CsvVerifyMISSES_PairedCacheHit_OneCache) {
    std::filesystem::path csv_file_now{normalConfig.csv_file};
    std::filesystem::path cache_file_now{csv_file_now.replace_extension("cache_bin")};
    std::cout << "Current test cache is : " << cache_file_now << ", CSV file: " << normalConfig.csv_file << std::endl;
    std::cout << "File exists?: " << std::filesystem::exists(cache_file_now) << std::endl << std::endl;

    ASSERT_TRUE(std::filesystem::exists(cache_file_now)) << "ABORT: Cannot read NON existent cache: " << cache_file_now;

    VPUCostModel model_active_cache{normalConfig.NN_CM_to_use, cache_file_now.string(), "" /*shavecache*/};

    FixedCache the_cache{""};
    EXPECT_TRUE(the_cache.read_cache(cache_file_now.string()));
    EXPECT_TRUE(the_cache.getMap().size() > 0);

    Serializer<FileFormat::CSV> serializer_IN{true};
    // force enable
    serializer_IN.initialize(clean_csv_exension(csv_cache_misses), FileMode::READONLY);  // open file, basic fields
    EXPECT_TRUE(serializer_IN.is_initialized());

    DPUOperation wl_buff;
    SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // Ground truth
    Series readFieldsAll{};

    std::cout << " \n initial Cache contains  :" << the_cache.getMap().size() << " entries \n\n";
    const auto& cacheStats{model_active_cache.getPreloadedCacheCounter()};
    std::cout << "\nCache statistics for COST MODEL loaded cache: " << cacheStats.printString() << std::endl;

    bool allGood{true};

    // iterate in CSV
    int i = 1;
    std::cout << "Pos starts from " << i << " \n";
    while (serializer_IN.deserialize(readFieldsAll, wl_buff, gt_SPRUN_buff)) {
        auto dpu_wl = wl_buff.clone_as_DPUWorkload();
        const std::string gt{gt_SPRUN_buff.value};
        const float gt_value{(float)std::atof(gt.c_str())};
        // if (gt.length() <= 0) {
        //     std::cout << "NO GT at pos: " << i << std::endl;
        // } else
        {
            // EXPECT_TRUE(gt_value > 0) << "GT ZERO or lower: " << gt_value << " -> at pos: " << i << std::endl;
            if ((gt_value) >= 0 && (gt_value < four_bilion)) {
                auto wl_model = dpu_wl;
                SanityReport sanityInfo{};
                // sanitize with the same NN version as fro what you want to generate
                const auto sane = normalConfig.sanitizerModel.sanitize_workload(dpu_wl, sanityInfo);

                std::string model_run_info{};
                const auto cycleTime{model_active_cache.DPU(std::move(wl_model), model_run_info)};

                if (sane) {
                    const auto isRelevant =
                            isCompatibleWithTrainedSpace<decltype(normalConfig)::pp_adapt_forRead>(dpu_wl);
                    const std::vector<float> descriptorNN{
                            normalConfig.preprop_forRead.transformSingle(dpu_wl)};  // a specific processor
                    const auto cache_hit = the_cache.contains(mkhsh(descriptorNN));
                    if (isRelevant) {
                        // EXPECT_TRUE(cache_hit) << "Reference Cache miss at pos: " << i << std::endl;
                        if (!cache_hit) {
                            std::cout << "Reference Cache miss at pos: " << i << " GT from CSV is: " << gt_value
                                      << std::endl;
                            allGood = false;
                        } else {  // check that cache hit is the same as the model
                            const float cache_value = the_cache.get(mkhsh(descriptorNN)).value();
                            const auto delta = abs(cache_value - (float)cycleTime);

                            if (delta > 2.0f /*cache_delta_tolerance*/) {
                                std::cout << "Model cycles Mismatch cache value  at pos: " << i
                                          << ", MOdel:" << cycleTime << " ,Cache: " << cache_value
                                          << ", Delta: " << (long int)delta
                                          << " WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                          << "Swizzling :"
                                          << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0])) << " ,"
                                          << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                          << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                          << ", WSparse: " << dpu_wl.weight_sparsity
                                          << " ,ASparse: " << dpu_wl.act_sparsity << std::endl;

                                allGood = false;
                            }
                        }
                    } else {
                        std::cout << "WL NOT RELEVANT value at pos: " << i
                                  << " ,Not checking! cache hit:  " << std::to_string(cache_hit)
                                  << "  WL Operation: " << Operation_ToText.at(static_cast<int>(dpu_wl.op))
                                  << "Swizzling :" << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[0]))
                                  << " ," << Swizzling_ToText.at(static_cast<int>(dpu_wl.input_swizzling[1])) << " ,"
                                  << Swizzling_ToText.at(static_cast<int>(dpu_wl.output_swizzling[0]))
                                  << ", WSparse: " << dpu_wl.weight_sparsity << " ,ASparse: " << dpu_wl.act_sparsity
                                  << ", DPU Cycle Time: " << cycleTime << std::endl;
                        allGood = false;
                    }

                } else {
                    const auto err{sanityInfo.value()};
                    std::cout << "Sanity (csv WL) failed at pos: " << i << "  Code: " << err
                              << " meaning: " << Cycles::toErrorText(err) << std::endl;
                    allGood = false;
                }
            } else {
                std::cout << "Negative or zero GT at pos: " << i << " , GT: " << gt_value << std::endl;
                allGood = false;
            }
        }

        ++i;  // finally
    }

    std::cout << "\n Processed :" << i << " workloads from csv \n";
    std::cout << " Cache contains  :" << the_cache.getMap().size() << " entries \n\n";
    std::cout << "\nCache statistics for COST MODEL loaded cache: " << cacheStats.printString() << std::endl;
    std::cout << "\nCache statistics for reference cache: " << the_cache.getCounter().printString() << std::endl;

    EXPECT_TRUE(allGood);
}

}  // namespace VPUNN_unit_tests
