// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#include "core/serializer.h"

#include <gtest/gtest.h>

#include <filesystem>
// #include <iomanip>
#include <iostream>
#include <utility>

#include "vpu/validation/data_dpu_operation.h"
#include "vpu/vpu_tensor.h"
#include "vpu_cost_model.h"

namespace VPUNN_unit_tests {
using namespace VPUNN;

class SerializerSimulator : public ::testing::Test {
protected:
    const std::string modelsRoot{"C:\\gitwrk\\libraries.performance.modeling.vpu.nn-cost-model\\models\\"};
    void SetUp() override {
        // set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "TRUE");
        set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
    }

    void TearDown() override {
        set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
    }

    const std::string folder_local{"c:\\gitwrk\\CM_Profilings\\UTests\\ALL_MEXP\\"};

    // considers  what's after :: as a cycletyme
    std::string getGTfromInfo(const std::string& info) const {
        const std::string sep{"::"};
        const auto found = info.rfind(sep);
        if (found != std::string::npos) {
            const auto start_pos{found + sep.length()};
            const auto gt_str{info.substr(start_pos)};  // till the end
            // clang and gcc does not support to use std::move here, so we need suppression
            /* coverity[copy_instead_of_move] */
            return gt_str;
        }
        return "";
    }

    std::vector<std::string> createStandardFieldNames() const {
        std::vector<std::string> stdfields(DPUOperation::_get_member_names().begin(),
                                           DPUOperation::_get_member_names().end());
        stdfields.emplace_back(info_tag);        // expected field
        stdfields.emplace_back(model_file_tag);  // optional field

        stdfields.emplace_back(pred_cycles_tag);  // expected field from the UT run (aka predicted cycles there)
        stdfields.emplace_back(gtUT_tag);         // optional, if GT UT exists

        stdfields.emplace_back(sim_time_VPUX_tag);
        stdfields.emplace_back(gtVPUX_tag);  // add new fields, Ground truth extracted from VPUX profiling

        stdfields.emplace_back(gtSPRUN_tag);  // gt super run
        stdfields.emplace_back(md5hash_tag);  // hash from  super run
        return stdfields;
    }

    const std::string info_tag{"info"};              // sometimes is path
    const std::string model_file_tag{"model_file"};  // model file name related

    const std::string pred_cycles_tag{"pred_cycles"};      //  cycles obtained when running VPUX during interrogation
    const std::string sim_time_VPUX_tag{"sim_time_VPUX"};  // cycles from MLIR, saved by VPUX after interrogating the NN
    const std::string gtVPUX_tag{"gt_VPUX"};  // GT in the context of the full network (MLIR + profile with Json...)

    const std::string gtUT_tag{"gt_uTest"};             // GT from unit tests. extracted from info string from UT
    const std::string gtSPRUN_tag{"gt_superrun"};       // GT obtained when running superrun, aka fast profiling
    const std::string md5hash_tag{"md5hash_superrun"};  // Hash obtained when running superrun, aka fast profiling.
                                                        // Unique per profiling workload(optional)

    //  std::unique_ptr<CSVSerializer> serializer;

public:
    auto createStandardSerializableFields() const {
        DPUOperation wl_buff;                                               // original structure
        SerializableField<std::string> info_buff{info_tag, ""};             // from input
        SerializableField<std::string> modelfile_buff{model_file_tag, ""};  // from input

        SerializableField<std::string> pred_cycles_buff{pred_cycles_tag, ""};  // read from original file
        SerializableField<std::string> gtUT_buff{gtUT_tag, ""};                // read from original file

        SerializableField<std::string> pred_cycles_VPUX_buff{sim_time_VPUX_tag, ""};  // read from original file
        SerializableField<std::string> gt_VPUX_buff{gtVPUX_tag, ""};                  // sometimes exists

        SerializableField<std::string> gt_SPRUN_buff{gtSPRUN_tag, ""};  // in case we merge a profiled list
        SerializableField<std::string> md5hash_buff{md5hash_tag, ""};

        return std::make_tuple(wl_buff, pred_cycles_buff, gtUT_buff, pred_cycles_VPUX_buff, info_buff, modelfile_buff,
                               gt_VPUX_buff, gt_SPRUN_buff, md5hash_buff);
    }

    const std::array<std::string, 2> extraRead_L1{
            "cycles_VPUNN-4011-05",  // highly specific
            "workload_uid",          //
    };

    auto createReadExtra_L1_SerializableFields() const {
        int i = 0;
        auto inc = [&i] {
            return i++;
        };
        auto readExtraFields{std::make_tuple(                             //
                SerializableField<std::string>{extraRead_L1[inc()], ""},  //
                SerializableField<std::string>{extraRead_L1[inc()], ""}   //
                )};
        static_assert(std::tuple_size_v<decltype(readExtraFields)> == std::tuple_size_v<decltype(extraRead_L1)>);

        return readExtraFields;
    }

    // L2 extra should be a superset of L1 read
    const std::array<std::string, 15> extraRead_L2vpunn{
            "cycles_VPUNN-4011-05",  // highly specific
            "workload_uid",          //
            "n_requested_tiles",     //
            "n_computed_tiles",      //
            "n_dpu",                 //
            "tiling_strategy",       //
            "name",                  //
            "level",                 //
            "layer_uid",             //
            "intra_tile_seq_id",     //
            // next are after GT algo for splits
            "gt_cycles",        //
            "pred_win",         //
            "gt_win",           //
            "sim_pred_cycles",  //
            "sim_gt_cycles"     //
    };

    auto createReadExtraL2_SerializableFields() const {
        int i = 0;
        auto inc = [&i] {
            return i++;
        };
        auto readExtraFields_L2{std::make_tuple(                               //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""},  //
                SerializableField<std::string>{extraRead_L2vpunn[inc()], ""}   //
                )};
        static_assert(std::tuple_size_v<decltype(readExtraFields_L2)> ==
                      std::tuple_size_v<decltype(extraRead_L2vpunn)>);

        return readExtraFields_L2;
    }

    std::vector<SerializableField<std::string>> createSerializableFieldsFromTags(const std::vector<std::string>& tags) {
        std::vector<SerializableField<std::string>> out{};
        for (const auto& t : tags) {
            out.emplace_back(SerializableField<std::string>{t, ""});
        }
        return out;
    }

    auto Model_Resim_L1(std::string inputCSV) {
        std::cout << "\n Processing : " << inputCSV;
        // new columns in CSV
        const std::string output0_cycles_tag{"mock_cycles"};
        const std::string output2_cycles_tag{"nn40_post8_cycles"};

        Serializer<FileFormat::CSV> serializer_IN{true};
        {
            // const auto stdfields{createStandardFieldNames()};
            const std::vector<std::string> noFields{};
            serializer_IN.initialize(inputCSV, FileMode::READONLY, std::move(noFields));  // open file, basic fields
            EXPECT_TRUE(serializer_IN.is_initialized());
        }

        Serializer<FileFormat::CSV> serializer_out{true};
        {
            auto out_fields{createStandardFieldNames()};  // + extra fields for output
            for (const auto& t : extraRead_L1) {
                out_fields.emplace_back(t);
            }
            out_fields.emplace_back(output0_cycles_tag);  // output 0, normally a identity resim with input pred cycles
            out_fields.emplace_back(output2_cycles_tag);  // add new fields
            serializer_out.initialize(inputCSV + "_Resim", FileMode::READ_WRITE, std::move(out_fields));
            EXPECT_TRUE(serializer_out.is_initialized());
        }

        VPUCostModel model_4_0_mock{modelsRoot + "vpu_4_0_cloned_from27.vpunn"};
        VPUCostModel model_4_0_Post8{modelsRoot + "vpu_4_0-LucaPost8.vpunn"};

        EXPECT_TRUE(model_4_0_mock.nn_initialized());
        EXPECT_TRUE(model_4_0_Post8.nn_initialized());

        SerializableField<std::string> info_buff{info_tag, ""};  // from input, read 2nd time
        // out
        SerializableField<CyclesInterfaceType> mock_cycles_buff{output0_cycles_tag, 0};   //
        SerializableField<CyclesInterfaceType> resim_cycles_buff{output2_cycles_tag, 0};  // new sim data
        // out over input
        SerializableField<std::string> gt_UT_buff{gtUT_tag, ""};  // recompute this standard field

        auto readFields{std::tuple_cat(createStandardSerializableFields(), createReadExtra_L1_SerializableFields(),
                                       std::tie(info_buff))};
        serializer_IN.jump_to_beginning();

        /// below is the iterative processing of the csv

        while (std::apply(
                [&](auto&&... args) {
                    return serializer_IN.deserialize(args...);
                },
                readFields)) {
            {
                const DPUOperation& wl_buff{std::get<DPUOperation>(readFields)};
                auto dpu_wl = wl_buff.clone_as_DPUWorkload();
                std::string info_mock, info_real;
                const CyclesInterfaceType cycles_mock =
                        model_4_0_mock.DPU(dpu_wl, info_mock);  // SEH exception how to catch?
                const CyclesInterfaceType cycles_LNLpost8 = model_4_0_Post8.DPU(std::move(dpu_wl), info_real);

                constexpr int lastIndex{std::tuple_size_v<decltype(readFields)> - 1};
                const SerializableField<std::string>& info{std::get<lastIndex>(readFields)};

                // gtUT_buff.value = std::stoul(getGTfromInfo(info_buff.value));
                gt_UT_buff.value = getGTfromInfo(info.value) + "";

                mock_cycles_buff.value = cycles_mock;
                resim_cycles_buff.value = cycles_LNLpost8;
            }

            auto outFields{std::tuple_cat(readFields, std::tie(mock_cycles_buff), std::tie(resim_cycles_buff),
                                          std::tie(gt_UT_buff))};

            // write output line
            std::apply(
                    [&](auto&&... args) {
                        return serializer_out.serialize(args...);
                    },
                    outFields);
            serializer_out.end();
        }
    }

    auto Model_Resim_L2(std::string inputCSV) {
        std::cout << "\n Processing : " << inputCSV;
        // new columns in CSV
        const std::string output0_cycles_tag{"mock_cycles"};
        const std::string output2_cycles_tag{"nn40_post8_cycles"};

        Serializer<FileFormat::CSV> serializer_IN{true};
        {
            const auto stdfields{createStandardFieldNames()};
            // const std::vector<std::string> noFields{};
            serializer_IN.initialize(inputCSV, FileMode::READONLY, std::move(stdfields));  // open file, basic fields
            EXPECT_TRUE(serializer_IN.is_initialized());
        }

        Serializer<FileFormat::CSV> serializer_out{true};
        {
            auto out_fields{createStandardFieldNames()};  // + extra fields for output
            for (const auto& t : extraRead_L2vpunn) {
                out_fields.emplace_back(t);
            }
            out_fields.emplace_back(output0_cycles_tag);  // output 0, normally a identity resim with input pred cycles
            out_fields.emplace_back(output2_cycles_tag);  // add new fields
            serializer_out.initialize(inputCSV + "_Resim", FileMode::READ_WRITE, std::move(out_fields));
            EXPECT_TRUE(serializer_out.is_initialized());
        }

        VPUCostModel model_4_0_mock{modelsRoot + "vpu_4_0_cloned_from27.vpunn"};
        VPUCostModel model_4_0_Post8{modelsRoot + "vpu_4_0-LucaPost8.vpunn"};

        EXPECT_TRUE(model_4_0_mock.nn_initialized());
        EXPECT_TRUE(model_4_0_Post8.nn_initialized());

        // auto stdSerialFields{createStandardSerializableFields()};

        // SerializableField<std::string> info_buff{info_tag, ""};  // from input, read 2nd time
        //  out
        SerializableField<CyclesInterfaceType> mock_cycles_buff{output0_cycles_tag, 0};   //
        SerializableField<CyclesInterfaceType> resim_cycles_buff{output2_cycles_tag, 0};  // new sim data
        // out over input
        SerializableField<std::string> gt_UT_buff{gtUT_tag, ""};  // recompute this standard field

        auto readFields{std::tuple_cat(createStandardSerializableFields(), createReadExtraL2_SerializableFields())};
        serializer_IN.jump_to_beginning();

        /// below is the iterative processing of the csv

        while (std::apply(
                [&](auto&&... args) {
                    return serializer_IN.deserialize(args...);
                },
                readFields)) {
            {
                const DPUOperation& wl_buff{std::get<DPUOperation>(readFields)};
                auto dpu_wl = wl_buff.clone_as_DPUWorkload();
                std::string info_mock, info_real;
                const CyclesInterfaceType cycles_mock =
                        model_4_0_mock.DPU(dpu_wl, info_mock);  // SEH exception how to catch?
                const CyclesInterfaceType cycles_LNLpost8 = model_4_0_Post8.DPU(std::move(dpu_wl), info_real);

                // constexpr int lastIndex{std::tuple_size_v<decltype(readFields)> - 1};
                // const SerializableField<std::string>& info{std::get<lastIndex>(readFields)};

                // gtUT_buff.value = std::stoul(getGTfromInfo(info_buff.value));
                // gt_UT_buff.value = getGTfromInfo(info.value) + "";

                mock_cycles_buff.value = cycles_mock;
                resim_cycles_buff.value = cycles_LNLpost8;
            }

            // auto outFieldsC{stdSerialFields};//copy of content . If I say const it will not write the serializable
            // fields
            auto outFields{std::tuple_cat(readFields, std::tie(mock_cycles_buff), std::tie(resim_cycles_buff))};

            // write output line
            std::apply(
                    [&](auto&&... args) {
                        return serializer_out.serialize(args...);
                    },
                    outFields);
            serializer_out.end();
        }
    }

    auto Model_Resim_Generic(std::string inputCSV) {
        std::cout << "\n Processing : " << inputCSV;
        // new columns in CSV
        const std::string output0_cycles_tag{"mock_cycles_now"};
        const std::string post6_cycles_tag{"post6"};
        const std::string post8_cycles_tag{"post8"};

        Serializer<FileFormat::CSV> serializer_IN{true};
        {
            // const auto stdfields{createStandardFieldNames()};
            const std::vector<std::string> noFields{};
            serializer_IN.initialize(inputCSV, FileMode::READONLY, std::move(noFields));  // open file, basic fields
            EXPECT_TRUE(serializer_IN.is_initialized());
        }

        const std::vector<std::string> input_fields_names{serializer_IN.get_field_names()};
        std::vector<std::string> output_fields_names{std::move(input_fields_names)};
        output_fields_names.emplace_back(
                output0_cycles_tag);  // output 0, normally a identity resim with input pred cycles
        output_fields_names.emplace_back(post6_cycles_tag);  // add new fields
        output_fields_names.emplace_back(post8_cycles_tag);  // add new fields
        output_fields_names.emplace_back(gtUT_tag);          // what if duplicate?

        Serializer<FileFormat::CSV> serializer_out{true};
        {
            serializer_out.initialize(inputCSV + "_Resim", FileMode::READ_WRITE, std::move(output_fields_names));
            EXPECT_TRUE(serializer_out.is_initialized());
        }

        VPUCostModel model_4_0_mock{modelsRoot + "vpu_4_0_cloned_from27.vpunn"};
        VPUCostModel model_4_0_Post6{modelsRoot + "vpu_4_0_post6.vpunn"};
        VPUCostModel model_4_0_Post8{modelsRoot + "vpu_4_0-LucaPost8.vpunn"};

        EXPECT_TRUE(model_4_0_mock.nn_initialized());
        EXPECT_TRUE(model_4_0_Post6.nn_initialized());
        EXPECT_TRUE(model_4_0_Post8.nn_initialized());

        DPUOperation wl_buff;
        SerializableField<std::string> info_buff{info_tag, ""};  // from input, read 2nd time
        // out
        SerializableField<CyclesInterfaceType> mock_cycles_buff{output0_cycles_tag, 0};  //
        SerializableField<CyclesInterfaceType> resim_post6_buff{post6_cycles_tag, 0};    // new sim data
        SerializableField<CyclesInterfaceType> resim_post8_buff{post8_cycles_tag, 0};    // new sim data
        // out over input
        SerializableField<std::string> gt_UT_buff{gtUT_tag, ""};  // recompute this standard field

        Series readFieldsAll{};
        // auto readFields{};
        serializer_IN.jump_to_beginning();

        /// below is the iterative processing of the csv

        while (serializer_IN.deserialize(readFieldsAll, wl_buff, info_buff, gt_UT_buff)) {
            {
                // const DPUOperation& wl_buff{std::get<DPUOperation>(readFields)};//HOW
                auto dpu_wl = wl_buff.clone_as_DPUWorkload();
                std::string info_mock, info_real;
                const CyclesInterfaceType cycles_LNLpost8 = model_4_0_Post8.DPU(dpu_wl, info_real);

                const CyclesInterfaceType cycles_mock =
                        model_4_0_mock.DPU(dpu_wl, info_mock);  // SEH exception how to catch?

                const CyclesInterfaceType cycles_LNLpost6 = model_4_0_Post6.DPU(std::move(dpu_wl));

                gt_UT_buff.value = getGTfromInfo(info_buff.value) + "";

                mock_cycles_buff.value = cycles_mock;
                resim_post8_buff.value = cycles_LNLpost8;
                resim_post6_buff.value = cycles_LNLpost6;
            }

            serializer_out.serialize(readFieldsAll, mock_cycles_buff, resim_post8_buff, resim_post6_buff, gt_UT_buff);

            serializer_out.end();
        }
    }  // method

    auto model_Resim_Generic_ModelsList(const std::string& inputCSV, const std::string& outputCSV,
                                        const std::vector<VPUCostModel*>& models) {
        std::cout << "\n Processing : " << inputCSV;

        for (const auto& m : models) {
            EXPECT_FALSE(m == nullptr);
            EXPECT_TRUE((*m).nn_initialized()) << (*m).get_NN_cost_provider().get_model_nickname();
        }

        // new columns in CSV
        auto make_unique_tags = [](const std::vector<VPUCostModel*>& theModels) {
            std::vector<std::string> tags{};
            int colision{1};
            for (const auto& m : theModels) {
                auto tag{m->get_NN_cost_provider().get_model_nickname()};

                if (std::find(tags.cbegin(), tags.cend(), tag) != tags.cend()) {
                    tag = tag + "_#" + std::to_string(colision++);  // colision protection
                }
                tags.emplace_back(tag);
            }
            return tags;
        };

        std::vector<std::string> sim_tags{make_unique_tags(models)};

        Serializer<FileFormat::CSV> serializer_IN{true};
        {
            // const auto stdfields{createStandardFieldNames()};
            const std::vector<std::string> noFields{};
            serializer_IN.initialize(inputCSV, FileMode::READONLY, std::move(noFields));  // open file, basic fields
            EXPECT_TRUE(serializer_IN.is_initialized());
        }

        const std::vector<std::string> input_fields_names{serializer_IN.get_field_names()};
        std::vector<std::string> output_fields_names{std::move(input_fields_names)};
        output_fields_names.emplace_back(gtUT_tag);  // what if duplicate?
        for (auto& tag : sim_tags) {
            output_fields_names.emplace_back(tag);  // newField
        }

        Serializer<FileFormat::CSV> serializer_out{true};
        {
            serializer_out.initialize(outputCSV + "_Resim", FileMode::READ_WRITE, std::move(output_fields_names));
            EXPECT_TRUE(serializer_out.is_initialized());
        }

        DPUOperation wl_buff;
        SerializableField<std::string> info_buff{info_tag, ""};  // from input, read 2nd time

        // out over input
        SerializableField<std::string> gt_UT_buff{gtUT_tag, ""};  // recompute this standard field

        auto makeSimFields = [](const std::vector<std::string>& tags) {
            Series simFields{};
            for (const auto& tag : tags) {
                simFields.emplace(tag, "");
            }
            return simFields;
        };

        Series simulateFieldsAll{makeSimFields(sim_tags)};
        Series readFieldsAll{};

        serializer_IN.jump_to_beginning();

        /// below is the iterative processing of the csv

        while (serializer_IN.deserialize(readFieldsAll, wl_buff, info_buff, gt_UT_buff)) {
            {
                gt_UT_buff.value = getGTfromInfo(info_buff.value) + "";

                auto dpu_wl = wl_buff.clone_as_DPUWorkload();

                auto run_model = [&dpu_wl](VPUCostModel& model) {
                    std::string info{};
                    return std::to_string(model.DPU(dpu_wl, info));
                };
                int i{0};
                for (auto& m : models) {
                    simulateFieldsAll[sim_tags[i++]] = run_model(*m);
                }
            }
            serializer_out.serialize(readFieldsAll, simulateFieldsAll, gt_UT_buff);

            serializer_out.end();
        }
    }  // method

    auto Shave_Resim_Generic_ModelsList(const std::string& inputCSV, const std::string& outputCSV) {
        std::cout << "\n Processing : " << inputCSV;

        SHAVECostModel empty_model{};

        std::vector<std::string> sim_tags{"model_cost_shave"};

        Serializer<FileFormat::CSV> serializer_IN{true};
        {
            // const auto stdfields{createStandardFieldNames()};
            const std::vector<std::string> noFields{};
            serializer_IN.initialize(inputCSV, FileMode::READONLY, std::move(noFields));  // open file, basic fields
            EXPECT_TRUE(serializer_IN.is_initialized());
        }

        const std::vector<std::string> input_fields_names{serializer_IN.get_field_names()};
        std::vector<std::string> output_fields_names{std::move(input_fields_names)};
        output_fields_names.emplace_back(gtUT_tag);  // what if duplicate?
        for (auto& tag : sim_tags) {
            output_fields_names.emplace_back(tag);  // newField
        }

        Serializer<FileFormat::CSV> serializer_out{true};
        {
            serializer_out.initialize(outputCSV + "_Resim", FileMode::READ_WRITE, std::move(output_fields_names));
            EXPECT_TRUE(serializer_out.is_initialized());
        }

        SHAVEWorkload wl_buff;
        SerializableField<std::string> info_buff{info_tag, ""};  // from input, read 2nd time

        // out over input
        SerializableField<std::string> gt_UT_buff{gtUT_tag, ""};  // recompute this standard field

        auto makeSimFields = [](const std::vector<std::string>& tags) {
            Series simFields{};
            for (const auto& tag : tags) {
                simFields.emplace(tag, "");
            }
            return simFields;
        };

        Series simulateFieldsAll{makeSimFields(sim_tags)};
        Series readFieldsAll{};

        serializer_IN.jump_to_beginning();

        /// below is the iterative processing of the csv

        while (serializer_IN.deserialize(readFieldsAll, wl_buff, info_buff, gt_UT_buff)) {
            {
                gt_UT_buff.value = getGTfromInfo(info_buff.value) + "";

                auto shave_wl = wl_buff;

                auto run_model = [&shave_wl](SHAVECostModel& model) {
                    std::string info{};
                    return std::to_string(model.computeCycles(shave_wl, info, true));
                };

                simulateFieldsAll[sim_tags[0]] = run_model(empty_model);
            }
            serializer_out.serialize(readFieldsAll, simulateFieldsAll, gt_UT_buff);

            serializer_out.end();
        }
    }  // method

    auto Model_Resim_Investigation(std::string inputCSV) {
        std::cout << "\n Processing : " << inputCSV;
        // new columns in CSV
        const std::string output0_cycles_tag{"mock_cycles_now"};
        const std::string post6_cycles_tag{"post6"};
        const std::string output2_cycles_tag{"post8"};

        Serializer<FileFormat::CSV> serializer_IN{true};
        {
            // const auto stdfields{createStandardFieldNames()};
            serializer_IN.initialize(inputCSV, FileMode::READONLY);  // open file, basic fields
            EXPECT_TRUE(serializer_IN.is_initialized());
        }

        const std::vector<std::string> input_fields_names{serializer_IN.get_field_names()};
        std::vector<std::string> output_fields_names{std::move(input_fields_names)};
        output_fields_names.emplace_back(
                output0_cycles_tag);  // output 0, normally a identity resim with input pred cycles
        output_fields_names.emplace_back(post6_cycles_tag);    // add new fields
        output_fields_names.emplace_back(output2_cycles_tag);  // add new fields

        Serializer<FileFormat::CSV> serializer_out{true};
        {
            serializer_out.initialize(inputCSV + "_Investigate", FileMode::READ_WRITE, std::move(output_fields_names));
            EXPECT_TRUE(serializer_out.is_initialized());
        }

        VPUCostModel model_4_0_mock{modelsRoot + "vpu_4_0_cloned_from27.vpunn"};
        VPUCostModel model_4_0_Post6{modelsRoot + "vpu_4_0_post6.vpunn"};
        VPUCostModel model_4_0_Post8{modelsRoot + "vpu_4_0-LucaPost8.vpunn"};

        EXPECT_TRUE(model_4_0_mock.nn_initialized());
        EXPECT_TRUE(model_4_0_Post6.nn_initialized());
        EXPECT_TRUE(model_4_0_Post8.nn_initialized());

        DPUOperation wl_buff;
        SerializableField<std::string> info_buff{info_tag, ""};  // from input, read 2nd time
        // out
        SerializableField<CyclesInterfaceType> mock_cycles_buff{output0_cycles_tag, 0};   //
        SerializableField<CyclesInterfaceType> resim_post6_buff{post6_cycles_tag, 0};     // new sim data
        SerializableField<CyclesInterfaceType> resim_cycles_buff{output2_cycles_tag, 0};  // new sim data
        // out over input

        Series readFieldsAll{};
        serializer_IN.jump_to_beginning();

        /// below is the iterative processing of the csv
        int i = 1;

        while (serializer_IN.deserialize(readFieldsAll, wl_buff, info_buff)) {
            {
                // const DPUOperation& wl_buff{std::get<DPUOperation>(readFields)};//HOW
                auto dpu_wl = wl_buff.clone_as_DPUWorkload();
                std::string info_mock, info_real;

                const CyclesInterfaceType cycles_LNLpost8 = model_4_0_Post8.DPU(dpu_wl, info_real);

                const CyclesInterfaceType cycles_mock =
                        model_4_0_mock.DPU(dpu_wl, info_mock);  // SEH exception how to catch?

                const CyclesInterfaceType cycles_LNLpost6 = model_4_0_Post6.DPU(dpu_wl);

                if (/*Cycles::isErrorCode(cycles_LNLpost8) ||*/ (cycles_LNLpost8 == 0)) {
                    // EXPECT_FALSE(Cycles::isErrorCode(cycles_LNLpost8));
                    std::cout << "\ni=" << i << "  " << dpu_wl << "\nPrediction: " << cycles_LNLpost8
                              << "INterpretation: " << Cycles::toErrorText(cycles_LNLpost8) << "\nINFO: " << info_real;
                }

                mock_cycles_buff.value = cycles_mock;
                resim_cycles_buff.value = cycles_LNLpost8;
                resim_post6_buff.value = cycles_LNLpost6;
            }

            serializer_out.serialize(readFieldsAll, mock_cycles_buff, resim_cycles_buff, resim_post6_buff);

            serializer_out.end();
            i++;
        }
    }  // method

    std::string removeExtension(const std::string& filename_withPath) const {
        std::filesystem::path path{filename_withPath};
        if (path.has_extension()) {
            path.replace_extension();  // extension removal
        }
        return path.string();
    }

    std::string addSubFolderInName(const std::string& filename_withPath, std::string subFolderToAdd) const {
        std::filesystem::path path{filename_withPath};
        if ((path.has_filename()) && (subFolderToAdd.length() > 0)) {
            const auto filename = path.filename();  // last part
            const std::string new_filename{subFolderToAdd + "\\" + filename.string()};
            path.replace_filename(new_filename);
        }
        return path.string();
    }

    auto model_Resim_in_subfolder(const std::string& folder, const std::string& input_csv, std::string subfolderOutput,
                                  const std::vector<VPUCostModel*>& models) {
        const std::string theFullName{folder + input_csv};
        const std::string theInputName{removeExtension(theFullName)};
        const std::string theOutputName{addSubFolderInName(theInputName, std::move(subfolderOutput))};

        std::cout << "\n Processing CSV path : " << theInputName << "\n Output: " << theOutputName << "\n";
        model_Resim_Generic_ModelsList(theInputName, theOutputName, models);
    }

    auto Shave_Resim_in_subfolder(const std::string& folder, const std::string& input_csv,
                                  std::string subfolderOutput) {
        const std::string theFullName{folder + input_csv};
        const std::string theInputName{removeExtension(theFullName)};
        const std::string theOutputName{addSubFolderInName(theInputName, std::move(subfolderOutput))};

        std::cout << "\n Processing CSV path : " << theInputName << "\n Output: " << theOutputName << "\n";
        Shave_Resim_Generic_ModelsList(theInputName, theOutputName);
    }

    std::array<VPUCostModel, 6> the_models{
            VPUCostModel{modelsRoot + "vpu_4_0-LucaPost8.vpunn"},  //
            VPUCostModel{modelsRoot + "vpu_4_0_8192.vpunn"},       //

            VPUCostModel{modelsRoot + "vpu_40_159.vpunn"},       //
            VPUCostModel{modelsRoot + "vpu_40_159.fast.vpunn"},  //

            VPUCostModel{modelsRoot + "vpu_40_159_strict.vpunn"},       //
            VPUCostModel{modelsRoot + "vpu_40_159_strict.fast.vpunn"},  //

            // VPUCostModel{modelsRoot + "\\runs\\MSPE_512\\model.vpunn"},      //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_512\\model_max.vpunn"},  //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_512\\model.min.vpunn"},  //

            // VPUCostModel{modelsRoot + "\\runs\\MSPE_1024\\model.vpunn"},      //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_1024\\model_max.vpunn"},  //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_1024\\model.min.vpunn"},  //

            // VPUCostModel{modelsRoot + "\\runs\\MSPE_2048\\model.vpunn"},      //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_2048\\model_max.vpunn"},  //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_2048\\model.min.vpunn"},  //

            // VPUCostModel{modelsRoot + "\\runs\\MSPE_4096\\model.vpunn"},      //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_4096\\model_max.vpunn"},  //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_4096\\model.min.vpunn"},  //

            // VPUCostModel{modelsRoot + "\\runs\\MSPE_8192\\model.vpunn"},      //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_8192\\model_max.vpunn"},  //
            // VPUCostModel{modelsRoot + "\\runs\\MSPE_8192\\model.min.vpunn"},  //
    };
};
/*

TEST_F(SerializerSimulator, DISABLED_Model_Merge_All) {
    const std::string folder{"c:\\gitwrk\\CM_Profilings\\UTests\\Models_2609\\"};
    const std::string sufix_A{"_Resim23464_"};  // L1
    const std::string sufix_B{"_Resim41440_"};  // L2

    const std::vector<std::string> inputs{
            // no extension
            folder + "l1$mlirv1.0$generated$model_e" + sufix_A,       //
            folder + "l1$mlirv1.0$generated$model_n" + sufix_A,       //
            folder + "l1$mlirv1.0$generated$model_c_cut" + sufix_A,   //
            folder + "l1$mlirv1.0$generated$model_a_cut" + sufix_A,   //
            folder + "l1$mlirv1.0$generated$mobilenet_v2" + sufix_A,  //

            folder + "l1$vpunn_168_post8$generated$model_e" + sufix_A,       //
            folder + "l1$vpunn_168_post8$generated$model_c_cut" + sufix_A,   //
            folder + "l1$vpunn_168_post8$generated$model_a_cut" + sufix_A,   //
            folder + "l1$vpunn_168_post8$generated$mobilenet_v2" + sufix_A,  //

            folder + "l2$vpunn_168_post8$generated$model_e" + sufix_B,       //
            folder + "l2$vpunn_168_post8$generated$model_c_cut" + sufix_B,   //
            folder + "l2$vpunn_168_post8$generated$model_a_cut" + sufix_B,   //
            folder + "l2$vpunn_168_post8$generated$mobilenet_v2" + sufix_B,  //

    };  // no extension.

    std::string outputCSV{folder + "outputMerge_"};

    // new columns in CSV
    const std::array<std::string, 2> extraTags{
            "mock_cycles",  // highly specific
            "nn40_post8_cycles",
    };
    auto extraTagsFields{std::make_tuple(                      //
            SerializableField<std::string>{extraTags[0], ""},  //
            SerializableField<std::string>{extraTags[1], ""}   //
            )};
    static_assert(std::tuple_size_v<decltype(extraTagsFields)> == std::tuple_size_v<decltype(extraTags)>);

    auto read_fields{createStandardFieldNames()};  // + extra fields for output
    for (const auto& t : extraRead_L2vpunn) {
        read_fields.emplace_back(t);
    }
    for (const auto& t : extraTags) {
        read_fields.emplace_back(t);
    }

    // global out
    Serializer<FileFormat::CSV> serializer_out;
    {
        auto out_fields{read_fields};  // + extra fields for output
        serializer_out.initialize(outputCSV + "_", FileMode::READ_WRITE, out_fields);
        EXPECT_TRUE(serializer_out.is_initialized());
    }

    const auto readFields{std::tuple_cat(createStandardSerializableFields(), createReadExtraL2_SerializableFields(),
                                         extraTagsFields)};  // last one is special

    auto mergeOneFile = [this, &readFields](Serializer<FileFormat::CSV>& serIN, Serializer<FileFormat::CSV>& serOUT) {
        std::cout << "\n Processing : " << serIN.get_file_name();
        EXPECT_TRUE(serIN.is_initialized()) << " In serializer not init:" << serIN.get_file_name() << "\n";
        serIN.jump_to_beginning();

        SerializableField<std::string> modelfile_buff{model_file_tag, ""};              // from input
        auto readwritw_Fields{std::tuple_cat(readFields, std::tuple(modelfile_buff))};  // last one is special

        // constexpr int lastIndex{std::tuple_size_v<decltype(readwritw_Fields)> - 1};
        // SerializableField<std::string>& src_file_name{std::get<lastIndex>(readwritw_Fields)};

        while (std::apply(
                [&serIN](auto&&... args) {
                    return serIN.deserialize(args...);
                },
                readwritw_Fields)) {
            constexpr int lastIndex{std::tuple_size_v<decltype(readwritw_Fields)> - 1};
            SerializableField<std::string>& src_file_name{std::get<lastIndex>(readwritw_Fields)};
            src_file_name.value = src_file_name.value + "FN :" + serIN.get_file_name();

            // write output line
            std::apply(
                    [&serOUT](auto&&... args) {
                        return serOUT.serialize(args...);
                    },
                    readwritw_Fields);
            serOUT.end();
            src_file_name.value = "";  // since it is not available in header , no index map is processed for input
        }
    };

    {
        for (const auto& input_csv : inputs) {
            std::string inputCSV{input_csv};
            Serializer<FileFormat::CSV> serializer_IN;
            {
                serializer_IN.initialize(inputCSV, FileMode::READONLY, read_fields);  // open file
                EXPECT_TRUE(serializer_IN.is_initialized());
            }
            mergeOneFile(serializer_IN, serializer_out);
        }
    }
}
*/

TEST_F(SerializerSimulator, DISABLE_Multi_Model_Resim_Agnostic_GT) {
    const std::string folder{"c:\\gitwrk\\CM_Profilings\\UTests\\ALL_MEXP\\"};

    const std::vector<std::string> inputs{};

    for (const auto& input_csv : inputs) {
        std::string theName{folder + input_csv};
        std::filesystem::path path{theName};
        if (path.has_extension()) {
            path.replace_extension();  // extension removal
        }
        Model_Resim_Generic(path.string());
    }
}

const std::string folder_MEXP_REMOTE{""};
const std::vector<std::string> inputs_MEXP{};

TEST_F(SerializerSimulator, Multi_Model_Resim_Agnostic_Configurable_GT) {
    const std::vector<std::string> inputs_custom{/*"all_l1_unique_profiled_13_nov_SMALL.csv"*/};

    const std::string folder{folder_local};
    const std::vector<std::string>& inputs{inputs_custom};

    std::vector<VPUCostModel*> models_ptr;
    for (auto& model : the_models) {
        models_ptr.emplace_back(&model);
    }

    for (const auto& input_csv : inputs) {
        model_Resim_in_subfolder(folder, input_csv, "res_all_l1", models_ptr);
    }
}

// THIS test has an Issue deserializing workloads, missing function for deserialize on Shave? Check the Cache creator
// app it has a specific function
TEST_F(SerializerSimulator, DISABLED_SHAVE_resim_serializer_tool) {
    const std::vector<std::string> inputs_custom{"merged_shave_workloads.csv"};
    const std::string folder{folder_local};
    const std::vector<std::string>& inputs{inputs_custom};

    for (const auto& input_csv : inputs) {
        Shave_Resim_in_subfolder(folder, input_csv, "res_all_shave");
    }
}

TEST_F(SerializerSimulator, DISABLED_Model_Merge_All_AGNOSTIC) {
    const std::string folder{"c:\\gitwrk"};
    const std::string sufix_A{"_Resim23464_"};  // L1
    const std::string sufix_B{"_Resim41440_"};  // L2

    const std::vector<std::string> inputs{
            // no extension
            folder + "l1$mlirv1.0$generated$model_e" + sufix_A,       //
            folder + "l1$mlirv1.0$generated$model_n" + sufix_A,       //
            folder + "l1$mlirv1.0$generated$model_c_cut" + sufix_A,   //
            folder + "l1$mlirv1.0$generated$model_a_cut" + sufix_A,   //
            folder + "l1$mlirv1.0$generated$mobilenet_v2" + sufix_A,  //

            folder + "l1$vpunn_168_post8$generated$model_e" + sufix_A,       //
            folder + "l1$vpunn_168_post8$generated$model_c_cut" + sufix_A,   //
            folder + "l1$vpunn_168_post8$generated$model_a_cut" + sufix_A,   //
            folder + "l1$vpunn_168_post8$generated$mobilenet_v2" + sufix_A,  //

            folder + "l2$vpunn_168_post8$generated$model_e" + sufix_B,       //
            folder + "l2$vpunn_168_post8$generated$model_c_cut" + sufix_B,   //
            folder + "l2$vpunn_168_post8$generated$model_a_cut" + sufix_B,   //
            folder + "l2$vpunn_168_post8$generated$mobilenet_v2" + sufix_B,  //

    };  // no extension.

    std::string outputCSV{folder + "outputMerge_"};

    // global out
    Serializer<FileFormat::CSV> serializer_out;

    auto mergeOneFile = [this](Serializer<FileFormat::CSV>& serIN, Serializer<FileFormat::CSV>& serOUT) {
        std::cout << "\n Processing : " << serIN.get_file_name();
        EXPECT_TRUE(serIN.is_initialized()) << " In serializer not init:" << serIN.get_file_name() << "\n";
        serIN.jump_to_beginning();

        Series readFieldsAll{};
        SerializableField<std::string> modelfile_buff{model_file_tag, ""};

        while (serIN.deserialize(readFieldsAll, modelfile_buff)) {
            modelfile_buff.value = modelfile_buff.value + "FN :" + serIN.get_file_name();

            // write output line
            serOUT.serialize(readFieldsAll, modelfile_buff);
            serOUT.end();

            modelfile_buff.value = "";  // since it is not available in header , no index map is processed for input
        }
    };

    {
        for (const auto& input_csv : inputs) {
            std::string inputCSV{input_csv};
            Serializer<FileFormat::CSV> serializer_IN;
            {
                serializer_IN.initialize(inputCSV, FileMode::READONLY);  // open file
                EXPECT_TRUE(serializer_IN.is_initialized());
            }
            std::vector<std::string> output_fields_names{serializer_IN.get_field_names()};
            output_fields_names.emplace_back(model_file_tag);

            serializer_out.initialize(outputCSV + "_", FileMode::READ_WRITE, std::move(output_fields_names));
            EXPECT_TRUE(serializer_out.is_initialized());

            mergeOneFile(serializer_IN, serializer_out);
        }
    }
}

TEST_F(SerializerSimulator, DISABLED_Investigation_Resim_Agnostic) {
    const std::string folder{""};
    const std::vector<std::string> inputs{
            // no extension
            folder + "",  // put here a csv with wl to be analysed
    };  // no extension.

    for (const auto& input_csv : inputs) {
        Model_Resim_Investigation(input_csv);
    }
}

class SerializerSimulator_MEXP_Local : public SerializerSimulator, public ::testing::WithParamInterface<std::string> {
protected:
    void SetUp() override {
        SerializerSimulator::SetUp();
        set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
    }
    void TearDown() override {
        SerializerSimulator::TearDown();
        set_env_var("ENABLE_VPUNN_DATA_SERIALIZATION", "");
    }

    void executeModels(std::string folderOfCSV, std::string csv_input) {
        const std::string folder{std::move(folderOfCSV)};
        const std::string csv_param{std::move(csv_input)};

        std::vector<VPUCostModel*> models_ptr;
        for (auto& model : the_models) {
            models_ptr.emplace_back(&model);
        }

        std::cout << "\n Processing CSV_param : " << csv_param << "\n";

        // for (const auto& input_csv : inputs)
        const auto& input_csv = csv_param;
        {
            model_Resim_in_subfolder(folder, input_csv, "Resim", models_ptr);
        }
    }

public:
private:
};

// TEST_P(SerializerSimulator_MEXP_Local, DISABLED_Resimulate16NN_) {
//     const std::string folder{folder_local};
//     const std::string csv_param{GetParam()};
//
//     executeModels(folder, csv_param);
// }
//
// const std::vector<std::string> inputs_local{};
// INSTANTIATE_TEST_SUITE_P(SerializerSimuLOCO_Local, SerializerSimulator_MEXP_Local, testing::ValuesIn(inputs_local));

class SerializerSimulator_MEXP_DRIVE : public SerializerSimulator_MEXP_Local {
protected:
public:
private:
};

// TEST_P(SerializerSimulator_MEXP_DRIVE, DISABLED_Resimulate16NN_) {
//     const std::string folder{folder_MEXP_REMOTE};
//     const std::string csv_param{GetParam()};
//
//     executeModels(folder, csv_param);
// }
// INSTANTIATE_TEST_SUITE_P(SerializerSimuDRIVE_ALL, SerializerSimulator_MEXP_DRIVE, testing::ValuesIn(inputs_MEXP));

}  // namespace VPUNN_unit_tests
