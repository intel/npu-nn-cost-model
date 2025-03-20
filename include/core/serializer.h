// Copyright © 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef VPUNN_SERIALIZER_H
#define VPUNN_SERIALIZER_H

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include "core/logger.h"
#include "core/utils.h"
#include "vpu/dpu_types.h"
#include "vpu/utils.h"
#include "vpu/serializer_utils.h"

namespace VPUNN {

// Serialization formats
enum class FileFormat { TEXT, CSV, FLATBUFFERS };

// File modes
enum class FileMode { READONLY, WRITE, APPEND, READ_WRITE };

/// @brief Get the extension of a file format
inline const std::string get_extension(const FileFormat& fmt) {
    switch (fmt) {
    case FileFormat::TEXT:
        return ".txt";
    case FileFormat::CSV:
        return ".csv";
    case FileFormat::FLATBUFFERS:
        return ".fb";
    default:
        return ".txt";
    }
}

/// @brief Wrapper over data that is serializable
/// Usually used for primitive types
template <typename T>
struct SerializableField {
    std::string name{};
    T value{};
    const T defaultValue{};

    // Intentional copy - SerializableField should always own the data
    SerializableField(const std::string& name, const T& value): name(name), value(value), defaultValue{value} {
    }
    void resetToDefault() {
        value = defaultValue;
    }
};

using Series = std::unordered_map<std::string, std::string>;

// Trait to check if a type is a SerializableField of any T
template <typename T>
struct is_serializable_field : std::false_type {};

template <typename T>
struct is_serializable_field<SerializableField<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_serializable_field_v = is_serializable_field<T>::value;

// Trait to check if a type T has a member called _member_map
template <typename T, typename = void>
struct has_member_map : std::false_type {};

template <typename T>
struct has_member_map<T, std::void_t<decltype(T::_member_map)>> : std::true_type {};

template <typename T>
inline constexpr bool has_member_map_v = has_member_map<T>::value;

/// @brief Handles file manipulation operations such as reading, writing, and deleting files.
template <FileFormat FMT>
/* coverity[rule_of_three_violation:FALSE] */
class FileHandler {
public:
    /// @brief Default constructor
    FileHandler() = default;
    FileHandler(const FileHandler&) = delete;
    FileHandler(FileHandler&) = delete;

    FileHandler& operator=(const FileHandler&) = delete;
    FileHandler& operator=(FileHandler&) = delete;
    FileHandler& operator=(FileHandler) = delete;

    /// @brief Destructor - if a stream is opened it will be closed
    ~FileHandler() noexcept {
        close();
    }

    /// @brief Open a file for reading and writing
    /// @param input file_name the name of the file to be created / red
    /// @param input mode the mode in which the file should be opened
    void open(const std::string& file, const FileMode mode) {
        file_path = std::filesystem::path(file);
        file_open_mode = mode;

        std::ios::openmode open_mode = std::ios::in;
        switch (mode) {
        case FileMode::READONLY:
            open_mode = std::ios::in;
            break;
        case FileMode::WRITE:
            open_mode = std::ios::out | std::ios::trunc;  // Create new file / overwrite existing content
            break;
        case FileMode::APPEND:
            open_mode = std::ios::out | std::ios::app;  // Append to existing / new file
            break;
        case FileMode::READ_WRITE:
            open_mode = std::ios::in | std::ios::out | std::ios::app;
            break;
        default:
            throw std::runtime_error("Invalid file mode: " + std::to_string(static_cast<int>(mode)));
            break;
        }

        if constexpr (FMT == FileFormat::FLATBUFFERS) {
            open_mode |= std::ios::binary;
        }

        file_stream.open(file_path.string(), open_mode);

        if (!is_open()) {
            throw std::runtime_error("Cannot open file: " + file_path.string() +
                                     " error: " + std::to_string(file_stream.rdstate()));
        }
    }

    /// @brief Close the file stream
    void close() noexcept {
        try {
            if (file_stream.is_open()) {
                file_stream.close();
            }
        } catch (...) {
            // ignore
        }
    }

    /// @brief Check if the file stream is open
    /// @return true if the file stream is open, false otherwise
    bool is_open() const {
        return file_stream.is_open();
    }

    /// @brief Check if the file stream is at the end of the file
    /// @return true if the file stream is at the end of the file, false otherwise
    bool is_eof() const {
        return file_stream.eof();
    }

    /// @brief Check if the file stream is in a good state
    /// @return true if the file stream is in a good state, false otherwise
    bool is_good() const {
        return file_stream.good();
    }

    /// @brief Check if the file stream is in a bad state
    /// @return true if the file stream is in a bad state, false otherwise
    bool is_bad() const {
        return file_stream.bad();
    }

    /// @brief Check if the file stream is in a fail state
    /// @return true if the file stream is in a fail state, false otherwise
    bool is_fail() const {
        return file_stream.fail();
    }

    /// @brief Check if the file stream is empty
    /// @return true if the file stream is empty, false otherwise
    bool is_empty() {
        if (!is_open()) {
            return true;
        }

        file_stream.seekg(0, std::ios::end);  // Go to the end of the file
        auto size = file_stream.tellg();
        file_stream.seekg(0, std::ios::beg);  // Go to the beginning of the file

        return size == 0;
    }

    /// @brief Get the file stream handler
    /// @return the file stream handler
    const std::fstream& get_file_stream() const {
        return file_stream;
    }

    /// @brief Get the file path
    /// @return Path to the file
    const std::filesystem::path get_file_path() const {
        return file_path;
    }

    /// @brief Writes and flushes data at end of file, appends new line if not specified otherwise.
    /// @param data variable to be written
    /// @param endline if set to true creates new line
    void write(const std::string& data, const bool endline = true) {
        file_stream.seekp(0, std::ios::end);
        file_stream.clear();
        if constexpr (FMT == FileFormat::CSV) {
            auto local{data};
            // replace , with .
            // std::replace(local.begin(), local.end(), ',', '.');
            file_stream << local;

        } else {
            file_stream << data;  // no checking
        }

        if (endline) {
            file_stream << std::endl;
        }

        file_stream.flush();
    }

    /// @brief Writes and flushes vector of data at end of file, appends new line if not specified otherwise.
    /// @param data variable to be written
    /// @param endline if set to true creates new line
    void write(const std::vector<std::string>& data, const bool endline = true) {
        std::ostringstream oss;

        if constexpr (FMT == FileFormat::CSV) {
            for (const auto& item : data) {
                auto local{item};
                // replace , with .
                std::replace(local.begin(), local.end(), ',', '.');
                oss << local << ",";
            }
        } else if constexpr (FMT == FileFormat::TEXT) {
            // TODO
        }

        std::string datastr = oss.str();
        if (endline) {
            if constexpr (FMT == FileFormat::CSV) {
                // Remove the trailing comma
                if (!datastr.empty()) {
                    datastr.pop_back();
                }
            } else if constexpr (FMT == FileFormat::TEXT) {
                // TODO
            }
        }

        if (!datastr.empty())
            write(datastr, endline);
    }

    /// @brief Reads header of a file depending on file format. Eg. columns in a CSV file.
    /// Currently assumed that the header spreads over a single line.
    /// @return A vector of header keys.
    std::vector<std::string> read_header() {
        std::vector<std::string> header_keys{};

        if (!is_open()) {
            return header_keys;
        }

        file_stream.seekg(0, std::ios::beg);  // Move to the beginning of the file
        std::string header_line;

        if constexpr (FMT == FileFormat::CSV) {
            // Read header line
            std::getline(file_stream, header_line);

            // Decode header line
            std::istringstream header_line_ss(header_line);
            std::string key;
            while (std::getline(header_line_ss, key, ',')) {
                header_keys.push_back(key);
            }

        } else if constexpr (FMT == FileFormat::TEXT) {
            // TODO: settle on a header format for text files
            std::getline(file_stream, header_line);

            // TODO: decode
        }

        return header_keys;
    }

    /// @brief Update file header with additional fields
    /// The original header and additional fields are concatenated into a unique set of fields
    /// @param additional_fields fields to be added - returns if empty
    /// @param orig_open_mode the file mode intended to open the file
    void update_header(const std::vector<std::string>& additional_fields, FileMode orig_open_mode,
                       bool strict_readonly_file) {
        // If no additional_fields just return
        if (additional_fields.empty())
            return;

        // Read existing reader
        const auto existing_header = read_header();

        // Concatenate existing header with additional fields into a unique set
        std::set<std::string> unique;
        std::vector<std::string> new_header;

        for (const auto& field : existing_header) {
            if (unique.insert(field).second) {
                new_header.push_back(field);
            }
        }

        for (const auto& field : additional_fields) {
            if (unique.insert(field).second) {
                new_header.push_back(field);
            }
        }

        // Return if no changes after concatenation
        if (existing_header == new_header)
            return;

        if (strict_readonly_file) {
            return;  // cannot alter content of this file
        }

        close();

        // write new header and copy rest of the file
        std::ifstream inputFile(file_path);
        std::string line;
        std::string remainingData;

        // Skip the first line (old header)
        std::getline(inputFile, line);

        // Read the rest of the file
        while (std::getline(inputFile, line)) {
            remainingData += line + "\n";
        }
        inputFile.close();

        open(file_path.string(), FileMode::WRITE);
        write(new_header);
        file_stream << remainingData;
        close();

        // open file as originally intended
        open(file_path.string(), orig_open_mode);
    }

    void create_line_index() {
        file_stream.seekg(0, std::ios::beg);  // Move to the beginning of the file
        std::string line;
        std::getline(file_stream, line);
        while (file_stream.good()) {
            line_positions.push_back(file_stream.tellg());
            std::getline(file_stream, line);
        }
    }

    /// @brief Reads a line from filestream as well as decoding depending on file format.
    /// @param read_tokens in List of values decoded from the line of text.
    bool readln(std::vector<std::string>& read_tokens, const int& index = -1) {
        if (index >= 0 && index < static_cast<int>(line_positions.size())) {
            file_stream.seekg(line_positions[index], std::ios::beg);
        }

        std::string line;
        if (!std::getline(file_stream, line)) {
            return false;
        }
        std::istringstream iss(line);
        std::string token;

        if constexpr (FMT == FileFormat::CSV) {
            // Decode CSV line
            while (std::getline(iss, token, ',')) {  // fails to read and empty info after last  delimiter
                read_tokens.push_back(token);
            }
            const auto last{line.rbegin()};  // special handling for empty last
            if ((last != line.crend()) && (*last == ',')) {
                read_tokens.push_back("");  // empty last
            }
        } else if constexpr (FMT == FileFormat::TEXT) {
            // TODO
        }

        return true;
    }

    /// @brief Jump to the beginning of the file and skip header line
    void jump_to_beginning() {
        file_stream.seekg(0, std::ios::beg);  // Move to the beginning of the file

        std::string line;
        std::getline(file_stream, line);
    }

    std::size_t get_num_rows() {
        return line_positions.size();
    }

private:
    std::fstream file_stream{};                    ///> File stream object
    std::filesystem::path file_path{};             ///> Absolute path to the file
    FileMode file_open_mode{};                     ///> File open mode
    std::vector<std::streampos> line_positions{};  ///> Store the position of each line in the file
};

/**
 * @brief Serializer class purpose is to serialize and deserialize any data to a file.
 * It currently supports CSV and TEXT (partial support) formats.
 */
template <FileFormat fmt>
class Serializer {
public:
    /// @brief Constructor - any time a serializer is created, it checks if serialization is enabled
    /// using the ENABLE_VPUNN_DATA_SERIALIZATION environment variable.
    Serializer(const bool force_enable = false)
            : serialization_enabled(!force_enable ? get_env_vars({"ENABLE_VPUNN_DATA_SERIALIZATION"})
                                                                    .at("ENABLE_VPUNN_DATA_SERIALIZATION") == "TRUE"
                                                  : true){};

    Serializer(const Serializer&) = delete;
    Serializer(Serializer&) = delete;

    Serializer& operator=(const Serializer&) = delete;
    Serializer& operator=(Serializer&) = delete;
    Serializer& operator=(Serializer) = delete;

    /// @brief Default Destructor
    ~Serializer() = default;

    /// @brief Check if serialization is enabled
    bool is_serialization_enabled() const {
        return serialization_enabled;
    }

    /// @brief Initialize the serializer with a file name and a list of fields
    /// checks serialization_enabled
    /// @param input file_name the name of the file to be created / read
    /// @param mode the mode in which the file should be opened
    /// @param input fields a list of fields (a field is a unique key identifying a serialized object value, eg. column
    /// name in a CSV file)
    void initialize(const std::string& file_name, const FileMode mode, const std::vector<std::string> fields = {},
                    const bool index_lines = false) {
        if (!serialization_enabled) {
            return;
        } else {
            // Reset the serializer - close file stream, empty all buffers
            reset();

            auto file_path = std::filesystem::path(file_name);
            if (!file_path.has_extension()) {
                file_path = std::filesystem::path(file_name + get_extension(fmt));
            }
            bool file_exists = std::filesystem::exists(file_path);

            bool existing_readonly_file = false;
            // If file does not already exist, create a new file with a unique name (unique to current process run)
            if (!file_exists) {
                file_path = std::filesystem::path(file_name + generate_uid() + get_extension(fmt));
            } else {
                // file exists
                if (FileMode::READONLY == mode) {
                    existing_readonly_file = true;
                }
            }

            file.open(file_path.string(), mode);

            if (!file.is_empty()) {
                file.update_header(fields, mode, existing_readonly_file);  // fields might be ignored if read only!
                create_index_map();
            }

            // If initial index mapping failed (eg. due to missing header), write the header and try again
            if (index_map.empty() && !fields.empty()) {
                // Each entry in fields has to be unique
                std::unordered_set<std::string> seen;
                std::vector<std::string> unique_fields;

                for (const auto& field : fields) {
                    if (seen.count(field) == 0) {
                        seen.insert(field);
                        unique_fields.push_back(field);
                    }
                }

                file.write(unique_fields);
                end();

                create_index_map();

                if (index_map.empty()) {
                    throw std::runtime_error("Failed to create index map for file: " + file_path.string());
                }
            }

            // Jump to beginning of file
            file.jump_to_beginning();

            if (index_lines) {
                file.create_line_index();

                // Jump to beginning of file
                file.jump_to_beginning();
            }
        }
    }

    /// @brief Reset the serializer - close file stream, empty all buffers
    void reset() {
        file.close();
        write_tokens.clear();
        index_map.clear();
    }

    /// @brief Jump to the beginning of the file and skip header line
    void jump_to_beginning() {
        file.jump_to_beginning();
    }

    /// @brief Check if the write buffer is clean
    /// @return true if the write buffer is clean, false otherwise
    bool is_write_buffer_clean() const {
        // Since write tokens might be populated with empty strings, check if any token is not empty
        bool isTokensEmpty = true;
        for (const auto& token : write_tokens) {
            if (!token.empty()) {
                isTokensEmpty = false;
                break;
            }
        }
        return isTokensEmpty;
    }

    /// @brief Clean the write buffer
    void clean_buffers() {
        write_tokens.clear();
        write_tokens.resize(index_map.size(), "");
    }

    /// @brief Get the field names - eg. columns in a CSV file
    /// @return A vector of field names
    std::vector<std::string> get_field_names() const {
        std::vector<std::string> field_names;

        {
            std::map<int, std::string> map_index_ordered;
            // create a temporary  map ordered by index = (order of appearance in csv)
            for (const auto& [key, index] : index_map) {
                map_index_ordered[index] = key;
            }

            for (const auto& [index, name] : map_index_ordered) {
                field_names.push_back(name);
            }
        }
        return field_names;
    }

    /// @brief Get the file format of the serializer
    FileFormat get_format() const {
        return format;
    }

    /// @brief Get the raw file stream handler (why?)
    const std::fstream& get_file_stream() const {
        return file.get_file_stream();
    }

    /// @brief Get the file name
    std::string get_file_name() const {
        return file.get_file_path().filename().string();
    }

    /// @brief Check if the serializer is initialized
    /// checks serialization_enabled
    bool is_initialized() const {
        return serialization_enabled && file.is_open() && !index_map.empty();
    }

    /// @brief Get the number of rows in the file
    std::size_t get_num_rows() {
        return file.get_num_rows();
    }

    /// @brief Serialize a list of variadic arguments to the write buffer
    /// Currently supported types: SerializableField, Series, and any type with a member map
    /// checks serialization_enabled
    /// @param args a list of arguments to be serialized
    template <typename... Args>
    void serialize(Args&&... args) {
        if (!is_initialized())
            return;

        // Define operation that will be mapped to each input argument
        auto operation = [&](auto& arg) {
            using argtype = std::remove_reference_t<decltype(arg)>;  // Decayed type of the argument

            // Check if currently evaluated type has a member map
            if constexpr (has_member_map_v<argtype>) {
                // Iterate over the member map and serialize each member if it exists in the index map
                for (auto& [key, value] : arg._member_map) {
                    if (index_map.count(key) > 0) {
                        const int idx = index_map[key];  // key index to position in the write buffer

                        // Member map is a heterogeneous map, so we need to use std::visit to handle different types
                        std::visit(
                                [&](auto&& _arg) {
                                    using _argtype =
                                            std::decay_t<decltype(_arg)>;  // Decayed type of the member map value

                                    // Handle special case for std::function<VPUNN::DimType(VPUNN::DimType)> - Needs to
                                    // be generalized
                                    if constexpr (std::is_same_v<_argtype, VPUNN::SetGet_MemberMapValues>) {
                                        // _arg have two parameters first one is false and that means that _arg is in
                                        // get_mode
                                        // (function will just return a value), second parameter could be any value, in
                                        // get_mode its value doesn't matter
                                        write_tokens[idx] = std::to_string(_arg(false, ""));
                                    } else {
                                        // All types are stored as references in the member map, so needs to be further
                                        // decayed into underying type
                                        using T = std::remove_reference_t<decltype(_arg.get())>;
                                        const T& val = _arg.get();  // Actual value

                                        // Special case for enums - map to text if possible
                                        if constexpr (has_mapToText<T>::value && has_enumName<T>::value) {
                                            std::string name = enumName<T>() + "." +
                                                               mapToText<T>().at(static_cast<const int>(val));
                                            write_tokens[idx] = std::move(name);
                                        } else {
                                            // Base case - convert to string -- Assumes type T has a valid
                                            // std::to_string implementation
                                            write_tokens[idx] = std::to_string(val);
                                        }
                                    }
                                },
                                value);
                    }
                }
            }

            // Handle argument of type SerializableField
            else if constexpr (is_serializable_field_v<argtype>) {
                if (index_map.count(arg.name) > 0) {
                    auto ss = std::ostringstream();
                    using T = decltype(arg.value);  // Type of the stored value

                    // Special case for enums - map to text if possible
                    if constexpr (has_mapToText<T>::value && has_enumName<T>::value) {
                        std::string name = enumName<T>() + "." + mapToText<T>().at(static_cast<const int>(arg.value));
                        ss << name;
                    } else {
                        ss << arg.value;  // Convert to string -- Assumes type T has a valid std::to_string
                                          // implementation
                    }

                    write_tokens[index_map[arg.name]] = ss.str();
                }
            }

            // Handle argument of type Series
            else if constexpr (std::is_same_v<argtype, Series>) {
                for (const auto& [key, value] : arg) {
                    if (index_map.count(key) > 0) {
                        write_tokens[index_map[key]] = value;
                    }
                }
            }
        };

        // map operation to each argument
        (operation(args), ...);
    }

    /// @brief Deserialize a line from the file stream to a list of variadic arguments
    /// Currently supported types: SerializableField and any type with a member map
    /// checks serialization_enabled
    /// @return true if deserialization was successful, false otherwise
    template <typename... Args>
    bool deserialize(Args&&... args) {
        if (!is_initialized())
            return false;

        // Read next line from the file stream, return false if eof/fail
        std::vector<std::string> read_tokens;  // position is from index map
        if (!file.readln(read_tokens, line_index))
            return false;

        // Define operation that will be mapped to each input argument
        auto operation = [&](auto& arg) {
            using argtype = std::remove_reference_t<decltype(arg)>;  // Decayed type of the argument

            // Check if currently evaluated type has a member map
            if constexpr (has_member_map_v<argtype>) {
                const auto member_names = argtype::_get_member_names();
                auto member_map = arg._member_map;
                // Iterate over the member_names and deserialize each member if it exists in the index map
                for (auto& key_member : member_names) {
                    const auto it = arg._member_map.find(key_member);
                    if (index_map.count(key_member) > 0 &&
                        index_map.at(key_member) < static_cast<int>(read_tokens.size())) {
                        std::istringstream ss(read_tokens[index_map.at(key_member)]);

                        const auto key_var{key_member};

                        // Member map is a heterogeneous map, so we need to use std::visit to handle different types
                        std::visit(
                                [&ss, &key_var](auto&& _arg) {
                                    // Special case - getter/setter style field - TODO: generalize
                                    if constexpr (std::is_same_v<std::decay_t<decltype(_arg)>,
                                                                 VPUNN::SetGet_MemberMapValues>) {
                                        std::string val;
                                        ss >> val;
                                        // arg have two parameters first one is true and that means that arg is in
                                        // set_mode, second parameter is the read value we want to assign to a variable
                                        /* coverity[copy_instead_of_move] */
                                        _arg(true, val);
                                    } else {
                                        ss >> _arg.get();  // Base case - convert to type of arg, throw if conversion
                                                           // fails

                                        if (ss.fail()) {
                                            throw std::runtime_error(
                                                    "Deserialize: Conversion failed for type: " +
                                                    std::string(typeid(std::remove_reference_t<decltype(_arg.get())>)
                                                                        .name()) +
                                                    "key:" + key_var + " ss:" + ss.str() + "$END");
                                        }
                                    }
                                },
                                it->second /*value*/);
                    } else {  // field not found in index_map, its value can be computed based on a set of rules or
                              // could be default, we handle this case in lambda functions
                        std::visit(
                                [](auto&& _arg) {
                                    // special case when value can be computed, handled in lambda function
                                    if constexpr (std::is_same_v<std::decay_t<decltype(_arg)>,
                                                                 VPUNN::SetGet_MemberMapValues>) {

                                        // arg have two parameters first one is true and that means that arg is in
                                        // set_mode, the second parameter, which is an empty string, indicates that the
                                        // value will be set by the lambda function either to the default value or based
                                        // on certain rules
                                        _arg(true, "");
                                    }
                                    // else case means that value is initialized with its default value
                                },
                                it->second /*value*/);
                    }
                }
            } else if constexpr (is_serializable_field_v<argtype>) {  // Handle argument of type SerializableField
                const std::string fieldName = arg.name;

                if (index_map.find(fieldName) !=
                    index_map.end()  // found in serializable fields, then must update value
                ) {
                    const bool is_in_tokens{(index_map.at(fieldName) < static_cast<int>(read_tokens.size()))};

                    std::istringstream ss(is_in_tokens ? read_tokens[index_map.at(fieldName)] : "");

                    if (ss.str().length() <= 0) {
                        arg.resetToDefault();  // reset to default!
                    } else {
                        ss >> arg.value;  // Base case - convert to type of value, throw if conversion fails
                    }

                    if (ss.fail()) {
                        throw std::runtime_error(
                                "DeSerialize:Conversion failed for type: " + std::string(typeid(arg.value).name()) +
                                " FiledName: " + fieldName + " IndxdMap:" + std::to_string(index_map.at(fieldName)) +
                                " TokensSize:" + std::to_string(read_tokens.size()) + " ss:" + ss.str() + "$END");
                    }
                }
            }
            // Handle argument of type Series
            else if constexpr (std::is_same_v<argtype, Series>) {
                for (const auto& [key, index] : index_map) {
                    if (index < static_cast<int>(read_tokens.size())) {
                        arg[key] = read_tokens[index];
                    } else {
                        arg[key] = "";  // not found in tokens, forget old value
                    }
                }
            }
        };

        // map operation to each argument
        (operation(args), ...);

        return true;
    }

    /// @brief End serialization of a block of data - write the write buffer to the file and clean buffers
    void end() {
        if (!write_tokens.empty())
            file.write(write_tokens);

        clean_buffers();
    }

    /// @brief Read a row from the file stream to a Series
    /// @param row the Series to store the read row
    /// @param filter a set of keys to filter out from the read row
    /// @return true if reading was successful, false otherwise
    bool read_row(Series& row, std::unordered_set<std::string> filter = {}) {
        if (!is_initialized())
            return false;

        std::vector<std::string> read_tokens;
        if (!file.readln(read_tokens, line_index))
            return false;

        for (const auto& [key, idx] : index_map) {
            if (filter.count(key) == 0) {
                if (idx < static_cast<int>(read_tokens.size())) {
                    row[key] = read_tokens[idx];
                } else {
                    row[key] = "";  // not found in tokens, forget old value
                }
            }
        }

        return true;
    }

    Serializer<fmt>& at(const int& index) {
        line_index = index;
        return *this;
    }

private:
    const bool serialization_enabled{
            false};  ///> Enable or disable serialization (set up at ctor by external mechanism/env variable)
    const FileFormat format{fmt};                      ///> Serialization format
    FileHandler<fmt> file{};                           ///> File handler
    std::unordered_map<std::string, int> index_map{};  ///> First stores key name (eg column name), then a position
    std::vector<std::string> write_tokens{};           ///> Buffer to store tokens to be written - needs to live between
                                                       /// serialization calls until end()
    int line_index{-1};                                ///> Index of the current line in the file

    /// @brief Create an index map to store positions of each unique identifier of a field.
    /// Eg. Positions of each column in a CSV file
    /// For CSV, the index map is based on the header line
    void create_index_map() {
        if (!file.is_open())
            return;

        if (file.is_empty())
            return;

        auto header_keys = file.read_header();
        if (header_keys.empty()) {
            return;
        }

        // Store the index of each key in the header
        for (size_t idx = 0; idx < header_keys.size(); ++idx) {
            index_map[header_keys[idx]] = static_cast<int>(idx);
        }

        write_tokens.resize(index_map.size(), "");
    }
};

using CSVSerializer = Serializer<FileFormat::CSV>;
using TextSerializer = Serializer<FileFormat::TEXT>;

}  // namespace VPUNN

#endif  // VPUNN_SERIALIZER_H
