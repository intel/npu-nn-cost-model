# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
# LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
# is subject to the terms and conditions of the software license agreements for the Software Package,
# which may also include notices, disclaimers, or license terms for third party or open source software
# included in or with the Software Package, and your use indicates your acceptance of all such terms.
# Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
# Software Package for additional details.

# This is a collection of CMake functions and macros used for file processing tasks
# such as reading, writing, and generating files.

# Function to safely write content to a file.
# @Params:
#   file - path to the file to write
#   content - content to write into the file
#   write_mode - mode to use for writing (e.g., "WRITE", "APPEND")
function(vpu_cost_model_safe_file_write file content write_mode)
    # tracking of rewrite is required to avoid rebuild of the whole project
    # in case of cmake rerun. Need to rebuild only if content of content is changed
    set(rewrite_file ON)
    if(EXISTS ${file})
        file(READ ${file} current_content)
        string(SHA256 current_hash "${current_content}")
        string(SHA256 new_hash "${content}")
        if(current_hash STREQUAL new_hash)
            set(rewrite_file OFF)
        endif()
    endif()

    if(rewrite_file)
        file(${write_mode} ${file} "${content}")
    endif()
endfunction()

# ===================== CSV Parsing Functions =====================

# Function to parse a CSV line handling quoted fields properly
# @Params:
#   line - input CSV line
#   out_fields - variable to store the list of parsed fields
function(PARSE_CSV_LINE line out_fields)
    set(fields "")
    set(current_field "")
    set(in_quotes OFF)
    set(i 0)

    string(LENGTH "${line}" line_length)

    while(i LESS line_length)
        string(SUBSTRING "${line}" ${i} 1 char)

        if(char STREQUAL "\"")
            # Toggle quote state
            # Current quote handling: if we see a quote, we toggle the in_quotes state.
            # This means that double or escaped quotes inside quoted fields are not handled specially, so expressions like:
            # "value \"with quote\"" or "value ""with quote""" will not be parsed correctly.
            # TODO: for more advanced parsing, shold be taken into consideration
            if(in_quotes)
                set(in_quotes OFF)
            else()
                set(in_quotes ON)
            endif()
        elseif(char STREQUAL "," AND NOT in_quotes)
            # end of field
            string(STRIP "${current_field}" current_field)

            # Remove surrounding quotes if present
            string(REGEX REPLACE "^\"(.*)\"$" "\\1" current_field "${current_field}")
            string(STRIP "${current_field}" current_field)
            list(APPEND fields "${current_field}")
            set(current_field "")
        else()
            # Add character to current field
            string(APPEND current_field "${char}")
        endif()

        math(EXPR i "${i} + 1")
    endwhile()
    
    # Last field
    string(STRIP "${current_field}" current_field)
    string(REGEX REPLACE "^\"(.*)\"$" "\\1" current_field "${current_field}")
    string(STRIP "${current_field}" current_field)
    list(APPEND fields "${current_field}")
    
    set(${out_fields} "${fields}" PARENT_SCOPE)
endfunction()

# Adding more generalized cmake parsing function for csv that will generate output file of a given format.
# It will write to output file at given location the content of introduced pattern, but instead
# of placeholders from pattern there will be values extracted from csv.
# @Params:
#   csv_file_path - path to input csv file
#   out_content - variable to store the generated content
#   out_pattern - pattern to use for each line of output file, use {0}, {1}, ... as placeholders for columns
#   DESCRIPTION - optional description to include in the output file header comment
function(GEN_CONTENT_FROM_CSV csv_file_path out_content out_pattern)
    set(FUNC_NAME "GEN_CONTENT_FROM_CSV")
    # Parse optional arguments
    set(options VERBOSE)          # list of boolean flags, with initial VERBOSE
    set(oneValueArgs DESCRIPTION) # arguments that take exactly one value (only description is allowed)
    set(multiValueArgs "")        # arguments that can take multiple values
    cmake_parse_arguments(PARSE_CSV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}) # The rest of the arguments after first 3 ones will be stored in ARGN

    # Set verbose flag (defaults to false if not specified)
    set(is_verbose ${PARSE_CSV_VERBOSE})

    # Validate required arguments
    if(NOT csv_file_path)
        message(FATAL_ERROR "${FUNC_NAME}: csv_file_path is required")
    endif()

    if(NOT out_content)
        message(FATAL_ERROR "${FUNC_NAME}: out_content is required")
    endif()

    if(NOT out_pattern)
        message(FATAL_ERROR "${FUNC_NAME}: out_pattern is required")
    endif()

    if(NOT EXISTS "${csv_file_path}")
        message(FATAL_ERROR "${FUNC_NAME}: Input CSV file does not exist: ${csv_file_path}")
    endif()

    # Log function start
    if(PARSE_CSV_DESCRIPTION)
        message(STATUS "${FUNC_NAME}: Processing CSV file for: ${PARSE_CSV_DESCRIPTION}")
    else()
        message(STATUS "${FUNC_NAME}: Processing CSV file: ${csv_file_path}")
    endif()

    # Read CSV file content
    file(READ "${csv_file_path}" csv_content)

    # Handle different line endings
    string(REPLACE "\r\n" "\n" csv_content "${csv_content}") # Handle Windows line endings
    # Convert into list of lines
    string(REPLACE "\n" ";" csv_lines "${csv_content}")  

    # Remove empty lines
    list(FILTER csv_lines EXCLUDE REGEX "^$")

    if(NOT csv_lines)
        message(FATAL_ERROR "${FUNC_NAME}: CSV file is empty or contains no valid data")
    endif()

    # Extract header to identify columns
    list(GET csv_lines 0 header_line)

    # Parse header using proper CSV parsing
    PARSE_CSV_LINE("${header_line}" clean_columns)

    # Log identified columns
    list(LENGTH clean_columns num_columns)
    if(is_verbose)
        message(STATUS "${FUNC_NAME}: Identified ${num_columns} columns: ")

        set(column_index 0)
        foreach(column ${clean_columns})
            message(STATUS "  [${column_index}] ${column}")
            math(EXPR column_index "${column_index} + 1")
        endforeach()

    endif()

    # Count placeholders in the output pattern
    string(REGEX MATCHALL "\\{[0-9]+\\}" placeholders "${out_pattern}")
    list(LENGTH placeholders num_placeholders)

    if(is_verbose)
        message(STATUS "${FUNC_NAME}: Found ${num_placeholders} placeholders in output pattern")
    endif()

    # Extract unique placeholder numbers and find the maximum
    set(found_placeholders "")
    set(max_placeholder -1)
    foreach(placeholder ${placeholders})
        # Extract the number from {N}
        string(REGEX REPLACE "\\{([0-9]+)\\}" "\\1" placeholder_num "${placeholder}")
        list(APPEND found_placeholders "${placeholder_num}")
        if(placeholder_num GREATER max_placeholder)
            set(max_placeholder ${placeholder_num})
        endif()
    endforeach()

    # Remove duplicates to get unique placeholder indices
    list(REMOVE_DUPLICATES found_placeholders)
    list(SORT found_placeholders COMPARE NATURAL)

    if(is_verbose)
        message(STATUS "${FUNC_NAME}: Unique placeholders used: ${found_placeholders}")
        message(STATUS "${FUNC_NAME}: CSV has ${num_columns} columns available")
    endif()

    # Validate that all used placeholder indices are within the available column range
    # (placeholders are 0-indexed, so valid range is 0 to num_columns-1)
    math(EXPR max_valid_index "${num_columns} - 1")
    if(max_placeholder GREATER max_valid_index)
        message(FATAL_ERROR "${FUNC_NAME}: Placeholder {${max_placeholder}} exceeds available columns! CSV has ${num_columns} columns (valid indices: 0-${max_valid_index})")
    endif()
    
    # Check for gaps in placeholder numbering and warn (not fatal)
    # This helps catch potential typos like using {0}, {1}, {5} when you meant {0}, {1}, {2}
    if(is_verbose)
        set(expected_index 0)
        set(has_gaps OFF)
        foreach(placeholder_num ${found_placeholders})
            if(NOT placeholder_num EQUAL expected_index)
                set(has_gaps ON)
            endif()
            math(EXPR expected_index "${expected_index} + 1")
        endforeach()
        
        if(has_gaps)
            message(STATUS "${FUNC_NAME}: Note: Pattern has gaps in placeholder numbering (e.g., uses {0}, {3} but not {1}, {2}). This is allowed but might indicate a typo.")
        endif()
    endif()

    # This validation for placeholders is meant to cover this situations:
    #  - Allows using subset: CSV with 5 columns, pattern uses only {0} and {3} -> valid
    #  - Allows duplicates: Pattern {0} and {0} -> valid
    #  - No forced sequential requirement: Can skip columns like {0}, {2}, {4} -> valid
    #  - Validates bounds only (placeholder indices don't exceed available columns)
    #  - CSV with 3 columns, pattern {0}, {5} -> invalid

    # Process data rows (skip header)
    list(REMOVE_AT csv_lines 0)

    # Prepare output content
    set(output_content "")

    # Add header comment if description is provided
    if(PARSE_CSV_DESCRIPTION)
        message(STATUS "${PARSE_CSV_DESCRIPTION}")
        string(APPEND output_content "// Generated from CSV: ${PARSE_CSV_DESCRIPTION}\n")
    endif()
    string(APPEND output_content "// Source: ${csv_file_path}\n")
    string(APPEND output_content "// Generated at: ")

    # Add timestamp
    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND output_content "${current_time}\n\n")

    # Process each data row
    set(row_count 0)
    foreach(data_row ${csv_lines})
        if(data_row STREQUAL "")
            continue()  # Skip empty lines
        endif()

        # Parse data row using proper CSV parsing
        PARSE_CSV_LINE("${data_row}" clean_data)

        # Check if row has correct number of fields
        list(LENGTH clean_data num_fields)
        if(NOT num_fields EQUAL num_columns)
            message(WARNING "${FUNC_NAME}: Row ${row_count} has ${num_fields} fields but expected ${num_columns} - skipping")
            continue()
        endif()

        # Replace placeholders in pattern
        set(formatted_line "${out_pattern}")
        set(field_index 0)
        foreach(field ${clean_data})
            string(REPLACE "{${field_index}}" "${field}" formatted_line "${formatted_line}")
            math(EXPR field_index "${field_index} + 1")
        endforeach()

        # Add formatted line to output
        string(APPEND output_content "${formatted_line}")

        math(EXPR row_count "${row_count} + 1")
    endforeach()

    if(is_verbose)
        message(STATUS "${FUNC_NAME}: Successfully processed ${row_count} rows from ${csv_file_path}")
    endif()

    # Return generated content
    set(${out_content} "${output_content}" PARENT_SCOPE)
endfunction()

function(GEN_FILE_FROM_CSV csv_file_path out_file_path out_pattern)
    set(FUNC_NAME "GEN_FILE_FROM_CSV")
    # Parse optional arguments
    set(options VERBOSE)          # list of boolean flags, with initial VERBOSE
    set(oneValueArgs DESCRIPTION) # arguments that take exactly one value (only description is allowed)
    set(multiValueArgs "")        # arguments that can take multiple values
    cmake_parse_arguments(PARSE_CSV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}) # The rest of the arguments after first 3 ones will be stored in ARGN

    # Set verbose flag (defaults to false if not specified)
    set(is_verbose ${PARSE_CSV_VERBOSE})

    if(NOT out_file_path)
        message(FATAL_ERROR "${FUNC_NAME}: out_file_path is required")
    endif()

    # Generate content from CSV
    set(output_content "")
    GEN_CONTENT_FROM_CSV("${csv_file_path}" output_content "${out_pattern}" ${ARGN})

    # Create output directory if it doesn't exist
    get_filename_component(output_dir "${out_file_path}" DIRECTORY)
    if(output_dir AND NOT EXISTS "${output_dir}")
        file(MAKE_DIRECTORY "${output_dir}")
    endif()

    message(STATUS "${FUNC_NAME}: Starting to generate ${out_file_path}")
    vpu_cost_model_safe_file_write("${out_file_path}" "${output_content}" WRITE)

    if(is_verbose)
        message(STATUS "${FUNC_NAME}: Output written to: ${out_file_path}")
    endif()

endfunction()
# =============================================================================================
