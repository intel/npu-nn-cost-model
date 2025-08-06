// Copyright © 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”)
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the “third-party-programs.txt” or other similarly-named text file included with the
// Software Package for additional details.

#ifndef HTTP_COST_PROVIDER_H_
#define HTTP_COST_PROVIDER_H_

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
#undef CPPHTTPLIB_ZLIB_SUPPORT
#endif

#include "httplib.h"

#ifdef NO_ERROR
#undef NO_ERROR  // winerror.h defines NO_ERROR, which is redefined for Cycles type
#endif

#ifdef ADD
#undef ADD
#endif

#include <nlohmann/json.hpp>

#include "vpu/types.h"
#include "vpu/validation/data_dpu_operation.h"

namespace VPUNN {

/**
 * @struct ProfilerResponse
 * @brief Represents the response from a profiling request.
 */
struct ProfilerResponse {
    bool success = false;                   ///< Indicates if the request was successful.
    std::string message;                    ///< Message detailing the response.
    std::string res_type;                   ///< Type of the response.
    std::string path;                       ///< Path related to the response (local to server storage).
    std::vector<CyclesInterfaceType> cost;  ///< Cost metrics associated with the response.
};

/**
 * @class HTTPClient
 * @brief Handles HTTP communication with a specified host and port.
 */
class HTTPClient {
public:
    /**
     * @brief Constructs an HTTPClient with the given host and port.
     * @param host The hostname or IP address of the server.
     * @param port The port number to connect to.
     */
    HTTPClient(const std::string host, int port)
            : _host(host), _port(port), _client(_host, _port) {
              };

    /**
     * @brief Sends a JSON request to the specified path.
     * @param payload The JSON payload to send.
     * @param path The endpoint path for the request.
     * @return The JSON response from the server.
     */
    nlohmann::json sendJsonRequest(const nlohmann::json& payload, const std::string& path);

protected:
    const std::string _host;        ///< The hostname or IP address.
    const int _port;                ///< The port number.
    httplib::Client _client;        ///< The HTTP client instance.
};

/**
 * @class HTTPProfilingClient
 * @brief Extends HTTPClient to handle profiling-service-specific responses.
 */
class HTTPProfilingClient : public HTTPClient {
public:
    HTTPProfilingClient(const std::string host, int port): HTTPClient(host, port) {};

     /**
     * @brief Processes the profiler's JSON response.
     * @param response The JSON response from the profiler.
     * @return A ProfilerResponse object containing the processed data.
     */
    ProfilerResponse handle_profiler_response(const nlohmann::json& response);

     /**
     * @brief Checks if the profiling service is available.
     * @param check_backend Optional parameter to specify backend to check.
     * @return True if available (or at least 1 backend available if check_backend not set), false otherwise.
     */
    bool is_available(const std::string& check_backend = "");
};

/**
 * @class HttpDPUCostProvider
 * @brief Provides DPU cost metrics via HTTP requests.
 */
class HttpDPUCostProvider {
public:
    HttpDPUCostProvider(const std::string host = "irlccggpu04.ir.intel.com", const int port = 5000): _client(host, port) {};
    
    /**
     * @brief Retrieves the cost associated with a given DPU operation.
     * @param op The DPUOperation for which to get the cost.
     * @param info A string to store additional information.
     * @param backend The backend to use for retrieving cost.
     * @return The cost as CyclesInterfaceType.
     */
    CyclesInterfaceType getCost(const DPUOperation& op, std::string& info, const std::string& backend = "silicon");

    bool is_available(const std::string& check_backend = "") {
        return _client.is_available(check_backend);
    }

private:
    HTTPProfilingClient _client; ///< The HTTPProfilingClient instance.
    
    // TODO -- extend serializer to output json serialized dpu ops.
    /**
     * @brief Converts a DPUOperation to its JSON representation.
     * @param op The DPUOperation to convert.
     * @return A JSON object representing the DPUOperation.
     */
    const nlohmann::json dpuop_as_json(const DPUOperation& op);
};

}  // namespace VPUNN
#endif
