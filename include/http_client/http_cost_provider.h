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

#include <nlohmann/json.hpp>
#include "httplib.h"

#ifdef NO_ERROR
#undef NO_ERROR  // winerror.h defines NO_ERROR, which is enum member for Cycles type
#endif

#ifdef ADD
#undef ADD  // arpa/nameser.h defines ADD macro on Linux
#endif

#include "vpu/types.h"
#include "vpu/validation/data_dpu_operation.h"
#include "vpu/profiling_service.h"
#include "core/utils.h"
#include "vpu/http_cost_provider_intf.h"

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
    HTTPClient(const std::string& host, int port);

    /**
     * @brief Sends a JSON request to the specified path.
     * @param payload The JSON payload to send.
     * @param path The endpoint path for the request.
     * @return The JSON response from the server.
     * @throws std::runtime_error if the request fails or the response cannot be parsed.
     */
    nlohmann::json sendJsonRequest(const nlohmann::json& payload, const std::string& path) const;

    /**
     * @brief Enable or disable debug output.
     * @param enable True to enable debug output, false to disable.
     */
    void setDebug(bool enable) { _debug = enable; }

protected:
    const std::string _host;            ///< The hostname or IP address.
    const int _port;                    ///< The port number.
    mutable httplib::Client _client;    ///< The HTTP client instance.
    bool _debug;                        ///< Debug flag for verbose output.
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
     * @throws std::runtime_error if the response cannot be parsed correctly.
     */
    ProfilerResponse handle_profiler_response(const nlohmann::json& response) const;

     /**
     * @brief Checks if the profiling service is available.
     * @param check_backend Optional parameter to specify backend to check.
     * @return True if available (or at least 1 backend available if check_backend not set), false otherwise.
     * @throws std::runtime_error if there is an error during sendJsonRequest.
     */
    bool is_available(const std::string& check_backend = "") const;
};

/**
 * @class HttpCostProvider
 * @brief Provides cost information by communicating with a profiling service over HTTP.
 */
class HttpCostProvider : public IHttpCostProvider {
public:
    HttpCostProvider(const std::string& host = default_host, int port = default_port);

    /**
     * @brief Factory static function that initializes HttpCostProvider from environment variables.
     * @return A unique pointer to the initialized HttpCostProvider, nullptr otherwise.
     */
    static std::unique_ptr<HttpCostProvider> initFromEnvironment();
    
    /**
     * @brief Checks if the profiling service is available.
     * @return True if available, false otherwise.
     */
    bool is_available() const override;

protected:
    /**
     * @brief Variant-based dispatcher for the templated getCost method.
     *
     * Uses std::visit to dispatch the HttpWorkloadVariant to the corresponding
     * getCost<T>() instantiation (e.g., DPUOperation or DMANN workload types).
     * 
     * @param op The workload operation wrapped in a HttpWorkloadVariant.
     * @param info A string to store additional information.
     * @return The cost as CyclesInterfaceType, in case of error returns Cycles::ERROR_PROFILING_SERVICE.
     */
    CyclesInterfaceType getCostImpl(const HttpWorkloadVariant& op, std::string& info) const override;
    
    /**
     * @brief Retrieves the cost associated with a given DPU operation.
     * @tparam WlT The type of the workload operation.
     * @param op The operation of type WlT for which to get the cost.
     * @param info A string to store additional information.
     * @param backend The backend to use for retrieving cost.
     * @return The cost as CyclesInterfaceType, in case of error returns Cycles::ERROR_PROFILING_SERVICE.
     */
    template <typename WlT>
    CyclesInterfaceType getHttpCost(const WlT& op, std::string& info) const;

public:
    /**
     * @brief Converts ProfilingServiceBackend enum to string representation.
     * @param backend The backend enum value to convert.
     * @return String representation of the backend, defaults to "SILICON" if invalid.
     */
    const std::string profilingBackendToString(ProfilingServiceBackend backend=ProfilingServiceBackend::SILICON) const override;

    /**
     * @brief Enable or disable debug output.
     * @param enable True to enable debug output, false to disable.
     */
    void setDebug(bool enable) override;

    ~HttpCostProvider() = default;

private:
    HTTPProfilingClient _client;    ///< The HTTPProfilingClient instance.
    std::string profiling_backend;  ///< The actual profiling backend that is used.
    bool _debug;                    ///< Debug flag for verbose output.

    /**
     * @brief Default values for the HttpCostProvider in case environment variables are not set.
     * or are invalid.
     */
    static constexpr const char*  default_host = "irlccggpu04.ir.intel.com";
    static constexpr int default_port = 5000;
    static constexpr const char* default_backend = "silicon";

    /**
     * @brief Converts a DMANNWorkload type to its JSON representation.
     * @tparam WlT The type of the workload operation.
     * @param op The DMANNWorkload to convert.
     * @return A JSON object representing the DMANNWorkload type.
     */
    template <typename WlT>
    const nlohmann::json toJson(const WlT& wl) const;
};
}  // namespace VPUNN
#endif // HTTP_COST_PROVIDER_H_
