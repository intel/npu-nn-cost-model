// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client/http_cost_provider.h"
#include "http_client/workload_json.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <variant>
#include "vpu/http_workload_variant.h"

namespace VPUNN {

HTTPClient::HTTPClient(const std::string& host, int port, bool debug)
        : _host(host), _port(port), _client(_host, _port), _debug(debug) {
}

nlohmann::json HTTPClient::sendJsonRequest(const nlohmann::json& request, const std::string& path) const {
    if (_debug) {
        std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Sending request to path: " << path << std::endl;
        std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Request payload: " << request.dump(2) << std::endl;
    }

    try {
        auto res = _client.post(path, request.dump(), "application/json");
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Response received, status: " << res.status << std::endl;
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Response body: " << res.body << std::endl;
        }
        return nlohmann::json::parse(res.body);
    } catch (const nlohmann::json::parse_error& e) {
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - JSON parse error: " << e.what() << std::endl;
        }
        throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
    } catch (const std::exception& e) {
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Exception: " << e.what() << std::endl;
        }
        throw std::runtime_error("Exception in http client: " + std::string(e.what()));
    } catch (...) {
        if (_debug) {
            std::cout << "[DEBUG] HTTPClient::sendJsonRequest - Unknown exception caught" << std::endl;
        }
        throw std::runtime_error("Unknown exception in http client");
    }
}

bool HTTPProfilingClient::is_available(const std::string& check_backend) const {
    if (_debug) {
        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Checking availability for backend: "
                  << (check_backend.empty() ? "any" : check_backend) << std::endl;
    }

    nlohmann::json status_request_payload;

    status_request_payload["params"] = nlohmann::json::object();
    status_request_payload["params"]["status"] = true;
    status_request_payload["params"]["name"] = "profiling_request";

    auto res = sendJsonRequest(status_request_payload, "/generate_workload");

    if (res.contains("info")) {
        if (res["info"] == "status") {
            if (check_backend.empty()) {
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::is_available - Service is available" << std::endl;
                }
                return true;
            } else {
                if (check_backend == "silicon") {
                    bool silicon_available = res["profiling"].get<std::string>() == "true";
                    if (_debug) {
                        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Silicon backend available: "
                                  << (silicon_available ? "true" : "false") << std::endl;
                    }
                    return silicon_available;
                } else {
                    if (_debug) {
                        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Backend " << check_backend
                                  << " is available" << std::endl;
                    }
                    return true;
                }
            }
        }
    }

    if (_debug) {
        std::cout << "[DEBUG] HTTPProfilingClient::is_available - Service is not available" << std::endl;
    }
    return false;
}

VPUNN::ProfilerResponse HTTPProfilingClient::handle_profiler_response(const nlohmann::json& response) const {
    if (_debug) {
        std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Processing response" << std::endl;
        std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Response: " << response.dump(2)
                  << std::endl;
    }

    ProfilerResponse profiler_response;
    try {
        if (response.contains("info")) {
            auto info_str = response["info"].get<std::string>();
            if (info_str == "success") {
                profiler_response.success = true;
                profiler_response.cost = response["latencies"].get<std::vector<CyclesInterfaceType>>();
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
                profiler_response.res_type = "success";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Success response, cost size: "
                              << profiler_response.cost.size() << std::endl;
                }
            }

            if (info_str == "generation_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "generation_error";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Generation error: "
                              << profiler_response.message << std::endl;
                }
            }

            if (info_str == "profiling_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "profiling_error";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Profiling error: "
                              << profiler_response.message << std::endl;
                }
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
            }

            if (info_str == "compilation_error") {
                profiler_response.success = false;
                profiler_response.message = response["msg"].get<std::string>();
                profiler_response.res_type = "compilation_error";
                if (_debug) {
                    std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Compilation error: "
                              << profiler_response.message << std::endl;
                }
                if (response.contains("path")) {
                    profiler_response.path = response["path"].get<std::string>();
                }
            }
        }

        if (response.contains("warning")) {
            profiler_response.success = false;
            profiler_response.message = response["msg"].get<std::string>();
            profiler_response.res_type = response["warning"].get<std::string>();
            profiler_response.path = response["path"].get<std::string>();
            if (_debug) {
                std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Warning: "
                          << profiler_response.message << std::endl;
            }
        }

        if (response.contains("error")) {
            profiler_response.success = false;
            if (_debug) {
                std::cout << "[DEBUG] HTTPProfilingClient::handle_profiler_response - Error response detected"
                          << std::endl;
            }

            if (response["error"].contains("msg")) {
                profiler_response.message = response["error"]["msg"].get<std::string>();
                profiler_response.res_type = "unknown";
            } else {
                profiler_response.res_type = response["error"].get<std::string>();
            }

            if (response.contains("path")) {
                profiler_response.path = response["path"].get<std::string>();
            }

            if (response.contains("msg")) {
                profiler_response.message = response["msg"].get<std::string>();
            }

            if (response.contains("trace")) {
                profiler_response.message += "\n" + response["trace"].get<std::string>();
            }
        }

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
    }
    return profiler_response;
}

std::unique_ptr<const HttpCostProvider> HttpCostProvider::initFromEnvironment() {
    bool use_profiling_service{get_env_vars({"ENABLE_VPUNN_PROFILING_SERVICE"}).at("ENABLE_VPUNN_PROFILING_SERVICE") ==
                               "TRUE"};

    std::string host{default_host};
    int port{default_port};
    std::string backend{default_backend};
    bool debug{false};

    if (use_profiling_service) {
        std::string env_host = get_env_vars({"VPUNN_PROFILING_SERVICE_HOST"}).at("VPUNN_PROFILING_SERVICE_HOST");
        int env_port = 0;
        try {
            env_port = std::stoi(get_env_vars({"VPUNN_PROFILING_SERVICE_PORT"}).at("VPUNN_PROFILING_SERVICE_PORT"));
        } catch (const std::exception&) {
            env_port = 0;
        }
        std::string env_backend =
                get_env_vars({"VPUNN_PROFILING_SERVICE_BACKEND"}).at("VPUNN_PROFILING_SERVICE_BACKEND");

        // Check for debug environment variable
        auto env_vars = get_env_vars({"VPUNN_HTTP_CLIENT_DEBUG"});
        const auto& debug_str = env_vars["VPUNN_HTTP_CLIENT_DEBUG"];
        if (!debug_str.empty()) {
            debug = (debug_str == "1" || debug_str == "true" || debug_str == "TRUE" || debug_str == "True");
        }

        // Use default values if environment variables are not set
        host = env_host.empty() ? default_host : env_host;
        port = env_port == 0 ? default_port : env_port;
        backend = env_backend.empty() ? default_backend : env_backend;
    }

    auto provider = std::make_unique<HttpCostProvider>(host, port, debug, backend);
    // provider->profiling_backend = std::move(backend);
    return provider;
}

HttpCostProvider::HttpCostProvider(const std::string& host, int port, bool debug, const std::string& profiling_backend_)
        : _client(host, port, debug), profiling_backend{profiling_backend_}, _debug(debug) {
}

const std::string HttpCostProvider::profilingBackendToString(ProfilingServiceBackend backend) const {
    const auto& backend_map = mapToText<ProfilingServiceBackend>();
    auto backend_idx = static_cast<int>(backend);

    // Use SILICON as default for invalid backends
    if (backend == ProfilingServiceBackend::__size || backend_map.find(backend_idx) == backend_map.end()) {
        backend_idx = static_cast<int>(ProfilingServiceBackend::SILICON);
    }

    return backend_map.at(backend_idx);
}

template <typename WlT>
CyclesInterfaceType HttpCostProvider::getHttpCost(const WlT& workload, std::string& info) const {
    std::string backend = profilingBackendToString(workload.profiling_service_backend_hint);

    if (_debug) {
        std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Getting cost for DPU operation" << std::endl;
        std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Backend: " << backend << std::endl;
        std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Workload UID: " << workload.hash() << std::endl;
    }

    try {
        if (!is_available()) {
            if (_debug) {
                std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Profiling service is not available"
                          << std::endl;
            }
            return Cycles::ERROR_PROFILING_SERVICE;
        }

        nlohmann::json payload;

        payload["params"] = nlohmann::json::object();
        payload["params"]["backend"] = backend;

        payload["params"]["name"] = "profiling_request";
        payload["params"]["timeout"] = -1;  // Need to wait for the profiling to finish

        constexpr const char* workload_key = WorkloadKeyTrait<WlT>::key;
        payload[workload_key] = toJson(workload);

        nlohmann::json response = _client.sendJsonRequest(payload, "/generate_workload");

        auto parsed_res = _client.handle_profiler_response(response);

        CyclesInterfaceType cycles = Cycles::ERROR_PROFILING_SERVICE;
        info = parsed_res.message;

        if (parsed_res.success) {
            if (parsed_res.cost.size() == 1) {
                cycles = parsed_res.cost[0];
                if (_debug) {
                    std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Single latency returned: " << cycles
                              << std::endl;
                }
            } else if (parsed_res.cost.size() > 1) {
                // If multiple latencies are returned, take the maximum
                cycles = *std::max_element(parsed_res.cost.begin(), parsed_res.cost.end());
                if (_debug) {
                    std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Multiple latencies returned, max: "
                              << cycles << std::endl;
                }
            }
        } else {
            if (_debug) {
                std::cout << "[DEBUG] HttpDPUCostProvider::getHttpCost - Failed to get cost: " << info << std::endl;
            }
        }
        return cycles;
    } catch (const std::exception& e) {
        info = std::string("Profiling service error: ") + e.what();
        return Cycles::ERROR_PROFILING_SERVICE;
    }
}

bool HttpCostProvider::is_available() const {
    try {
        return _client.is_available(profiling_backend);
    } catch (const std::exception&) {
        // in case of any exception, consider the profiling service as unavailable
        return false;
    }
}

CyclesInterfaceType HttpCostProvider::getCostImpl(const HttpWorkloadVariant& op, std::string& info) const {
    return std::visit(
            [this, &info](const auto& workload) -> CyclesInterfaceType {
                return getHttpCost(workload.get(), info);
            },
            op.data);
}

}  // namespace VPUNN
