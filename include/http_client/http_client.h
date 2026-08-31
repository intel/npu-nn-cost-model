// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#ifndef HTTP_CLIENT_H_
#define HTTP_CLIENT_H_

#include <string>

namespace VPUNN {

/**
 * @struct HttpResponse
 * @brief Result of a single HTTP request: status code and response body.
 */
struct HttpResponse {
    int status = 0;    ///< HTTP status code (e.g. 200).
    std::string body;  ///< Response body.
};

/**
 * @class HttpClient
 * @brief Minimal plain-HTTP/1.1 client supporting a single blocking POST.
 *
 * Replaces the third-party httplib dependency for the only usage needed by the
 * profiling-service cost provider: one plain-HTTP POST per request. No TLS, no
 * compression, no keep-alive. Networking is implemented over BSD sockets on
 * POSIX and Winsock on Windows; platform headers are confined to the .cpp.
 */
class HttpClient {
public:
    /**
     * @brief Constructs a client targeting the given host and port.
     * @param host Hostname or IP address of the server.
     * @param port TCP port to connect to.
     * @param connect_timeout_ms Timeout for establishing the TCP connection, in milliseconds.
     *        A non-positive value performs a blocking connect with no timeout.
     * @param io_timeout_ms Send/receive timeout in milliseconds. Zero (default) blocks
     *        indefinitely, which suits profiling requests that intentionally wait for the
     *        server to finish; TCP keep-alive is always enabled to detect dead peers.
     */
    HttpClient(std::string host, int port, int connect_timeout_ms = 10000, int io_timeout_ms = 0);

    /**
     * @brief Sends a single blocking HTTP/1.1 POST request.
     * @param path Request target path (e.g. "/generate_workload").
     * @param body Request body.
     * @param content_type Value for the Content-Type header.
     * @return The response status code and body.
     * @throws std::runtime_error on connection, send, or receive failure.
     */
    HttpResponse post(const std::string& path, const std::string& body, const std::string& content_type) const;

private:
    std::string _host;        ///< Target hostname or IP address.
    int _port;                ///< Target TCP port.
    int _connect_timeout_ms;  ///< Connection establishment timeout in milliseconds.
    int _io_timeout_ms;       ///< Send/receive timeout in milliseconds (0 = block indefinitely).
};

}  // namespace VPUNN

#endif  // HTTP_CLIENT_H_
