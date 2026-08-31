// Copyright © 2026 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
// LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package")
// is subject to the terms and conditions of the software license agreements for the Software Package,
// which may also include notices, disclaimers, or license terms for third party or open source software
// included in or with the Software Package, and your use indicates your acceptance of all such terms.
// Please refer to the "third-party-programs.txt" or other similarly-named text file included with the
// Software Package for additional details.

#include "http_client/http_client.h"

#include <array>
#include <cctype>
#include <climits>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace VPUNN {
namespace {

// Upper bound on the accepted response size to avoid unbounded memory growth
// from a malformed or malicious server (OWASP: uncontrolled resource consumption).
constexpr size_t max_response_bytes = 64u * 1024u * 1024u;

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t invalid_socket_value = INVALID_SOCKET;

// Initializes and tears down Winsock for the lifetime of a request.
class WinsockGuard {
public:
    WinsockGuard() {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
    }
    ~WinsockGuard() {
        WSACleanup();
    }
    WinsockGuard(const WinsockGuard&) = delete;
    WinsockGuard& operator=(const WinsockGuard&) = delete;
};

void closeSocket(socket_t sock) {
    closesocket(sock);
}
#else
using socket_t = int;
constexpr socket_t invalid_socket_value = -1;

void closeSocket(socket_t sock) {
    ::close(sock);
}
#endif

// RAII wrapper that closes the socket on scope exit.
class SocketHandle {
public:
    explicit SocketHandle(socket_t sock): _sock(sock) {
    }
    ~SocketHandle() {
        if (_sock != invalid_socket_value) {
            closeSocket(_sock);
        }
    }
    SocketHandle(const SocketHandle&) = delete;
    SocketHandle& operator=(const SocketHandle&) = delete;

    socket_t get() const {
        return _sock;
    }

private:
    socket_t _sock;
};

// Toggles non-blocking mode on a socket.
void setNonBlocking(socket_t sock, bool nonblocking) {
#ifdef _WIN32
    u_long mode = nonblocking ? 1u : 0u;
    ::ioctlsocket(sock, FIONBIO, &mode);
#else
    const int flags = ::fcntl(sock, F_GETFL, 0);
    if (flags < 0) {
        return;
    }
    (void)::fcntl(sock, F_SETFL, nonblocking ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
#endif
}

// Waits until a non-blocking connect completes (socket becomes writable) or times out.
// Returns true only if the connection was actually established.
bool waitConnected(socket_t sock, int timeout_ms) {
    fd_set write_set;
    FD_ZERO(&write_set);
    FD_SET(sock, &write_set);

    timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

#ifdef _WIN32
    const int nfds = 0;  // ignored by Winsock select
#else
    const int nfds = static_cast<int>(sock) + 1;
#endif
    if (::select(nfds, nullptr, &write_set, nullptr, &tv) <= 0) {
        return false;  // timeout or select error
    }

    int so_error = 0;
#ifdef _WIN32
    int len = static_cast<int>(sizeof(so_error));
    if (::getsockopt(sock, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so_error), &len) != 0) {
        return false;
    }
#else
    socklen_t len = sizeof(so_error);
    if (::getsockopt(sock, SOL_SOCKET, SO_ERROR, &so_error, &len) != 0) {
        return false;
    }
#endif
    return so_error == 0;
}

// Connects the socket to the address in rp, honoring a connect timeout (ms).
// A non-positive timeout performs a plain blocking connect.
bool connectWithTimeout(socket_t sock, const addrinfo* rp, int timeout_ms) {
    const auto doConnect = [&]() {
#ifdef _WIN32
        return ::connect(sock, rp->ai_addr, static_cast<int>(rp->ai_addrlen));
#else
        return ::connect(sock, rp->ai_addr, rp->ai_addrlen);
#endif
    };

    if (timeout_ms <= 0) {
        return doConnect() == 0;
    }

    setNonBlocking(sock, true);
    bool connected = false;
    if (doConnect() == 0) {
        connected = true;
    } else {
#ifdef _WIN32
        const bool in_progress = (WSAGetLastError() == WSAEWOULDBLOCK);
#else
        const bool in_progress = (errno == EINPROGRESS);
#endif
        if (in_progress) {
            connected = waitConnected(sock, timeout_ms);
        }
    }
    setNonBlocking(sock, false);
    return connected;
}

// Enables TCP keep-alive so a dead peer is eventually detected.
void enableKeepAlive(socket_t sock) {
    const int enable = 1;
#ifdef _WIN32
    (void)::setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, reinterpret_cast<const char*>(&enable), sizeof(enable));
#else
    (void)::setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable));
#endif
}

// Applies a send/receive timeout (ms) to the socket. Non-positive leaves it blocking.
void setIoTimeout(socket_t sock, int timeout_ms) {
    if (timeout_ms <= 0) {
        return;
    }
#ifdef _WIN32
    DWORD tv = static_cast<DWORD>(timeout_ms);
    (void)::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
    (void)::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
    timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    (void)::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    (void)::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

// Opens a TCP connection to host:port with a connect timeout, enabling keep-alive and
// applying an optional send/receive timeout to the returned socket.
socket_t connectTo(const std::string& host, int port, int connect_timeout_ms, int io_timeout_ms) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    const std::string port_str = std::to_string(port);
    addrinfo* result = nullptr;
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0 || result == nullptr) {
        throw std::runtime_error("Failed to resolve host: " + host);
    }

    socket_t sock = invalid_socket_value;
    for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        sock = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == invalid_socket_value) {
            continue;
        }
        if (connectWithTimeout(sock, rp, connect_timeout_ms)) {
            break;
        }
        closeSocket(sock);
        sock = invalid_socket_value;
    }
    freeaddrinfo(result);

    if (sock == invalid_socket_value) {
        throw std::runtime_error("Failed to connect to " + host + ":" + std::to_string(port));
    }

    // Keep-alive detects a dead peer even when no I/O timeout is set (the profiling service
    // may legitimately hold the connection open for a long time while a workload is measured).
    enableKeepAlive(sock);
    setIoTimeout(sock, io_timeout_ms);
    return sock;
}

// Sends the full buffer, looping until all bytes are written.
void sendAll(socket_t sock, const std::string& data) {
    const size_t total = data.size();
    if (total > static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error("HTTP request body exceeds maximum sendable size");
    }
    size_t sent = 0;
    while (sent < total) {
        size_t remaining = total - sent;
        if (remaining > static_cast<size_t>(INT_MAX)) {
            remaining = static_cast<size_t>(INT_MAX);
        }
        const int to_send = static_cast<int>(remaining);
        const auto n = ::send(sock, data.data() + sent, to_send, 0);
        if (n <= 0 || n > to_send) {
            throw std::runtime_error("Failed to send HTTP request");
        }
        sent += static_cast<size_t>(n);
    }
}

// Reads until the peer closes the connection (server uses Connection: close).
std::string receiveAll(socket_t sock) {
    std::string response;
    std::array<char, 8192> buffer;
    for (;;) {
        const auto n = ::recv(sock, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (n < 0) {
            throw std::runtime_error("Failed to receive HTTP response");
        }
        if (n == 0) {
            break;  // connection closed by peer
        }
        if (response.size() + static_cast<size_t>(n) > max_response_bytes) {
            throw std::runtime_error("HTTP response exceeds maximum allowed size");
        }
        response.append(buffer.data(), static_cast<size_t>(n));
    }
    return response;
}

// Parses the status code out of a status line such as "HTTP/1.1 200 OK".
int parseStatusCode(const std::string& status_line) {
    const auto first_space = status_line.find(' ');
    if (first_space == std::string::npos) {
        throw std::runtime_error("Malformed HTTP status line");
    }
    const auto code_start = first_space + 1;
    const auto second_space = status_line.find(' ', code_start);
    const auto code = status_line.substr(code_start, second_space - code_start);
    try {
        return std::stoi(code);
    } catch (const std::exception&) {
        throw std::runtime_error("Malformed HTTP status code");
    }
}

// Case-insensitive lookup of a header value. The name is matched only as the full field
// name at the start of a header line (up to the first colon), so "Content-Length" does not
// match "X-Content-Length".
std::string findHeader(const std::string& headers, const std::string& name) {
    const auto equalsIgnoreCase = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) {
                return false;
            }
        }
        return true;
    };

    size_t line_start = 0;
    while (line_start < headers.size()) {
        auto line_end = headers.find("\r\n", line_start);
        if (line_end == std::string::npos) {
            line_end = headers.size();
        }

        const auto colon = headers.find(':', line_start);
        if (colon != std::string::npos && colon < line_end) {
            const std::string field_name = headers.substr(line_start, colon - line_start);
            if (equalsIgnoreCase(field_name, name)) {
                std::string value = headers.substr(colon + 1, line_end - colon - 1);
                const auto first = value.find_first_not_of(" \t");
                if (first == std::string::npos) {
                    return {};
                }
                const auto last = value.find_last_not_of(" \t");
                return value.substr(first, last - first + 1);
            }
        }

        if (line_end == headers.size()) {
            break;
        }
        line_start = line_end + 2;  // skip CRLF
    }
    return {};
}

// Decodes an HTTP/1.1 chunked transfer-encoded body.
std::string decodeChunked(const std::string& body) {
    std::string decoded;
    size_t pos = 0;
    while (pos < body.size()) {
        const auto line_end = body.find("\r\n", pos);
        if (line_end == std::string::npos) {
            break;
        }
        size_t chunk_size = 0;
        try {
            chunk_size = static_cast<size_t>(std::stoul(body.substr(pos, line_end - pos), nullptr, 16));
        } catch (const std::exception&) {
            throw std::runtime_error("Malformed chunk size in HTTP response");
        }
        if (chunk_size == 0) {
            break;  // last chunk
        }
        const auto chunk_start = line_end + 2;
        if (chunk_start + chunk_size > body.size()) {
            throw std::runtime_error("Truncated chunk in HTTP response");
        }
        decoded.append(body, chunk_start, chunk_size);
        pos = chunk_start + chunk_size + 2;  // skip chunk data + trailing CRLF
    }
    return decoded;
}

}  // namespace

HttpClient::HttpClient(std::string host, int port, int connect_timeout_ms, int io_timeout_ms)
        : _host(std::move(host)), _port(port), _connect_timeout_ms(connect_timeout_ms), _io_timeout_ms(io_timeout_ms) {
}

HttpResponse HttpClient::post(const std::string& path, const std::string& body, const std::string& content_type) const {
#ifdef _WIN32
    WinsockGuard winsock_guard;
#endif

    const SocketHandle socket_handle(connectTo(_host, _port, _connect_timeout_ms, _io_timeout_ms));

    std::string request;
    request.reserve(body.size() + 256);
    request += "POST " + path + " HTTP/1.1\r\n";
    request += "Host: " + _host + ":" + std::to_string(_port) + "\r\n";
    request += "Content-Type: " + content_type + "\r\n";
    request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    request += "Connection: close\r\n";
    request += "\r\n";
    request += body;

    sendAll(socket_handle.get(), request);
    const std::string raw = receiveAll(socket_handle.get());

    const auto header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        throw std::runtime_error("Malformed HTTP response: no header terminator");
    }

    const auto status_line_end = raw.find("\r\n");
    const std::string status_line = raw.substr(0, status_line_end);
    const std::string headers = raw.substr(0, header_end);
    std::string response_body = raw.substr(header_end + 4);

    if (findHeader(headers, "Transfer-Encoding").find("chunked") != std::string::npos) {
        response_body = decodeChunked(response_body);
    } else {
        const auto content_length = findHeader(headers, "Content-Length");
        if (!content_length.empty()) {
            try {
                const auto length = static_cast<size_t>(std::stoul(content_length));
                if (length < response_body.size()) {
                    response_body.resize(length);
                }
            } catch (const std::exception&) {
                // Ignore a malformed Content-Length and keep the body as received.
            }
        }
    }

    HttpResponse response;
    response.status = parseStatusCode(status_line);
    response.body = std::move(response_body);
    return response;
}

}  // namespace VPUNN