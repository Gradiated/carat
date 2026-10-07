#include "carat/http.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <variant>

namespace carat::http {
namespace {

enum class RequestError {
  malformed,
  headers_too_large,
  body_too_large,
  transfer_encoding_unsupported,
  timeout,
  closed,
};

struct RequestHead {
  std::string method;
  std::string target;
  std::string authorization;
  std::size_t content_length{0};
  bool has_transfer_encoding{false};
};

const char *status_text(const int status) {
  switch (status) {
  case 200:
    return "OK";
  case 204:
    return "No Content";
  case 400:
    return "Bad Request";
  case 401:
    return "Unauthorized";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 408:
    return "Request Timeout";
  case 413:
    return "Content Too Large";
  case 431:
    return "Request Header Fields Too Large";
  case 500:
    return "Internal Server Error";
  case 501:
    return "Not Implemented";
  case 503:
    return "Service Unavailable";
  default:
    break;
  }

  switch (status / 100) {
  case 1:
    return "Informational";
  case 2:
    return "Success";
  case 3:
    return "Redirection";
  case 4:
    return "Client Error";
  case 5:
    return "Server Error";
  default:
    return "Unknown";
  }
}

std::optional<HttpResponse> error_response(const RequestError error) {
  const auto text = [](const int status, const char *body) {
    return HttpResponse{status, "text/plain; charset=utf-8", body};
  };

  switch (error) {
  case RequestError::malformed:
    return text(400, "bad request\n");
  case RequestError::headers_too_large:
    return text(431, "request headers too large\n");
  case RequestError::body_too_large:
    return text(413, "request body too large\n");
  case RequestError::transfer_encoding_unsupported:
    return text(501, "transfer encoding not supported\n");
  case RequestError::timeout:
    return text(408, "request timeout\n");
  case RequestError::closed:
    return std::nullopt;
  }

  return std::nullopt;
}

std::string_view trim(std::string_view value) {
  const auto first = value.find_first_not_of(" \t");
  if (first == std::string_view::npos) {
    return {};
  }

  const auto last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

bool equals_ignoring_case(const std::string_view left, const std::string_view right) {
  return std::equal(left.begin(), left.end(), right.begin(), right.end(),
                    [](const unsigned char a, const unsigned char b) {
                      return std::tolower(a) == std::tolower(b);
                    });
}

std::optional<std::size_t> parse_size(const std::string_view text) {
  std::size_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
    return std::nullopt;
  }

  return value;
}

bool parse_request_line(const std::string_view line, RequestHead &head) {
  const auto first_space = line.find(' ');
  const auto second_space = first_space == std::string_view::npos ? std::string_view::npos
                                                                  : line.find(' ', first_space + 1);
  if (second_space == std::string_view::npos ||
      line.find(' ', second_space + 1) != std::string_view::npos) {
    return false;
  }

  const auto method = line.substr(0, first_space);
  const auto target = line.substr(first_space + 1, second_space - first_space - 1);
  const auto version = line.substr(second_space + 1);
  if (method.empty() || !target.starts_with('/') ||
      (version != "HTTP/1.0" && version != "HTTP/1.1")) {
    return false;
  }

  head.method = method;
  head.target = target;
  return true;
}

bool parse_header_line(const std::string_view line, RequestHead &head,
                       std::optional<std::size_t> &content_length) {
  const auto colon = line.find(':');
  if (colon == std::string_view::npos || colon == 0) {
    return false;
  }

  const auto name = line.substr(0, colon);
  if (name.find_first_of(" \t") != std::string_view::npos) {
    return false;
  }

  const auto value = trim(line.substr(colon + 1));
  if (equals_ignoring_case(name, "authorization")) {
    head.authorization = value;
  } else if (equals_ignoring_case(name, "transfer-encoding")) {
    head.has_transfer_encoding = true;
  } else if (equals_ignoring_case(name, "content-length")) {
    const auto parsed = parse_size(value);
    if (!parsed || (content_length && *content_length != *parsed)) {
      return false;
    }

    content_length = parsed;
  }

  return true;
}

std::variant<RequestHead, RequestError> parse_head(const std::string_view block) {
  RequestHead head;
  std::optional<std::size_t> content_length;
  std::size_t offset = 0;
  bool request_line = true;

  while (offset <= block.size()) {
    const auto end = std::min(block.find("\r\n", offset), block.size());
    const auto line = block.substr(offset, end - offset);
    offset = end + 2;

    if (line.find_first_of("\r\n") != std::string_view::npos) {
      return RequestError::malformed;
    }

    const bool valid = request_line ? parse_request_line(line, head)
                                    : parse_header_line(line, head, content_length);
    if (!valid) {
      return RequestError::malformed;
    }

    request_line = false;
  }

  head.content_length = content_length.value_or(0);
  return head;
}

constexpr std::chrono::milliseconds stop_check_interval{200};
constexpr std::chrono::milliseconds descriptor_exhaustion_backoff{100};

struct ReadLimits {
  const HttpServerOptions &options;
  std::chrono::steady_clock::time_point deadline;
  const std::atomic<bool> &stopping;
};

std::variant<std::size_t, RequestError> receive(const int client, std::span<char> buffer,
                                                const ReadLimits &limits) {
  while (true) {
    if (limits.stopping.load(std::memory_order_relaxed)) {
      return RequestError::closed;
    }

    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        limits.deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      return RequestError::timeout;
    }

    pollfd descriptor{client, POLLIN, 0};
    const auto wait = std::min(remaining, stop_check_interval);
    const int ready = poll(&descriptor, 1, static_cast<int>(wait.count()));
    if (ready == 0) {
      continue;
    }
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }

      return RequestError::closed;
    }

    const auto received = recv(client, buffer.data(), buffer.size(), 0);
    if (received > 0) {
      return static_cast<std::size_t>(received);
    }
    if (received < 0 && errno == EINTR) {
      continue;
    }

    return RequestError::closed;
  }
}

std::variant<HttpRequest, RequestError> read_request(const int client, const ReadLimits &limits) {
  const auto &options = limits.options;
  std::array<char, 16384> buffer{};
  std::string raw;
  std::size_t header_end = std::string::npos;

  while (header_end == std::string::npos) {
    const auto received = receive(client, buffer, limits);
    if (const auto *error = std::get_if<RequestError>(&received)) {
      return *error;
    }

    const auto searched = raw.size() < 3 ? 0 : raw.size() - 3;
    raw.append(buffer.data(), std::get<std::size_t>(received));
    header_end = raw.find("\r\n\r\n", searched);
    if (std::min(header_end, raw.size()) > options.max_header_bytes) {
      return RequestError::headers_too_large;
    }
  }

  auto parsed = parse_head(std::string_view(raw).substr(0, header_end));
  if (const auto *error = std::get_if<RequestError>(&parsed)) {
    return *error;
  }

  auto &head = std::get<RequestHead>(parsed);
  if (head.has_transfer_encoding) {
    return RequestError::transfer_encoding_unsupported;
  }
  if (head.content_length > options.max_body_bytes) {
    return RequestError::body_too_large;
  }

  const auto body_start = header_end + 4;
  while (raw.size() < body_start + head.content_length) {
    const auto received = receive(client, buffer, limits);
    if (const auto *error = std::get_if<RequestError>(&received)) {
      return *error;
    }

    raw.append(buffer.data(), std::get<std::size_t>(received));
  }

  return HttpRequest{std::move(head.method), std::move(head.target), std::move(head.authorization),
                     raw.substr(body_start, head.content_length)};
}

bool send_all(const int fd, const std::string &data) {
  std::size_t sent = 0;
  while (sent < data.size()) {
    const auto result = send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
    if (result <= 0) {
      return false;
    }

    sent += static_cast<std::size_t>(result);
  }
  return true;
}

int create_socket(const int family, const int type, const int protocol, const bool nonblocking) {
  int socket_type = type;
#ifdef SOCK_CLOEXEC
  socket_type |= SOCK_CLOEXEC;
#endif
#ifdef SOCK_NONBLOCK
  if (nonblocking) {
    socket_type |= SOCK_NONBLOCK;
  }
#endif
  const int fd = socket(family, socket_type, protocol);
  if (fd < 0) {
    return fd;
  }
#ifndef SOCK_CLOEXEC
  static_cast<void>(fcntl(fd, F_SETFD, FD_CLOEXEC));
#endif
#ifndef SOCK_NONBLOCK
  if (nonblocking) {
    static_cast<void>(fcntl(fd, F_SETFL, O_NONBLOCK));
  }
#endif
  return fd;
}

int accept_socket(const int listen_fd) {
#ifdef __linux__
  const int client = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
#else
  const int client = accept(listen_fd, nullptr, nullptr);
  if (client >= 0) {
    static_cast<void>(fcntl(client, F_SETFD, FD_CLOEXEC));
  }
#endif
  if (client >= 0) {
    const int flags = fcntl(client, F_GETFL, 0);
    if (flags >= 0) {
      static_cast<void>(fcntl(client, F_SETFL, flags & ~O_NONBLOCK));
    }
  }

  return client;
}

void write_response(const int client, const HttpResponse &response) {
  std::ostringstream headers;
  headers << "HTTP/1.1 " << response.status << ' ' << status_text(response.status) << "\r\n"
          << "Content-Type: " << response.content_type << "\r\n";

  if (!response.stream) {
    if (response.status != 204) {
      headers << "Content-Length: " << response.body.size() << "\r\n";
    }
    headers << "Connection: close\r\n\r\n";
    static_cast<void>(send_all(client, headers.str() + response.body));
    return;
  }

  headers << "Transfer-Encoding: chunked\r\n"
          << "Cache-Control: no-cache\r\n"
          << "X-Accel-Buffering: no\r\n"
          << "Connection: close\r\n\r\n";
  if (!send_all(client, headers.str())) {
    return;
  }

  bool connected = true;
  const HttpChunkWriter write = [&](const std::string_view chunk) {
    if (!connected) {
      return false;
    }

    std::ostringstream frame;
    frame << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
    connected = send_all(client, frame.str());
    return connected;
  };

  response.stream(write);
  if (connected) {
    static_cast<void>(send_all(client, "0\r\n\r\n"));
  }
}

void set_send_timeout(const int client, const std::chrono::seconds timeout) {
  timeval value{};
  value.tv_sec = static_cast<time_t>(timeout.count());
  setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value));
}

template <typename Action> void run_logging_failures(const Action &action) {
  try {
    action();
  } catch (const std::exception &error) {
    std::cerr << "carat: http handler failed: " << error.what() << '\n';
  } catch (...) {
    std::cerr << "carat: http handler failed: unknown exception\n";
  }
}

void report_accept_failure(const int error) {
  if (error == EAGAIN || error == EWOULDBLOCK || error == EINTR || error == ECONNABORTED) {
    return;
  }

  std::cerr << "carat: http accept failed: " << std::strerror(error) << '\n';
  if (error == EMFILE || error == ENFILE) {
    std::this_thread::sleep_for(descriptor_exhaustion_backoff);
  }
}

} // namespace

HttpServer::HttpServer(const std::string &host, const std::uint16_t port, HttpHandler handler,
                       const HttpServerOptions options)
    : handler_(std::move(handler)), options_(options) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    throw std::runtime_error("Carat HTTP bind host must be an IPv4 address");
  }

  // A peer reset can invalidate poll readiness before accept. The listening socket is nonblocking
  // so that a stale readiness edge cannot block the sole accept loop.
  listen_fd_ = create_socket(AF_INET, SOCK_STREAM, 0, true);
  if (listen_fd_ < 0) {
    throw std::runtime_error(std::string("failed to create HTTP socket: ") + std::strerror(errno));
  }

  int enabled = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
  if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      listen(listen_fd_, 64) != 0) {
    const std::string reason = std::strerror(errno);
    close(listen_fd_);
    throw std::runtime_error("failed to listen on HTTP socket: " + reason);
  }
}

HttpServer::~HttpServer() {
  close(listen_fd_);
}

std::uint16_t HttpServer::port() const {
  sockaddr_in address{};
  socklen_t length = sizeof(address);
  if (getsockname(listen_fd_, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
    throw std::runtime_error(std::string("failed to read HTTP socket port: ") +
                             std::strerror(errno));
  }

  return ntohs(address.sin_port);
}

void HttpServer::run() {
  while (!stopping_.load(std::memory_order_relaxed)) {
    pollfd descriptor{listen_fd_, POLLIN, 0};
    if (poll(&descriptor, 1, 500) > 0 && (descriptor.revents & POLLIN) != 0) {
      accept_client();
    }
  }

  std::unique_lock lock(clients_mutex_);
  clients_condition_.wait(lock, [&] { return active_clients_ == 0; });
}

void HttpServer::stop() {
  stopping_.store(true, std::memory_order_relaxed);
}

void HttpServer::accept_client() {
  const int client = accept_socket(listen_fd_);
  if (client < 0) {
    report_accept_failure(errno);
    return;
  }

  set_send_timeout(client, options_.io_timeout);
#ifdef SO_NOSIGPIPE
  int no_sigpipe = 1;
  setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif

  if (!try_admit_client()) {
    write_response(client, {503, "text/plain; charset=utf-8", "server busy\n"});
    close(client);
    return;
  }

  try {
    std::thread([this, client] {
      handle_client(client);
      release_client();
    }).detach();
  } catch (const std::system_error &) {
    std::cerr << "carat: http client thread failed\n";
    close(client);
    release_client();
  }
}

bool HttpServer::try_admit_client() {
  std::lock_guard lock(clients_mutex_);
  if (active_clients_ >= options_.max_clients) {
    return false;
  }

  ++active_clients_;
  return true;
}

void HttpServer::release_client() {
  std::lock_guard lock(clients_mutex_);
  --active_clients_;
  clients_condition_.notify_all();
}

void HttpServer::handle_client(const int client) {
  const ReadLimits limits{
      .options = options_,
      .deadline = std::chrono::steady_clock::now() + options_.io_timeout,
      .stopping = stopping_,
  };
  auto request = read_request(client, limits);
  if (const auto *error = std::get_if<RequestError>(&request)) {
    if (const auto response = error_response(*error)) {
      write_response(client, *response);
    }
    close(client);
    return;
  }

  HttpResponse response{500, "text/plain; charset=utf-8", "internal server error\n"};
  run_logging_failures([&] { response = handler_(std::get<HttpRequest>(request)); });
  run_logging_failures([&] { write_response(client, response); });

  close(client);
}

} // namespace carat::http
