#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace carat::http {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string authorization;
  std::string body;
};

using HttpChunkWriter = std::function<bool(std::string_view)>;
using HttpStream = std::function<void(const HttpChunkWriter &)>;

struct HttpResponse {
  int status{200};
  std::string content_type{"text/plain; charset=utf-8"};
  std::string body;
  HttpStream stream;

  HttpResponse() = default;
  HttpResponse(int response_status, std::string response_content_type, std::string response_body,
               HttpStream response_stream = {})
      : status(response_status), content_type(std::move(response_content_type)),
        body(std::move(response_body)), stream(std::move(response_stream)) {}
};

using HttpHandler = std::function<HttpResponse(const HttpRequest &)>;

struct HttpServerOptions {
  std::chrono::seconds io_timeout{30};
  std::size_t max_header_bytes{64 * 1024};
  std::size_t max_body_bytes{1024 * 1024};
  std::size_t max_clients{1024};
};

class HttpServer {
public:
  HttpServer(const std::string &host, std::uint16_t port, HttpHandler handler,
             HttpServerOptions options = {});
  ~HttpServer();
  HttpServer(const HttpServer &) = delete;
  HttpServer &operator=(const HttpServer &) = delete;

  [[nodiscard]] std::uint16_t port() const;
  void run();
  void stop();

private:
  void accept_client();
  void handle_client(int client);
  [[nodiscard]] bool try_admit_client();
  void release_client();

  HttpHandler handler_;
  HttpServerOptions options_;
  int listen_fd_{-1};
  std::atomic<bool> stopping_{false};
  std::mutex clients_mutex_;
  std::condition_variable clients_condition_;
  std::size_t active_clients_{0};
};

} // namespace carat::http
