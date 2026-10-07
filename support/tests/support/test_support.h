#pragma once

#include "carat/http.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace carat::http::testing {

using TestFunction = void (*)();

struct TestRegistration {
  TestRegistration(const char *name, TestFunction function);
};

struct RequirementFailed {};

void record_failure(const char *file, int line, const char *assertion);

[[nodiscard]] bool contains(std::string_view text, std::string_view fragment);

template <typename Function> [[nodiscard]] bool throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

template <typename Predicate>
[[nodiscard]] bool eventually(Predicate &&predicate,
                              const std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

class TestHttpServer {
public:
  explicit TestHttpServer(HttpHandler handler, HttpServerOptions options = {});
  ~TestHttpServer();
  TestHttpServer(const TestHttpServer &) = delete;
  TestHttpServer &operator=(const TestHttpServer &) = delete;

  [[nodiscard]] std::uint16_t port() const;

private:
  HttpServer server_;
  std::thread thread_;
};

[[nodiscard]] HttpHandler respond_with(int status, std::string body = {});

class RequestLog {
public:
  void record(const HttpRequest &request);
  [[nodiscard]] std::vector<HttpRequest> requests() const;
  [[nodiscard]] std::size_t size() const;

private:
  mutable std::mutex mutex_;
  std::vector<HttpRequest> requests_;
};

class RecordingServer {
public:
  explicit RecordingServer(HttpServerOptions options = {});

  void respond_with(int status);
  void respond_in_sequence(std::vector<int> statuses);
  [[nodiscard]] const RequestLog &log() const;
  [[nodiscard]] std::uint16_t port() const;

private:
  [[nodiscard]] int next_status();

  RequestLog log_;
  mutable std::mutex statuses_mutex_;
  std::vector<int> statuses_{204};
  std::size_t responses_{0};
  TestHttpServer server_;
};

class RawHttpConnection {
public:
  explicit RawHttpConnection(std::uint16_t port);
  ~RawHttpConnection();
  RawHttpConnection(const RawHttpConnection &) = delete;
  RawHttpConnection &operator=(const RawHttpConnection &) = delete;

  void send(std::string_view bytes);
  void close_write();
  [[nodiscard]] std::string receive_all();

private:
  int fd_;
};

enum class WriteSide { keep_open, shut_down };

[[nodiscard]] std::string raw_http_exchange(std::uint16_t port, std::string_view request,
                                            WriteSide write_side = WriteSide::keep_open);

class ListeningSocket {
public:
  ListeningSocket();
  ~ListeningSocket();
  ListeningSocket(const ListeningSocket &) = delete;
  ListeningSocket &operator=(const ListeningSocket &) = delete;

  [[nodiscard]] int fd() const;
  [[nodiscard]] std::uint16_t port() const;

private:
  int fd_;
};

class ScriptedListener {
public:
  explicit ScriptedListener(std::string reply);
  ~ScriptedListener();
  ScriptedListener(const ScriptedListener &) = delete;
  ScriptedListener &operator=(const ScriptedListener &) = delete;

  [[nodiscard]] std::uint16_t port() const;

private:
  ListeningSocket socket_;
  std::string reply_;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};

[[nodiscard]] std::uint16_t closed_port();

[[nodiscard]] std::string local_url(std::uint16_t port, std::string_view path = "/",
                                    std::string_view scheme = "http");

class TempDirectory {
public:
  TempDirectory();
  ~TempDirectory();
  TempDirectory(const TempDirectory &) = delete;
  TempDirectory &operator=(const TempDirectory &) = delete;

  [[nodiscard]] std::string file(std::string_view name) const;

private:
  std::string path_;
};

} // namespace carat::http::testing

#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      ::carat::http::testing::record_failure(__FILE__, __LINE__, "CHECK(" #condition ")");         \
    }                                                                                              \
  } while (false)

#define REQUIRE(condition)                                                                         \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      ::carat::http::testing::record_failure(__FILE__, __LINE__, "REQUIRE(" #condition ")");       \
      throw ::carat::http::testing::RequirementFailed{};                                           \
    }                                                                                              \
  } while (false)

#define HTTP_TEST(name)                                                                            \
  static void name();                                                                              \
  static const ::carat::http::testing::TestRegistration name##_registration{#name, name};          \
  static void name()
