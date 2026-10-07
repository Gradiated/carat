#include "test_support.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace carat::http::testing {
namespace {

struct RegisteredTest {
  const char *name;
  TestFunction function;
};

std::vector<RegisteredTest> &registered_tests() {
  static std::vector<RegisteredTest> tests;
  return tests;
}

int failures = 0;

int connect_loopback(const std::uint16_t port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
    close(fd);
    return -1;
  }

  return fd;
}

void wake_accept_loop(const std::uint16_t port) {
  const int connection = connect_loopback(port);
  if (connection >= 0) {
    close(connection);
  }
}

} // namespace

TestRegistration::TestRegistration(const char *name, const TestFunction function) {
  registered_tests().push_back({name, function});
}

bool contains(const std::string_view text, const std::string_view fragment) {
  return text.find(fragment) != std::string_view::npos;
}

void record_failure(const char *file, const int line, const char *assertion) {
  ++failures;
  std::cerr << file << ':' << line << ": " << assertion << " failed\n";
}

TestHttpServer::TestHttpServer(HttpHandler handler, const HttpServerOptions options)
    : server_("127.0.0.1", 0, std::move(handler), options), thread_([this] { server_.run(); }) {}

TestHttpServer::~TestHttpServer() {
  server_.stop();
  wake_accept_loop(port());
  thread_.join();
}

std::uint16_t TestHttpServer::port() const {
  return server_.port();
}

HttpHandler respond_with(const int status, std::string body) {
  return [status, body = std::move(body)](const HttpRequest &) {
    return HttpResponse{status, "text/plain; charset=utf-8", body};
  };
}

void RequestLog::record(const HttpRequest &request) {
  std::lock_guard lock(mutex_);
  requests_.push_back(request);
}

std::vector<HttpRequest> RequestLog::requests() const {
  std::lock_guard lock(mutex_);
  return requests_;
}

std::size_t RequestLog::size() const {
  std::lock_guard lock(mutex_);
  return requests_.size();
}

RecordingServer::RecordingServer(const HttpServerOptions options)
    : server_(
          [this](const HttpRequest &request) {
            log_.record(request);
            return HttpResponse{next_status(), "application/json", ""};
          },
          options) {}

void RecordingServer::respond_with(const int status) {
  respond_in_sequence({status});
}

void RecordingServer::respond_in_sequence(std::vector<int> statuses) {
  std::lock_guard lock(statuses_mutex_);
  statuses_ = std::move(statuses);
  responses_ = 0;
}

int RecordingServer::next_status() {
  std::lock_guard lock(statuses_mutex_);
  const auto index = std::min(responses_++, statuses_.size() - 1);
  return statuses_[index];
}

const RequestLog &RecordingServer::log() const {
  return log_;
}

std::uint16_t RecordingServer::port() const {
  return server_.port();
}

RawHttpConnection::RawHttpConnection(const std::uint16_t port) : fd_(connect_loopback(port)) {
  if (fd_ < 0) {
    throw std::runtime_error("failed to connect test socket");
  }

  timeval timeout{5, 0};
  setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
}

RawHttpConnection::~RawHttpConnection() {
  close(fd_);
}

void RawHttpConnection::send(const std::string_view bytes) {
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const auto written = ::send(fd_, bytes.data() + sent, bytes.size() - sent, 0);
    if (written <= 0) {
      return;
    }

    sent += static_cast<std::size_t>(written);
  }
}

void RawHttpConnection::close_write() {
  shutdown(fd_, SHUT_WR);
}

std::string RawHttpConnection::receive_all() {
  std::string response;
  std::array<char, 4096> buffer{};
  while (true) {
    const auto received = recv(fd_, buffer.data(), buffer.size(), 0);
    if (received <= 0) {
      return response;
    }

    response.append(buffer.data(), static_cast<std::size_t>(received));
  }
}

std::string raw_http_exchange(const std::uint16_t port, const std::string_view request,
                              const WriteSide write_side) {
  RawHttpConnection connection(port);
  connection.send(request);
  if (write_side == WriteSide::shut_down) {
    connection.close_write();
  }

  return connection.receive_all();
}

ListeningSocket::ListeningSocket() : fd_(socket(AF_INET, SOCK_STREAM, 0)) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (fd_ < 0 || bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      listen(fd_, 16) != 0) {
    if (fd_ >= 0) {
      close(fd_);
    }
    throw std::runtime_error("failed to create test listener");
  }
}

ListeningSocket::~ListeningSocket() {
  close(fd_);
}

int ListeningSocket::fd() const {
  return fd_;
}

std::uint16_t ListeningSocket::port() const {
  sockaddr_in address{};
  socklen_t length = sizeof(address);
  getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &length);
  return ntohs(address.sin_port);
}

ScriptedListener::ScriptedListener(std::string reply)
    : reply_(std::move(reply)), thread_([this] {
        while (!stopping_.load()) {
          pollfd descriptor{socket_.fd(), POLLIN, 0};
          if (poll(&descriptor, 1, 50) <= 0) {
            continue;
          }

          const int client = accept(socket_.fd(), nullptr, nullptr);
          if (client < 0) {
            continue;
          }

          timeval timeout{1, 0};
          setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
          std::array<char, 65536> request{};
          static_cast<void>(recv(client, request.data(), request.size(), 0));
          static_cast<void>(send(client, reply_.data(), reply_.size(), 0));
          close(client);
        }
      }) {}

ScriptedListener::~ScriptedListener() {
  stopping_.store(true);
  thread_.join();
}

std::uint16_t ScriptedListener::port() const {
  return socket_.port();
}

std::uint16_t closed_port() {
  const ListeningSocket socket;
  return socket.port();
}

std::string local_url(const std::uint16_t port, const std::string_view path,
                      const std::string_view scheme) {
  return std::string(scheme) + "://127.0.0.1:" + std::to_string(port) + std::string(path);
}

TempDirectory::TempDirectory() {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "carat-http-test-XXXXXX").string();
  if (mkdtemp(pattern.data()) == nullptr) {
    throw std::runtime_error("failed to create temp directory");
  }

  path_ = std::move(pattern);
}

TempDirectory::~TempDirectory() {
  std::error_code ignored;
  std::filesystem::remove_all(path_, ignored);
}

std::string TempDirectory::file(const std::string_view name) const {
  return (std::filesystem::path(path_) / name).string();
}

} // namespace carat::http::testing

int main(const int argc, char **argv) {
  const std::string_view only = argc > 1 ? argv[1] : "";
  for (const auto &test : carat::http::testing::registered_tests()) {
    if (!only.empty() && only != test.name) {
      continue;
    }

    const int failures_before = carat::http::testing::failures;
    const auto started = std::chrono::steady_clock::now();
    try {
      test.function();
    } catch (const carat::http::testing::RequirementFailed &) {
    } catch (const std::exception &error) {
      ++carat::http::testing::failures;
      std::cerr << test.name << ": unexpected exception: " << error.what() << '\n';
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
    std::cout << (carat::http::testing::failures == failures_before ? "PASS " : "FAIL ")
              << test.name << " (" << std::fixed << std::setprecision(1) << elapsed.count()
              << "s)\n";
  }
  return carat::http::testing::failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
