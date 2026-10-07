#include "carat/http.h"

#include "test_support.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using carat::http::testing::contains;
using carat::http::testing::eventually;
using carat::http::testing::raw_http_exchange;
using carat::http::testing::RawHttpConnection;
using carat::http::testing::RecordingServer;
using carat::http::testing::respond_with;
using carat::http::testing::TestHttpServer;
using carat::http::testing::throws;
using carat::http::testing::WriteSide;

constexpr carat::http::HttpServerOptions small_limits{
    .io_timeout = 1s,
    .max_header_bytes = 1024,
    .max_body_bytes = 64,
    .max_clients = 2,
};

HTTP_TEST(port_zero_reports_bound_port) {
  const carat::http::HttpServer server("127.0.0.1", 0, respond_with(200));

  CHECK(server.port() != 0);
}

HTTP_TEST(http_server_bind_to_used_port_throws) {
  const carat::http::HttpServer first("127.0.0.1", 0, respond_with(200));

  CHECK(throws(
      [&] { carat::http::HttpServer second("127.0.0.1", first.port(), respond_with(200)); }));
}

HTTP_TEST(http_server_rejects_non_ipv4_host) {
  CHECK(throws([] { carat::http::HttpServer server("localhost", 0, respond_with(200)); }));
}

HTTP_TEST(http_server_writes_status_line_and_framing_headers) {
  const TestHttpServer server(respond_with(503, "down\n"));

  const auto response = raw_http_exchange(server.port(), "GET / HTTP/1.1\r\n\r\n");

  CHECK(response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
  CHECK(contains(response, "Content-Length: 5\r\n"));
  CHECK(contains(response, "Connection: close\r\n"));
  CHECK(response.ends_with("\r\n\r\ndown\n"));
}

HTTP_TEST(http_server_uses_exact_and_class_reason_phrases) {
  const TestHttpServer no_content(respond_with(204));
  const TestHttpServer teapot(respond_with(418));

  const auto empty = raw_http_exchange(no_content.port(), "GET / HTTP/1.1\r\n\r\n");
  const auto unknown = raw_http_exchange(teapot.port(), "GET / HTTP/1.1\r\n\r\n");

  CHECK(empty.starts_with("HTTP/1.1 204 No Content\r\n"));
  CHECK(!contains(empty, "Content-Length"));
  CHECK(unknown.starts_with("HTTP/1.1 418 Client Error\r\n"));
}

HTTP_TEST(http_server_parses_method_path_authorization_and_exact_body) {
  const RecordingServer server;

  const auto response = raw_http_exchange(
      server.port(),
      "POST /v1/submit?x=1 HTTP/1.0\r\nAUTHORIZATION:   Bearer secret  \r\nContent-Length: 5\r\n"
      "Content-Length: 5\r\n\r\nhelloTRAILING");

  CHECK(response.starts_with("HTTP/1.1 204 No Content\r\n"));
  const auto requests = server.log().requests();
  REQUIRE(requests.size() == 1);
  CHECK(requests[0].method == "POST");
  CHECK(requests[0].path == "/v1/submit?x=1");
  CHECK(requests[0].authorization == "Bearer secret");
  CHECK(requests[0].body == "hello");
}

HTTP_TEST(http_server_reads_body_split_across_packets) {
  const RecordingServer server;

  static_cast<void>(raw_http_exchange(
      server.port(), "POST / HTTP/1.1\r\nContent-Length: 70000\r\n\r\n" + std::string(70000, 'b')));

  const auto requests = server.log().requests();
  REQUIRE(requests.size() == 1);
  CHECK(requests[0].body == std::string(70000, 'b'));
}

HTTP_TEST(malformed_requests_are_bad_request_and_not_dispatched) {
  const RecordingServer server(small_limits);

  for (const auto *request : {
           "garbage\r\n\r\n",
           "GET /\r\n\r\n",
           "GET  / HTTP/1.1\r\n\r\n",
           "GET / HTTP/1.1 extra\r\n\r\n",
           "GET relative HTTP/1.1\r\n\r\n",
           "GET / HTTP/2.0\r\n\r\n",
           "GET / HTTP/1.1\r\nNo-Colon\r\n\r\n",
           "GET / HTTP/1.1\r\nBad Name: x\r\n\r\n",
           "GET / HTTP/1.1\r\n: empty-name\r\n\r\n",
           "GET / HTTP/1.1\r\nX: a\nContent-Length: 3\r\n\r\n",
           "POST / HTTP/1.1\r\nContent-Length: 12abc\r\n\r\n",
           "POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n",
           "POST / HTTP/1.1\r\nContent-Length:\r\n\r\n",
           "POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab",
       }) {
    const auto response = raw_http_exchange(server.port(), request);

    CHECK(response.starts_with("HTTP/1.1 400 Bad Request\r\n"));
  }
  CHECK(server.log().size() == 0);
}

HTTP_TEST(headers_over_the_limit_are_431) {
  const RecordingServer server(small_limits);

  const auto unterminated =
      raw_http_exchange(server.port(), "GET / HTTP/1.1\r\nX-Padding: " + std::string(2048, 'h'));
  const auto terminated = raw_http_exchange(
      server.port(), "GET / HTTP/1.1\r\nX-Padding: " + std::string(2048, 'h') + "\r\n\r\n");

  CHECK(unterminated.starts_with("HTTP/1.1 431 Request Header Fields Too Large\r\n"));
  CHECK(terminated.starts_with("HTTP/1.1 431 Request Header Fields Too Large\r\n"));
  CHECK(server.log().size() == 0);
}

HTTP_TEST(content_length_over_the_limit_is_413) {
  const RecordingServer server(small_limits);

  const auto response =
      raw_http_exchange(server.port(), "POST / HTTP/1.1\r\nContent-Length: 65\r\n\r\n");

  CHECK(response.starts_with("HTTP/1.1 413 Content Too Large\r\n"));
  CHECK(server.log().size() == 0);
}

HTTP_TEST(transfer_encoding_is_501_instead_of_an_empty_body) {
  const RecordingServer server(small_limits);

  const auto response = raw_http_exchange(
      server.port(),
      "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n");

  CHECK(response.starts_with("HTTP/1.1 501 Not Implemented\r\n"));
  CHECK(server.log().size() == 0);
}

HTTP_TEST(truncated_body_gets_no_response_and_server_keeps_serving) {
  const RecordingServer server(small_limits);

  const auto truncated = raw_http_exchange(
      server.port(), "POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc", WriteSide::shut_down);
  const auto partial_head =
      raw_http_exchange(server.port(), "GET / HTTP/1.1\r\n", WriteSide::shut_down);
  const auto next = raw_http_exchange(server.port(), "GET / HTTP/1.1\r\n\r\n");

  CHECK(truncated.empty());
  CHECK(partial_head.empty());
  CHECK(next.starts_with("HTTP/1.1 204 No Content\r\n"));
  CHECK(server.log().size() == 1);
}

HTTP_TEST(silent_client_gets_408_after_the_io_timeout) {
  const TestHttpServer server(respond_with(200), small_limits);
  RawHttpConnection connection(server.port());

  const auto started = std::chrono::steady_clock::now();
  const auto response = connection.receive_all();
  const auto elapsed = std::chrono::steady_clock::now() - started;

  CHECK(response.starts_with("HTTP/1.1 408 Request Timeout\r\n"));
  CHECK(elapsed >= 900ms && elapsed < 2500ms);
}

HTTP_TEST(throwing_handler_is_500_without_exception_text) {
  const TestHttpServer server([](const carat::http::HttpRequest &) -> carat::http::HttpResponse {
    throw std::runtime_error("secret internal detail");
  });

  const auto response = raw_http_exchange(server.port(), "GET / HTTP/1.1\r\n\r\n");

  CHECK(response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"));
  CHECK(!contains(response, "secret internal detail"));
}

HTTP_TEST(streamed_response_uses_chunked_framing) {
  const TestHttpServer server([](const carat::http::HttpRequest &) {
    return carat::http::HttpResponse{
        200, "text/event-stream", {}, [](const carat::http::HttpChunkWriter &write) {
          CHECK(write("data: one\n\n"));
          CHECK(write("data: [DONE]\n\n"));
        }};
  });

  const auto response = raw_http_exchange(server.port(), "GET /stream HTTP/1.1\r\n\r\n");

  CHECK(response.starts_with("HTTP/1.1 200 OK\r\n"));
  CHECK(contains(response, "Transfer-Encoding: chunked\r\n"));
  CHECK(!contains(response, "Content-Length"));
  CHECK(response.ends_with("\r\n\r\nb\r\ndata: one\n\n\r\ne\r\ndata: [DONE]\n\n\r\n0\r\n\r\n"));
}

HTTP_TEST(stream_writer_reports_a_disconnected_client) {
  std::atomic<bool> write_failed{false};
  const TestHttpServer server([&](const carat::http::HttpRequest &) {
    return carat::http::HttpResponse{
        200, "text/event-stream", {}, [&](const carat::http::HttpChunkWriter &write) {
          const std::string chunk(64 * 1024, 'x');
          for (int attempt = 0; attempt < 400; ++attempt) {
            if (!write(chunk)) {
              write_failed = true;
              CHECK(!write(chunk));
              return;
            }
            std::this_thread::sleep_for(5ms);
          }
        }};
  });

  {
    RawHttpConnection connection(server.port());
    connection.send("GET / HTTP/1.1\r\n\r\n");
  }

  CHECK(eventually([&] { return write_failed.load(); }));
}

HTTP_TEST(clients_beyond_the_limit_get_503) {
  const TestHttpServer server(respond_with(200), small_limits);
  RawHttpConnection first(server.port());
  RawHttpConnection second(server.port());

  RawHttpConnection third(server.port());

  CHECK(third.receive_all().starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
}

HTTP_TEST(run_returns_only_after_an_in_flight_stream_finishes) {
  std::atomic<bool> streaming{false};
  std::atomic<bool> finished{false};
  carat::http::HttpServer server("127.0.0.1", 0, [&](const carat::http::HttpRequest &) {
    return carat::http::HttpResponse{
        200, "text/event-stream", {}, [&](const carat::http::HttpChunkWriter &write) {
          streaming = true;
          std::this_thread::sleep_for(300ms);
          static_cast<void>(write("done"));
          finished = true;
        }};
  });
  std::thread thread([&] { server.run(); });
  RawHttpConnection connection(server.port());
  connection.send("GET / HTTP/1.1\r\n\r\n");
  CHECK(eventually([&] { return streaming.load(); }));

  server.stop();
  thread.join();

  CHECK(finished.load());
  CHECK(contains(connection.receive_all(), "done"));
}

HTTP_TEST(run_does_not_wait_for_an_idle_client_after_stop) {
  carat::http::HttpServer server("127.0.0.1", 0, respond_with(200));
  std::thread thread([&] { server.run(); });
  RawHttpConnection idle(server.port());
  CHECK(raw_http_exchange(server.port(), "GET / HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 200"));

  const auto started = std::chrono::steady_clock::now();
  server.stop();
  thread.join();

  CHECK(std::chrono::steady_clock::now() - started < 2s);
  CHECK(idle.receive_all().empty());
}

HTTP_TEST(http_server_remains_responsive_after_a_connection_burst) {
  const TestHttpServer server(respond_with(200, "ready\n"));

  std::atomic<int> failures{0};
  std::vector<std::thread> clients;
  for (int worker = 0; worker < 32; ++worker) {
    clients.emplace_back([&] {
      for (int request = 0; request < 16; ++request) {
        const auto response = raw_http_exchange(server.port(), "GET /health HTTP/1.1\r\n\r\n");
        if (!response.starts_with("HTTP/1.1 200 OK\r\n")) {
          failures.fetch_add(1);
        }
      }
    });
  }
  for (auto &client : clients) {
    client.join();
  }

  CHECK(failures.load() == 0);
  CHECK(raw_http_exchange(server.port(), "GET / HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 200"));
}

} // namespace
