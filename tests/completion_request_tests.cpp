#include "runtime/completion_request.h"

#include "support/test_support.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace {

using carat::BearerAuthorization;
using carat::CompletionLimits;
using carat::parse_completion_request;

const std::vector<int> eos{1, 106};
const CompletionLimits limits{
    .maximum_context = 256, .vocabulary_size = 200, .implicit_stop_ids = eos};
const CompletionLimits tight{
    .maximum_context = 16, .vocabulary_size = 200, .implicit_stop_ids = eos};

ENGINE_TEST(completion_request_applies_defaults_and_the_implicit_stops) {
  const auto request = parse_completion_request(R"({"input_ids":[2,3]})", limits);

  CHECK(request.input_ids == std::vector<int>({2, 3}));
  CHECK(request.max_tokens == 128);
  CHECK(request.stop_ids == std::unordered_set<int>({1, 106}));
  CHECK(!request.stream);
  CHECK(request.priority == 0);
}

ENGINE_TEST(completion_request_reads_every_field) {
  const auto request = parse_completion_request(
      R"({"input_ids":[0,199],"max_tokens":4,"stop_token_ids":[7,106],"stream":true,"priority":3})",
      limits);

  CHECK(request.input_ids == std::vector<int>({0, 199}));
  CHECK(request.max_tokens == 4);
  CHECK(request.stop_ids == std::unordered_set<int>({1, 7, 106}));
  CHECK(request.stream);
  CHECK(request.priority == 3);
}

ENGINE_TEST(completion_request_without_implicit_stops_uses_only_the_requested_stops) {
  const CompletionLimits no_eos{
      .maximum_context = 256, .vocabulary_size = 200, .implicit_stop_ids = {}};

  CHECK(parse_completion_request(R"({"input_ids":[2]})", no_eos).stop_ids.empty());
  CHECK(parse_completion_request(R"({"input_ids":[2],"stop_token_ids":[9]})", no_eos).stop_ids ==
        std::unordered_set<int>({9}));
}

ENGINE_TEST(completion_request_rejects_invalid_input_ids) {
  CHECK_THROWS(parse_completion_request(R"({})", limits), "input_ids");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[]})", limits),
               "input_ids must not be empty");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[200]})", limits),
               "input_ids has an invalid token id");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[-1]})", limits),
               "input_ids has an invalid token id");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":"2"})", limits), "");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2.5]})", limits), "");
  CHECK_THROWS(
      parse_completion_request(R"({"input_ids":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]})", tight),
      "input_ids exceeds the context");
}

ENGINE_TEST(completion_request_rejects_max_tokens_beyond_the_remaining_context) {
  const std::string fifteen_tokens = R"("input_ids":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])";

  CHECK(parse_completion_request("{" + fifteen_tokens + R"(,"max_tokens":1})", tight).max_tokens ==
        1);
  CHECK_THROWS(parse_completion_request("{" + fifteen_tokens + "}", tight),
               "max_tokens exceeds the remaining context");
  CHECK_THROWS(parse_completion_request("{" + fifteen_tokens + R"(,"max_tokens":2})", tight),
               "max_tokens exceeds the remaining context");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2],"max_tokens":0})", limits),
               "max_tokens is out of range");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2],"max_tokens":-1})", limits),
               "max_tokens is out of range");
}

ENGINE_TEST(completion_request_rejects_invalid_options) {
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2],"stop_token_ids":[200]})", limits),
               "stop_token_ids has an invalid token id");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2],"stream":"yes"})", limits), "");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2],"priority":-1})", limits),
               "priority is out of range");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2],"priority":1000001})", limits),
               "priority is out of range");
  CHECK_THROWS(parse_completion_request(R"({"input_ids":[2])", limits), "");
}

ENGINE_TEST(authorization_without_an_api_key_permits_every_request) {
  const BearerAuthorization authorization(std::nullopt);

  CHECK(authorization.permits(""));
  CHECK(authorization.permits("Bearer anything"));
}

ENGINE_TEST(authorization_permits_only_the_exact_bearer_token) {
  const BearerAuthorization authorization(std::string("secret"));

  CHECK(authorization.permits("Bearer secret"));
  CHECK(!authorization.permits(""));
  CHECK(!authorization.permits("secret"));
  CHECK(!authorization.permits("Bearer secre"));
  CHECK(!authorization.permits("Bearer secret2"));
  CHECK(!authorization.permits("Bearer secreT"));
  CHECK(!authorization.permits("bearer secret"));
}

} // namespace
