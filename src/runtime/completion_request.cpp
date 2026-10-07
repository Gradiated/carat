#include "runtime/completion_request.h"

#include "carat/json.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace carat {
namespace {

constexpr int default_max_tokens = 128;
constexpr int maximum_integer_field = 1000000;

std::vector<int> token_ids(const Json &value, std::string_view field, int vocabulary_size) {
  std::vector<int> result;
  result.reserve(value.as_array().size());
  for (const Json &item : value.as_array()) {
    const std::int64_t parsed = item.as_i64();
    if (parsed < 0 || parsed >= vocabulary_size) {
      throw std::runtime_error(std::string(field) + " has an invalid token id");
    }

    result.push_back(static_cast<int>(parsed));
  }
  return result;
}

int optional_integer(const Json &object, std::string_view field, int fallback, int minimum,
                     int maximum) {
  const Json *value = object.find(field);
  if (value == nullptr)
    return fallback;

  const std::int64_t parsed = value->as_i64();
  if (parsed < minimum || parsed > maximum)
    throw std::runtime_error(std::string(field) + " is out of range");

  return static_cast<int>(parsed);
}

} // namespace

CompletionRequest parse_completion_request(std::string_view body, const CompletionLimits &limits) {
  const Json root = Json::parse(body);
  CompletionRequest request;

  request.input_ids = token_ids(root.at("input_ids"), "input_ids", limits.vocabulary_size);
  if (request.input_ids.empty())
    throw std::runtime_error("input_ids must not be empty");

  const auto input_tokens = static_cast<int>(request.input_ids.size());
  if (input_tokens >= limits.maximum_context)
    throw std::runtime_error("input_ids exceeds the context");

  request.max_tokens =
      optional_integer(root, "max_tokens", default_max_tokens, 1, maximum_integer_field);
  if (request.max_tokens > limits.maximum_context - input_tokens) {
    throw std::runtime_error("max_tokens exceeds the remaining context");
  }

  request.stop_ids.insert(limits.implicit_stop_ids.begin(), limits.implicit_stop_ids.end());
  if (const Json *stop_ids = root.find("stop_token_ids")) {
    for (const int token : token_ids(*stop_ids, "stop_token_ids", limits.vocabulary_size)) {
      request.stop_ids.insert(token);
    }
  }

  const Json *stream = root.find("stream");
  request.stream = stream != nullptr && stream->as_bool();
  request.priority = optional_integer(root, "priority", 0, 0, maximum_integer_field);
  return request;
}

BearerAuthorization::BearerAuthorization(const std::optional<std::string> &api_key) {
  if (api_key)
    expected_header_ = "Bearer " + *api_key;
}

bool BearerAuthorization::permits(std::string_view authorization_header) const {
  if (!expected_header_)
    return true;

  const std::string_view expected = *expected_header_;
  const auto byte_at = [](std::string_view text, std::size_t index) {
    return index < text.size() ? static_cast<unsigned char>(text[index]) : 0U;
  };

  std::size_t difference = authorization_header.size() ^ expected.size();
  const std::size_t length = std::max(authorization_header.size(), expected.size());
  for (std::size_t index = 0; index < length; ++index) {
    difference |=
        static_cast<std::size_t>(byte_at(authorization_header, index) ^ byte_at(expected, index));
  }
  return difference == 0;
}

} // namespace carat
