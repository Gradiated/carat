#include "carat/json.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace carat {
namespace {

constexpr std::size_t maximum_depth = 64;

class Parser {
public:
  explicit Parser(std::string_view input) : input_(input) {}

  Json parse_document() {
    Json result = parse_value();
    whitespace();
    if (position_ != input_.size()) {
      fail("trailing input");
    }
    return result;
  }

private:
  [[noreturn]] void fail(const std::string &message) const {
    throw std::runtime_error("JSON byte " + std::to_string(position_) + ": " + message);
  }

  void whitespace() {
    while (position_ < input_.size()) {
      const char c = input_[position_];
      if (c != ' ' && c != '\n' && c != '\r' && c != '\t')
        break;

      ++position_;
    }
  }

  bool consume(char expected) {
    whitespace();
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  class NestingGuard {
  public:
    explicit NestingGuard(Parser &parser) : parser_(parser) {
      parser_.whitespace();
      if (parser_.depth_ == maximum_depth)
        parser_.fail("nesting too deep");

      ++parser_.depth_;
    }
    ~NestingGuard() {
      --parser_.depth_;
    }
    NestingGuard(const NestingGuard &) = delete;
    NestingGuard &operator=(const NestingGuard &) = delete;

  private:
    Parser &parser_;
  };

  void literal(std::string_view expected) {
    if (input_.substr(position_, expected.size()) != expected)
      fail("invalid literal");
    position_ += expected.size();
  }

  Json parse_value() {
    whitespace();
    if (position_ == input_.size())
      fail("expected value");
    switch (input_[position_]) {
    case 'n':
      literal("null");
      return Json(nullptr);
    case 't':
      literal("true");
      return Json(true);
    case 'f':
      literal("false");
      return Json(false);
    case '"':
      return Json(parse_string());
    case '[':
      return parse_array();
    case '{':
      return parse_object();
    default:
      if (input_[position_] == '-' || (input_[position_] >= '0' && input_[position_] <= '9')) {
        return parse_number();
      }

      fail("expected value");
    }
  }

  static void append_utf8(std::string &output, std::uint32_t codepoint) {
    if (codepoint <= 0x7fU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
      output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
      output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
  }

  std::uint32_t hex4() {
    if (position_ + 4 > input_.size())
      fail("short unicode escape");
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      const char c = input_[position_++];
      value <<= 4U;
      if (c >= '0' && c <= '9')
        value |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        value |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        value |= static_cast<std::uint32_t>(c - 'A' + 10);
      else
        fail("invalid unicode escape");
    }
    return value;
  }

  std::string parse_string() {
    if (!consume('"'))
      fail("expected string");
    std::string output;
    while (position_ < input_.size()) {
      const char c = input_[position_++];
      if (c == '"')
        return output;

      if (static_cast<unsigned char>(c) < 0x20U)
        fail("control character in string");
      if (c != '\\') {
        output.push_back(c);
        continue;
      }
      if (position_ == input_.size())
        fail("short escape");
      switch (input_[position_++]) {
      case '"':
        output.push_back('"');
        break;
      case '\\':
        output.push_back('\\');
        break;
      case '/':
        output.push_back('/');
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      case 'u': {
        std::uint32_t codepoint = hex4();
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
          if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
              input_[position_ + 1] != 'u')
            fail("missing low surrogate");
          position_ += 2;
          const std::uint32_t low = hex4();
          if (low < 0xdc00U || low > 0xdfffU)
            fail("invalid low surrogate");
          codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
          fail("unexpected low surrogate");
        }
        append_utf8(output, codepoint);
        break;
      }
      default:
        fail("invalid escape");
      }
    }
    fail("unterminated string");
  }

  Json parse_number() {
    whitespace();
    const std::size_t begin = position_;
    if (input_[position_] == '-')
      ++position_;
    if (position_ == input_.size())
      fail("short number");
    if (input_[position_] == '0') {
      ++position_;
    } else {
      if (input_[position_] < '1' || input_[position_] > '9')
        fail("invalid number");
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
        ++position_;
    }
    bool integral = true;
    if (position_ < input_.size() && input_[position_] == '.') {
      integral = false;
      ++position_;
      const std::size_t digits = position_;
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
        ++position_;
      if (digits == position_)
        fail("missing fractional digits");
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      integral = false;
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-'))
        ++position_;
      const std::size_t digits = position_;
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
        ++position_;
      if (digits == position_)
        fail("missing exponent digits");
    }
    const std::string_view token = input_.substr(begin, position_ - begin);
    if (integral) {
      std::int64_t value = 0;
      const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
      if (result.ec != std::errc{})
        fail("integer out of range");
      return Json(value);
    }
    const std::string owned(token);
    char *end = nullptr;
    const double value = std::strtod(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size() || !std::isfinite(value)) {
      fail("invalid floating-point number");
    }
    return Json(value);
  }

  Json parse_array() {
    const NestingGuard nesting(*this);
    if (!consume('['))
      fail("expected array");
    Json::Array values;
    if (consume(']'))
      return Json(std::move(values));

    do {
      values.push_back(parse_value());
    } while (consume(','));
    if (!consume(']'))
      fail("expected array end");
    return Json(std::move(values));
  }

  Json parse_object() {
    const NestingGuard nesting(*this);
    if (!consume('{'))
      fail("expected object");
    Json::Object values;
    if (consume('}'))
      return Json(std::move(values));

    do {
      whitespace();
      std::string key = parse_string();
      if (!consume(':'))
        fail("expected colon");
      if (!values.emplace(std::move(key), parse_value()).second)
        fail("duplicate object key");
    } while (consume(','));
    if (!consume('}'))
      fail("expected object end");
    return Json(std::move(values));
  }

  std::string_view input_;
  std::size_t position_{0};
  std::size_t depth_{0};
};

template <typename T> const T &get(const Json::Value &value, const char *expected) {
  const auto *result = std::get_if<T>(&value);
  if (result == nullptr)
    throw std::runtime_error(std::string("JSON value is not ") + expected);

  return *result;
}

} // namespace

Json Json::parse(std::string_view input) {
  return Parser(input).parse_document();
}

bool Json::is_null() const {
  return std::holds_alternative<std::nullptr_t>(value_);
}

bool Json::as_bool() const {
  return get<bool>(value_, "a boolean");
}

std::int64_t Json::as_i64() const {
  return get<std::int64_t>(value_, "an integer");
}

std::uint64_t Json::as_u64() const {
  const auto value = as_i64();
  if (value < 0)
    throw std::runtime_error("JSON integer is negative");

  return static_cast<std::uint64_t>(value);
}

double Json::as_double() const {
  if (const auto *value = std::get_if<double>(&value_))
    return *value;
  if (const auto *value = std::get_if<std::int64_t>(&value_))
    return static_cast<double>(*value);

  throw std::runtime_error("JSON value is not a number");
}

const std::string &Json::as_string() const {
  return get<std::string>(value_, "a string");
}
const Json::Array &Json::as_array() const {
  return get<Array>(value_, "an array");
}
const Json::Object &Json::as_object() const {
  return get<Object>(value_, "an object");
}

const Json &Json::at(std::string_view key) const {
  const auto &object = as_object();
  const auto iterator = object.find(key);
  if (iterator == object.end())
    throw std::runtime_error("missing JSON key: " + std::string(key));

  return iterator->second;
}

const Json *Json::find(std::string_view key) const {
  const auto &object = as_object();
  const auto iterator = object.find(key);
  return iterator == object.end() ? nullptr : &iterator->second;
}

std::string read_text_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open " + path);

  std::ostringstream output;
  output << input.rdbuf();
  if (!input.eof() && input.fail())
    throw std::runtime_error("cannot read " + path);

  return output.str();
}

} // namespace carat
