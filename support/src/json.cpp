#include "carat/json.h"

#include <array>

namespace carat::http {

std::string json_string(const std::string_view value) {
  constexpr std::array<char, 16> hex{'0', '1', '2', '3', '4', '5', '6', '7',
                                     '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

  std::string output;
  output.reserve(value.size() + 2);
  output += '"';
  for (const char character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (character) {
    case '"':
      output += "\\\"";
      break;
    case '\\':
      output += "\\\\";
      break;
    case '\b':
      output += "\\b";
      break;
    case '\f':
      output += "\\f";
      break;
    case '\n':
      output += "\\n";
      break;
    case '\r':
      output += "\\r";
      break;
    case '\t':
      output += "\\t";
      break;
    default:
      if (byte < 0x20) {
        output += "\\u00";
        output += hex[byte >> 4];
        output += hex[byte & 0x0f];
      } else {
        output += character;
      }
    }
  }
  output += '"';
  return output;
}

JsonObjectWriter &JsonObjectWriter::field(const std::string_view name, const bool value) {
  return raw_field(name, value ? "true" : "false");
}

JsonObjectWriter &JsonObjectWriter::field(const std::string_view name,
                                          const std::string_view value) {
  return raw_field(name, json_string(value));
}

JsonObjectWriter &JsonObjectWriter::field(const std::string_view name, const char *value) {
  return field(name, std::string_view(value));
}

JsonObjectWriter &JsonObjectWriter::raw_field(const std::string_view name,
                                              const std::string_view json) {
  body_ += body_.empty() ? '{' : ',';
  body_ += json_string(name);
  body_ += ':';
  body_ += json;

  return *this;
}

std::string JsonObjectWriter::str() const {
  return body_.empty() ? "{}" : body_ + '}';
}

} // namespace carat::http
