#pragma once

#include <concepts>
#include <optional>
#include <string>
#include <string_view>

namespace carat::http {

[[nodiscard]] std::string json_string(std::string_view value);

class JsonObjectWriter {
public:
  template <std::integral Integer>
    requires(!std::same_as<Integer, bool>)
  JsonObjectWriter &field(const std::string_view name, const Integer value) {
    return raw_field(name, std::to_string(value));
  }

  JsonObjectWriter &field(std::string_view name, bool value);
  JsonObjectWriter &field(std::string_view name, std::string_view value);
  JsonObjectWriter &field(std::string_view name, const char *value);

  template <typename Value>
  JsonObjectWriter &field(const std::string_view name, const std::optional<Value> &value) {
    return value ? field(name, *value) : *this;
  }

  JsonObjectWriter &raw_field(std::string_view name, std::string_view json);
  [[nodiscard]] std::string str() const;

private:
  std::string body_;
};

} // namespace carat::http
