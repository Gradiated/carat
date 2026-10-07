#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace carat {

class Json {
public:
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json, std::less<>>;
  using Value =
      std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>;

  explicit Json(Value value) : value_(std::move(value)) {}

  static Json parse(std::string_view input);

  [[nodiscard]] bool is_null() const;
  [[nodiscard]] bool as_bool() const;
  [[nodiscard]] std::int64_t as_i64() const;
  [[nodiscard]] std::uint64_t as_u64() const;
  [[nodiscard]] double as_double() const;
  [[nodiscard]] const std::string &as_string() const;
  [[nodiscard]] const Array &as_array() const;
  [[nodiscard]] const Object &as_object() const;
  [[nodiscard]] const Json &at(std::string_view key) const;
  [[nodiscard]] const Json *find(std::string_view key) const;

private:
  Value value_;
};

std::string read_text_file(const std::string &path);

} // namespace carat
