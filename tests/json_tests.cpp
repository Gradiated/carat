#include "carat/json.h"

#include "support/test_support.h"

#include <cstdint>
#include <string>

namespace {

using carat::Json;

ENGINE_TEST(json_parses_every_value_kind) {
  const auto json = Json::parse(
      R"( {"integer":42,"negative":-7,"float":1e-6,"unicode":"\u20ac","array":[true,false,null],)"
      R"("object":{"nested":"x"},"escapes":"\"\\\/\b\f\n\r\t"} )");

  CHECK(json.at("integer").as_u64() == 42);
  CHECK(json.at("negative").as_i64() == -7);
  CHECK(json.at("float").as_double() == 1e-6);
  CHECK(json.at("integer").as_double() == 42.0);
  CHECK(json.at("unicode").as_string() == "\xe2\x82\xac");
  CHECK(json.at("array").as_array()[0].as_bool());
  CHECK(!json.at("array").as_array()[1].as_bool());
  CHECK(json.at("array").as_array()[2].is_null());
  CHECK(json.at("object").at("nested").as_string() == "x");
  CHECK(json.at("escapes").as_string() == "\"\\/\b\f\n\r\t");
  CHECK(json.find("absent") == nullptr);
  CHECK(json.find("integer") != nullptr);
}

ENGINE_TEST(json_decodes_a_surrogate_pair_as_four_byte_utf8) {
  CHECK(Json::parse(R"("\ud83d\ude00")").as_string() == "\xf0\x9f\x98\x80");
}

ENGINE_TEST(json_rejects_structural_errors) {
  CHECK_THROWS(Json::parse("{} x"), "trailing input");
  CHECK_THROWS(Json::parse(R"({"a":1,"a":2})"), "duplicate object key");
  CHECK_THROWS(Json::parse(""), "expected value");
  CHECK_THROWS(Json::parse("[1,"), "expected value");
  CHECK_THROWS(Json::parse("[1"), "expected array end");
  CHECK_THROWS(Json::parse(R"({"a" 1})"), "expected colon");
  CHECK_THROWS(Json::parse(R"({"a":1)"), "expected object end");
  CHECK_THROWS(Json::parse("{1:2}"), "expected string");
  CHECK_THROWS(Json::parse("nul"), "invalid literal");
}

ENGINE_TEST(json_reports_the_failing_byte_offset) {
  CHECK_THROWS(Json::parse("[1,x]"), "JSON byte 3:");
}

ENGINE_TEST(json_rejects_invalid_strings) {
  CHECK_THROWS(Json::parse(std::string("\"a\nb\"")), "control character in string");
  CHECK_THROWS(Json::parse(R"("abc)"), "unterminated string");
  CHECK_THROWS(Json::parse(R"("\q")"), "invalid escape");
  CHECK_THROWS(Json::parse(R"("\)"), "short escape");
}

ENGINE_TEST(json_rejects_invalid_unicode_escapes) {
  CHECK_THROWS(Json::parse(R"("\u12g4")"), "invalid unicode escape");
  CHECK_THROWS(Json::parse(R"("\u12")"), "short unicode escape");
  CHECK_THROWS(Json::parse(R"("\ud83d")"), "missing low surrogate");
  CHECK_THROWS(Json::parse(R"("\ud83dx")"), "missing low surrogate");
  CHECK_THROWS(Json::parse(R"("\ud83d\u0041")"), "invalid low surrogate");
  CHECK_THROWS(Json::parse(R"("\ude00")"), "unexpected low surrogate");
}

ENGINE_TEST(json_rejects_malformed_numbers) {
  CHECK_THROWS(Json::parse("01"), "trailing input");
  CHECK_THROWS(Json::parse("1."), "missing fractional digits");
  CHECK_THROWS(Json::parse("1e"), "missing exponent digits");
  CHECK_THROWS(Json::parse("1e+"), "missing exponent digits");
  CHECK_THROWS(Json::parse("-"), "short number");
  CHECK_THROWS(Json::parse("-a"), "invalid number");
}

ENGINE_TEST(json_rejects_numbers_out_of_range) {
  CHECK(Json::parse("9223372036854775807").as_i64() == INT64_MAX);
  CHECK(Json::parse("-9223372036854775808").as_i64() == INT64_MIN);
  CHECK_THROWS(Json::parse("9223372036854775808"), "integer out of range");
  CHECK_THROWS(Json::parse("1e999"), "invalid floating-point number");
  CHECK_THROWS(Json::parse("-1e999"), "invalid floating-point number");
}

ENGINE_TEST(json_rounds_double_underflow_to_zero) {
  CHECK(Json::parse("1e-400").as_double() == 0.0);
}

ENGINE_TEST(json_accepts_nesting_up_to_the_depth_limit) {
  const std::string arrays = std::string(64, '[') + std::string(64, ']');
  std::string objects;
  for (int depth = 0; depth < 64; ++depth)
    objects += R"({"a":)";
  objects += "1" + std::string(64, '}');

  CHECK(Json::parse(arrays).as_array().size() == 1);
  CHECK(Json::parse(objects).at("a").as_object().size() == 1);
}

ENGINE_TEST(json_rejects_nesting_beyond_the_depth_limit) {
  CHECK_THROWS(Json::parse(std::string(65, '[') + std::string(65, ']')),
               "JSON byte 64: nesting too deep");
  CHECK_THROWS(Json::parse(std::string(64, '[') + "{}" + std::string(64, ']')), "nesting too deep");
}

ENGINE_TEST(json_rejects_a_one_mebibyte_nested_body_without_overflowing_the_stack) {
  CHECK_THROWS(Json::parse(std::string(std::size_t{1} << 20U, '[')), "nesting too deep");
}

ENGINE_TEST(json_accessors_reject_the_wrong_type) {
  const auto json = Json::parse(R"({"n":-1,"s":"x","f":1.5})");

  CHECK_THROWS(json.at("n").as_u64(), "JSON integer is negative");
  CHECK_THROWS(json.at("s").as_bool(), "JSON value is not a boolean");
  CHECK_THROWS(json.at("s").as_i64(), "JSON value is not an integer");
  CHECK_THROWS(json.at("f").as_i64(), "JSON value is not an integer");
  CHECK_THROWS(json.at("s").as_double(), "JSON value is not a number");
  CHECK_THROWS(json.at("n").as_string(), "JSON value is not a string");
  CHECK_THROWS(json.at("s").as_array(), "JSON value is not an array");
  CHECK_THROWS(json.at("s").as_object(), "JSON value is not an object");
  CHECK_THROWS(json.at("s").at("k"), "JSON value is not an object");
  CHECK_THROWS(json.at("missing"), "missing JSON key: missing");
}

} // namespace
