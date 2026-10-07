#pragma once

#include <exception>
#include <string_view>

namespace carat::testing {

using TestFunction = void (*)();

struct TestRegistration {
  TestRegistration(const char *name, TestFunction function);
};

struct RequirementFailed {};

void record_failure(const char *file, int line, const char *assertion);

template <typename Function>
[[nodiscard]] bool throws_containing(Function &&function, std::string_view fragment) {
  try {
    function();
  } catch (const std::exception &error) {
    return std::string_view(error.what()).find(fragment) != std::string_view::npos;
  }
  return false;
}

} // namespace carat::testing

#define CHECK(condition)                                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      ::carat::testing::record_failure(__FILE__, __LINE__, "CHECK(" #condition ")");               \
    }                                                                                              \
  } while (false)

#define REQUIRE(condition)                                                                         \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      ::carat::testing::record_failure(__FILE__, __LINE__, "REQUIRE(" #condition ")");             \
      throw ::carat::testing::RequirementFailed{};                                                 \
    }                                                                                              \
  } while (false)

#define CHECK_THROWS(expression, fragment)                                                         \
  CHECK(::carat::testing::throws_containing([&] { static_cast<void>(expression); }, fragment))

#define ENGINE_TEST(name)                                                                          \
  static void name();                                                                              \
  static const ::carat::testing::TestRegistration name##_registration{#name, name};                \
  static void name()
