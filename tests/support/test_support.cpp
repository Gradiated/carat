#include "test_support.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace carat::testing {
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

} // namespace

TestRegistration::TestRegistration(const char *name, TestFunction function) {
  registered_tests().push_back({name, function});
}

void record_failure(const char *file, int line, const char *assertion) {
  ++failures;
  std::cerr << file << ':' << line << ": " << assertion << '\n';
}

} // namespace carat::testing

int main(int argc, char **argv) {
  namespace testing = carat::testing;
  const std::string_view only = argc > 1 ? argv[1] : "";

  for (const auto &test : testing::registered_tests()) {
    if (!only.empty() && only != test.name)
      continue;

    const int failures_before = testing::failures;
    const auto started = std::chrono::steady_clock::now();
    try {
      test.function();
    } catch (const testing::RequirementFailed &) {
    } catch (const std::exception &error) {
      ++testing::failures;
      std::cerr << test.name << ": unexpected exception: " << error.what() << '\n';
    }

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
    std::cout << (testing::failures == failures_before ? "PASS " : "FAIL ") << test.name << " ("
              << std::fixed << std::setprecision(2) << elapsed.count() << "s)\n";
  }

  return testing::failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
