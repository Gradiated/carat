#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace carat {

// Default stream formatting keeps six significant digits, which silently drops whole seconds from
// long-running counters. This writes the exact value with trailing fractional zeros removed.
inline void write_decimal(std::ostream &output, std::uint64_t scaled_value, int fraction_digits) {
  std::uint64_t scale = 1;
  for (int digit = 0; digit < fraction_digits; ++digit)
    scale *= 10;

  output << scaled_value / scale;

  const std::uint64_t fraction = scaled_value % scale;
  if (fraction == 0)
    return;

  std::string digits = std::to_string(fraction);
  digits.insert(0, static_cast<std::size_t>(fraction_digits) - digits.size(), '0');
  digits.erase(digits.find_last_not_of('0') + 1);
  output << '.' << digits;
}

} // namespace carat
