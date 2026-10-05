// Small shared utilities.
#pragma once

#include <chrono>

namespace tinyredis {

inline long long NowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

inline long long NowUs() {
  using namespace std::chrono;
  return duration_cast<microseconds>(system_clock::now().time_since_epoch())
      .count();
}

}  // namespace tinyredis
