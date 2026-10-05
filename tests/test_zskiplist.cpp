// Skiplist tests: ordering, rank spans, ranges, pops.
#include <cstdio>
#include <string>
#include <vector>

#include "zskiplist.h"

using namespace tinyredis::zskiplist;

static int failures = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      ++failures;                                                \
      std::fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond, \
                   __LINE__);                                    \
    }                                                            \
  } while (0)

using Pair = std::pair<double, std::string>;

int main() {
  {
    SkipList sl;
    CHECK(sl.Empty());
    sl.Insert(5, "e");
    sl.Insert(1, "a");
    sl.Insert(3, "c");
    sl.Insert(2, "b");
    sl.Insert(4, "d");
    CHECK(sl.size() == 5);
    // The skiplist is keyed by (score, member): re-scoring an existing
    // member is done by the ZSet layer as Erase + Insert.
    CHECK(sl.Erase(5, "e"));
    CHECK(sl.Insert(9, "e") == true);
    CHECK(sl.size() == 5);
    CHECK(sl.Contains(9, "e"));
    CHECK(!sl.Contains(5, "e"));
    // Ascending order with ties broken by member.
    auto v = sl.RangeByRank(0, 4, false);
    CHECK(v.size() == 5);
    CHECK(v[0] == Pair(1, "a"));
    CHECK(v[1] == Pair(2, "b"));
    CHECK(v[2] == Pair(3, "c"));
    CHECK(v[3] == Pair(4, "d"));
    CHECK(v[4] == Pair(9, "e"));
    // Ranks.
    CHECK(sl.Rank(3, "c") == 2);
    CHECK(sl.Rank(9, "e") == 4);
    CHECK(sl.RevRank(1, "a") == 4);
    CHECK(sl.Rank(3, "zzz") == SIZE_MAX);
    // Negative-index windows.
    v = sl.RangeByRank(-2, -1, false);
    CHECK(v.size() == 2 && v[0] == Pair(4, "d") && v[1] == Pair(9, "e"));
    v = sl.RangeByRank(1, 2, true);
    // Descending view: rank 1..2 = (4,"d"), (3,"c") in that order.
    CHECK(v.size() == 2 && v[0] == Pair(4, "d") && v[1] == Pair(3, "c"));
    v = sl.RangeByRank(10, 20, false);
    CHECK(v.empty());
    // Score ties ordered by member.
    sl.Insert(3, "c2");
    v = sl.RangeByRank(0, 5, false);
    CHECK(v.size() == 6);
    CHECK(v[2] == Pair(3, "c"));
    CHECK(v[3] == Pair(3, "c2"));
    // Erase.
    CHECK(sl.Erase(3, "c"));
    CHECK(!sl.Erase(3, "c"));
    CHECK(sl.size() == 5);
    // Pops.
    auto mn = sl.PopMin();
    auto mx = sl.PopMax();
    CHECK(mn && mn->first == 1 && mn->second == "a");
    CHECK(mx && mx->first == 9 && mx->second == "e");
    CHECK(sl.size() == 3);
  }
  {
    SkipList sl;
    for (int i = 0; i < 100; ++i) {
      sl.Insert((i * 37) % 100, "m" + std::to_string(i));
    }
    CHECK(sl.size() == 100);
    // Full ascending sweep is sorted by (score, member).
    auto v = sl.RangeByRank(0, 99, false);
    CHECK(v.size() == 100);
    for (size_t i = 1; i < v.size(); ++i) {
      CHECK(v[i - 1].first < v[i].first ||
            (v[i - 1].first == v[i].first &&
             v[i - 1].second < v[i].second));
    }
    // Score range inclusive.
    sl = SkipList();
    sl.Insert(1, "one");
    sl.Insert(2, "two");
    sl.Insert(3, "three");
    sl.Insert(4, "four");
    ScoreRange r{2, 3, false, false};
    auto v2 = sl.RangeByScore(r, false, 0, -1);
    CHECK(v2.size() == 2 && v2[0].first == 2 && v2[1].first == 3);
    // Exclusive bounds.
    r.min_exclusive = true;
    r.max_exclusive = true;
    v2 = sl.RangeByScore(r, false, 0, -1);
    CHECK(v2.empty());
    // LIMIT offset/count.
    r = ScoreRange{1, 4, false, false};
    v2 = sl.RangeByScore(r, false, 1, 2);
    CHECK(v2.size() == 2 && v2[0].first == 2 && v2[1].first == 3);
    // Reverse.
    v2 = sl.RangeByScore(r, true, 0, -1);
    CHECK(v2.size() == 4 && v2[0].first == 4 && v2[3].first == 1);
    // Reverse with limit.
    v2 = sl.RangeByScore(r, true, 1, 2);
    CHECK(v2.size() == 2 && v2[0].first == 3 && v2[1].first == 2);
    // CountInRange.
    CHECK(sl.CountInRange(r) == 4);
    r.min_exclusive = true;
    CHECK(sl.CountInRange(r) == 3);
    // (min, +inf) full upper tail.
    r = ScoreRange{3, 1e300, true, false};
    CHECK(sl.CountInRange(r) == 1);
  }
  if (failures == 0) {
    std::printf("test_zskiplist: OK\n");
    return 0;
  }
  std::printf("test_zskiplist: %d FAILURES\n", failures);
  return 1;
}
