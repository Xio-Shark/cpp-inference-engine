// Hand-written skiplist ordered by (score, member), the sorted-set backbone.
// Levels carry spans so rank lookups are O(log n), mirroring the original
// design of Redis' t_zset.c (ZSKIPLIST_MAXLEVEL=32, p=0.25).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tinyredis::zskiplist {

constexpr int kMaxLevel = 32;
constexpr double kBranchProb = 0.25;

struct Node {
  double score = 0.0;
  std::string member;
  // forward[i] -> next node at level i; span[i] -> # of nodes skipped at
  // level i (used for O(log n) rank queries).
  std::vector<Node*> forward;
  std::vector<unsigned long> span;
};

struct ScoreRange {
  double min = 0.0, max = 0.0;
  bool min_exclusive = false, max_exclusive = false;
};

class SkipList {
 public:
  SkipList();
  ~SkipList();
  SkipList(const SkipList&) = delete;
  SkipList& operator=(const SkipList&) = delete;
  SkipList(SkipList&& other) noexcept;
  SkipList& operator=(SkipList&& other) noexcept;

  size_t size() const { return length_; }

  // Insert or update (score, member). Returns true if a new node was created,
  // false if an existing member's score was updated.
  bool Insert(double score, std::string member);
  // Remove by (score, member). Returns true if a node was removed.
  bool Erase(double score, std::string_view member);
  // Lookup by (score, member) — O(log n).
  bool Contains(double score, std::string_view member) const;

  // 0-based rank of (score, member) from the smallest; SIZE_MAX if absent.
  size_t Rank(double score, std::string_view member) const;
  // 0-based rank from the largest.
  size_t RevRank(double score, std::string_view member) const;

  // Inclusive [start, stop], 0-based, clamped to [0, size-1]. rev=true yields
  // descending order. Empty if the window is out of range.
  std::vector<std::pair<double, std::string>> RangeByRank(long long start,
                                                          long long stop,
                                                          bool rev) const;
  // Score-range query with optional exclusivity, reverse order, and LIMIT
  // offset/count semantics (count<0 means "no limit").
  std::vector<std::pair<double, std::string>> RangeByScore(
      const ScoreRange& r, bool rev, long long offset,
      long long count) const;
  // Number of members within the score range.
  size_t CountInRange(const ScoreRange& r) const;
  // Nodes-per-score-range used by ZRANGEBYSCORE counts; exact via traversal.
  double ScoreOf(std::string_view member) const;  // 0 if absent (callers use dict)
  std::optional<std::pair<double, std::string>> PopMin();
  std::optional<std::pair<double, std::string>> PopMax();

  // First node's score (0 if empty); used by ZCOUNT fast paths.
  bool Empty() const { return length_ == 0; }

  Node* head() const { return header_; }
  int level() const { return level_; }

 private:
  int RandomLevel();
  Node* FindNode(double score, std::string_view member,
                 std::vector<Node*>* update) const;  // fills update if !=null
  void DeleteNode(Node* x, const std::vector<Node*>& update);

  Node* header_ = nullptr;
  int level_ = 1;
  size_t length_ = 0;
};

}  // namespace tinyredis::zskiplist
