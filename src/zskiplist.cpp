#include "zskiplist.h"

#include <cmath>
#include <cstdlib>

namespace tinyredis::zskiplist {

SkipList::SkipList() {
  header_ = new Node();
  header_->forward.resize(kMaxLevel, nullptr);
  header_->span.resize(kMaxLevel, 0);
}

SkipList::~SkipList() {
  Node* x = header_->forward[0];
  while (x) {
    Node* next = x->forward[0];
    delete x;
    x = next;
  }
  delete header_;
}

SkipList::SkipList(SkipList&& other) noexcept
    : header_(other.header_), level_(other.level_), length_(other.length_) {
  other.header_ = new Node();
  other.header_->forward.resize(kMaxLevel, nullptr);
  other.header_->span.resize(kMaxLevel, 0);
  other.level_ = 1;
  other.length_ = 0;
}

SkipList& SkipList::operator=(SkipList&& other) noexcept {
  if (this != &other) {
    this->~SkipList();
    new (this) SkipList(std::move(other));
  }
  return *this;
}

int SkipList::RandomLevel() {
  int level = 1;
  while (level < kMaxLevel &&
         ((double)::random() / (double)RAND_MAX) < kBranchProb) {
    ++level;
  }
  return level;
}

// Comparator: (score, member) ascending, exactly like Redis.
static inline bool KeyLess(double s1, std::string_view m1, double s2,
                           std::string_view m2) {
  if (s1 != s2) return s1 < s2;
  return m1 < m2;
}
static inline bool KeyEqual(double s1, std::string_view m1, double s2,
                            std::string_view m2) {
  return s1 == s2 && m1 == m2;
}

Node* SkipList::FindNode(double score, std::string_view member,
                          std::vector<Node*>* update) const {
  std::vector<Node*> upd(kMaxLevel, nullptr);
  std::vector<unsigned long> rank(kMaxLevel, 0);
  Node* x = header_;
  for (int i = level_ - 1; i >= 0; --i) {
    // Accumulate rank while descending.
    if (i < level_ - 1) rank[i] = rank[i + 1];
    while (x->forward[i] &&
           KeyLess(x->forward[i]->score, x->forward[i]->member, score,
                   member)) {
      rank[i] += x->span[i];
      x = x->forward[i];
    }
    upd[i] = x;
  }
  if (update) *update = std::move(upd);
  Node* cand = x->forward[0];
  if (cand && KeyEqual(cand->score, cand->member, score, member)) return cand;
  return nullptr;
}

bool SkipList::Contains(double score, std::string_view member) const {
  return FindNode(score, member, nullptr) != nullptr;
}

bool SkipList::Insert(double score, std::string member) {
  std::vector<Node*> update(kMaxLevel, nullptr);
  std::vector<unsigned long> rank(kMaxLevel, 0);
  Node* x = header_;
  for (int i = level_ - 1; i >= 0; --i) {
    if (i < level_ - 1) rank[i] = rank[i + 1];
    while (x->forward[i] &&
           KeyLess(x->forward[i]->score, x->forward[i]->member, score,
                   member)) {
      rank[i] += x->span[i];
      x = x->forward[i];
    }
    update[i] = x;
  }
  // We may have equal keys only if a node with the same (score, member)
  // exists — update its score in place (score is identical, but keep
  // semantics clear).
  Node* cand = x->forward[0];
  if (cand && KeyEqual(cand->score, cand->member, score, member)) {
    cand->score = score;  // same value; no structural change
    return false;
  }
  int level = RandomLevel();
  if (level > level_) {
    for (int i = level_; i < level; ++i) {
      rank[i] = 0;
      update[i] = header_;
      update[i]->span[i] = length_;
    }
    level_ = level;
  }
  Node* node = new Node();
  node->score = score;
  node->member = std::move(member);
  node->forward.resize(level, nullptr);
  node->span.resize(level, 0);
  for (int i = 0; i < level; ++i) {
    node->forward[i] = update[i]->forward[i];
    update[i]->forward[i] = node;
    // span update (matches zslInsert: rank[] is the per-level 1-based rank
    // of each update[] predecessor):
    node->span[i] = update[i]->span[i] - (rank[0] - rank[i]);
    update[i]->span[i] = (rank[0] - rank[i]) + 1;
  }
  // Increment spans for untouched higher levels.
  for (int i = level; i < level_; ++i) {
    update[i]->span[i]++;
  }
  ++length_;
  return true;
}

void SkipList::DeleteNode(Node* x, const std::vector<Node*>& update) {
  for (int i = 0; i < level_; ++i) {
    if (update[i]->forward[i] == x) {
      if (x->forward[i]) {
        update[i]->span[i] += x->span[i] - 1;
      } else {
        update[i]->span[i] = 0;
      }
      update[i]->forward[i] = x->forward[i];
    } else {
      update[i]->span[i] -= 1;
    }
  }
  while (level_ > 1 && header_->forward[level_ - 1] == nullptr) --level_;
  --length_;
  delete x;
}

bool SkipList::Erase(double score, std::string_view member) {
  std::vector<Node*> update;
  Node* x = FindNode(score, member, &update);
  if (!x) return false;
  DeleteNode(x, update);
  return true;
}

size_t SkipList::Rank(double score, std::string_view member) const {
  // Same shape as zslGetRank: walk *onto* the target node, accumulating
  // spans; the accumulated rank is the 1-based position.
  unsigned long rank = 0;
  Node* x = header_;
  for (int i = level_ - 1; i >= 0; --i) {
    while (x->forward[i] &&
           (x->forward[i]->score < score ||
            (x->forward[i]->score == score &&
             x->forward[i]->member <= member))) {
      rank += x->span[i];
      x = x->forward[i];
    }
    if (x != header_ && x->score == score && x->member == member) {
      return rank - 1;  // convert 1-based to 0-based
    }
  }
  return SIZE_MAX;
}

size_t SkipList::RevRank(double score, std::string_view member) const {
  size_t r = Rank(score, member);
  if (r == SIZE_MAX) return SIZE_MAX;
  if (r >= length_) return SIZE_MAX;
  return length_ - 1 - r;
}

std::vector<std::pair<double, std::string>> SkipList::RangeByRank(
    long long start, long long stop, bool rev) const {
  std::vector<std::pair<double, std::string>> out;
  if (length_ == 0) return out;
  long long llen = (long long)length_;
  if (start < 0) start = llen + start;
  if (stop < 0) stop = llen + stop;
  if (start < 0) start = 0;
  if (start > llen - 1) return out;
  if (stop < 0) return out;
  if (stop > llen - 1) stop = llen - 1;
  if (start > stop) return out;

  if (rev) {
    // Descending view: ranks [start, stop] map to ascending window
    // [llen-1-stop, llen-1-start], emitted in reverse order.
    long long asc_start = llen - 1 - stop;
    long long asc_stop = llen - 1 - start;
    start = asc_start;
    stop = asc_stop;
  }

  // Level-0 walk is simple and branch-free; rank windows are small in
  // practice and this keeps the logic trivially correct.
  out.reserve((size_t)(stop - start + 1));
  Node* x = header_->forward[0];
  long long r = 0;
  if (rev) {
    // Collect forward, then reverse.
    std::vector<std::pair<double, std::string>> tmp;
    tmp.reserve((size_t)(stop - start + 1));
    while (x && r <= stop) {
      if (r >= start)
        tmp.emplace_back(x->score, x->member);
      x = x->forward[0];
      ++r;
    }
    out.assign(tmp.rbegin(), tmp.rend());
  } else {
    while (x && r <= stop) {
      if (r >= start) out.emplace_back(x->score, x->member);
      x = x->forward[0];
      ++r;
    }
  }
  return out;
}

std::vector<std::pair<double, std::string>> SkipList::RangeByScore(
    const ScoreRange& r, bool rev, long long offset,
    long long count) const {
  std::vector<std::pair<double, std::string>> out;
  if (count == 0) return out;
  if (!rev) {
    // Position at the first node with score >= min.
    Node* x = header_;
    for (int i = level_ - 1; i >= 0; --i) {
      while (x->forward[i] && x->forward[i]->score < r.min) {
        x = x->forward[i];
      }
    }
    Node* cur = x->forward[0];
    // Handle min_exclusive: skip any ties at exactly r.min.
    if (cur && r.min_exclusive) {
      while (cur && cur->score == r.min) cur = cur->forward[0];
    }
    for (long long skipped = 0; cur; cur = cur->forward[0]) {
      double v = cur->score;
      if (r.min_exclusive ? v <= r.min : v < r.min) continue;
      if (r.max_exclusive ? v >= r.max : v > r.max) break;
      if (offset > 0 && skipped < offset) {
        ++skipped;
        continue;
      }
      out.emplace_back(v, cur->member);
      if (count > 0 && (long long)out.size() >= count) break;
    }
  } else {
    // Reverse traversal: collect ascending, then reverse the window.
    std::vector<std::pair<double, std::string>> asc;
    for (Node* cur = header_->forward[0]; cur; cur = cur->forward[0]) {
      double v = cur->score;
      if (r.min_exclusive ? v <= r.min : v < r.min) continue;
      if (r.max_exclusive ? v >= r.max : v > r.max) continue;
      asc.emplace_back(v, cur->member);
    }
    long long total = (long long)asc.size();
    long long end = total - offset;  // exclusive bound from the top
    if (end <= 0) return out;
    long long start_idx = count > 0 ? end - count : 0;
    if (start_idx < 0) start_idx = 0;
    // Emit [start_idx, end) in descending order.
    out.assign(asc.rbegin() + (asc.size() - (size_t)end),
               asc.rbegin() + (asc.size() - (size_t)start_idx));
  }
  return out;
}

size_t SkipList::CountInRange(const ScoreRange& r) const {
  size_t n = 0;
  for (Node* cur = header_->forward[0]; cur; cur = cur->forward[0]) {
    double v = cur->score;
    if (r.min_exclusive ? v <= r.min : v < r.min) continue;
    if (r.max_exclusive ? v >= r.max : v > r.max) break;
    ++n;
  }
  return n;
}

double SkipList::ScoreOf(std::string_view member) const {
  // Slow fallback; ZSet uses its dict for lookups.
  for (Node* cur = header_->forward[0]; cur; cur = cur->forward[0]) {
    if (cur->member == member) return cur->score;
  }
  return 0.0;
}

std::optional<std::pair<double, std::string>> SkipList::PopMin() {
  Node* first = header_->forward[0];
  if (!first) return std::nullopt;
  auto res = std::make_pair(first->score, first->member);
  std::vector<Node*> update;
  FindNode(first->score, first->member, &update);
  DeleteNode(first, update);
  return res;
}

std::optional<std::pair<double, std::string>> SkipList::PopMax() {
  // Find the last node (no backward pointers: descend greedily).
  Node* x = header_;
  for (int i = level_ - 1; i >= 0; --i) {
    while (x->forward[i]) x = x->forward[i];
  }
  if (x == header_) return std::nullopt;
  auto res = std::make_pair(x->score, x->member);
  std::vector<Node*> update;
  FindNode(x->score, x->member, &update);
  DeleteNode(x, update);
  return res;
}

}  // namespace tinyredis::zskiplist
