// Database: 16 key spaces + TTL bookkeeping + memory accounting.
// Expiration follows the real Redis model:
//   - lazy expiry on access (lookup removes stale keys)
//   - active expire cycle (random sampling driven by the event loop)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "value.h"

namespace tinyredis {

struct Entry {
  Value value;
  long long expires_at_ms = 0;  // 0 = no expiry
};

using Table = std::unordered_map<std::string, Entry>;

class Database {
 public:
  static constexpr int kNumDatabases = 16;

  Database();
  ~Database();

  // Entry owns a move-only skiplist: copy is impossible, move is fine.
  Database(const Database&) = delete;
  Database& operator=(const Database&) = delete;
  Database(Database&&) = default;
  Database& operator=(Database&&) = default;

  // Lazy-expire aware lookup. Returns nullptr if the key is missing or
  // expired (and deletes the stale entry as a side effect).
  Entry* Lookup(int idx, const std::string& key);

  // Insert or replace. Handles memory accounting.
  void Set(int idx, std::string key, Value value,
           long long expires_at_ms = 0);
  // Remove (if present). Returns true if the key existed.
  bool Delete(int idx, const std::string& key);

  // Raw table access (for scans / INFO / RDB iteration).
  Table& table(int idx) { return dbs_[idx]; }
  const Table& table(int idx) const { return dbs_[idx]; }

  long long Size(int idx) const { return (long long)dbs_[idx].size(); }
  long long TotalKeys() const;
  long long TotalExpires() const;

  void FlushDb(int idx);
  void FlushAll();

  // Removes stale keys; samples a bounded window per db per call.
  // Returns the number of expired keys deleted.
  size_t ActiveExpireCycle();

  size_t used_memory() const { return used_; }
  size_t max_memory() const { return max_memory_; }
  void set_max_memory(size_t cap) { max_memory_ = cap; }
  size_t total_expired() const { return total_expired_; }

  // Approximate serialized size of a value (for INFO used_memory).
  static size_t ValueSize(const Value& v);

 private:
  static size_t EntrySize(const std::string& key, const Entry& e);
  void MemAdd(size_t n) { used_ += n; }
  void MemSub(size_t n) {
    if (used_ >= n) {
      used_ -= n;
    } else {
      used_ = 0;
    }
  }

  std::vector<Table> dbs_;
  size_t used_ = 0;
  size_t max_memory_ = 0;  // 0 = unlimited
  size_t total_expired_ = 0;
};

}  // namespace tinyredis
