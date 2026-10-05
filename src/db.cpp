#include "db.h"

#include <random>

#include "utils.h"

namespace tinyredis {

Database::Database() : dbs_(kNumDatabases) {}

Database::~Database() = default;

size_t Database::ValueSize(const Value& v) {
  size_t n = 0;
  switch (v.index()) {
    case 0: {  // RawString
      n = 16 + std::get<RawString>(v).s.size();
      break;
    }
    case 1: {  // HashValue
      const auto& h = std::get<HashValue>(v).h;
      n = 32;
      for (const auto& kv : h) n += 16 + kv.first.size() + kv.second.size();
      break;
    }
    case 2: {  // ListValue
      const auto& l = std::get<ListValue>(v).l;
      n = 16;
      for (const auto& s : l) n += 8 + s.size();
      break;
    }
    case 3: {  // SetValue
      const auto& s = std::get<SetValue>(v).s;
      n = 16;
      for (const auto& e : s) n += 16 + e.size();
      break;
    }
    case 4: {  // ZSetValue
      const auto& z = std::get<ZSetValue>(v);
      n = 32;
      for (const auto& kv : z.dict) n += 16 + kv.first.size();
      break;
    }
    default:
      n = 16;
  }
  return n;
}

size_t Database::EntrySize(const std::string& key, const Entry& e) {
  return 48 + key.size() + ValueSize(e.value);
}

Entry* Database::Lookup(int idx, const std::string& key) {
  Table& t = dbs_[idx];
  auto it = t.find(key);
  if (it == t.end()) return nullptr;
  Entry& e = it->second;
  if (e.expires_at_ms != 0 && NowMs() >= e.expires_at_ms) {
    // Lazy expiry: drop the stale entry.
    MemSub(EntrySize(key, e));
    t.erase(it);
    ++total_expired_;
    return nullptr;
  }
  return &e;
}

void Database::Set(int idx, std::string key, Value value,
                   long long expires_at_ms) {
  Table& t = dbs_[idx];
  auto [it, inserted] = t.emplace(std::move(key), Entry{});
  if (inserted) {
    it->second.value = std::move(value);
    it->second.expires_at_ms = expires_at_ms;
    MemAdd(EntrySize(it->first, it->second));
  } else {
    MemSub(EntrySize(it->first, it->second));
    it->second.value = std::move(value);
    it->second.expires_at_ms = expires_at_ms;
    MemAdd(EntrySize(it->first, it->second));
  }
}

bool Database::Delete(int idx, const std::string& key) {
  Table& t = dbs_[idx];
  auto it = t.find(key);
  if (it == t.end()) return false;
  MemSub(EntrySize(key, it->second));
  t.erase(it);
  return true;
}

long long Database::TotalKeys() const {
  long long n = 0;
  for (int i = 0; i < kNumDatabases; ++i) n += (long long)dbs_[i].size();
  return n;
}

long long Database::TotalExpires() const {
  long long n = 0;
  for (int i = 0; i < kNumDatabases; ++i) {
    for (const auto& kv : dbs_[i]) {
      if (kv.second.expires_at_ms != 0) ++n;
    }
  }
  return n;
}

void Database::FlushDb(int idx) {
  Table& t = dbs_[idx];
  for (const auto& kv : t) MemSub(EntrySize(kv.first, kv.second));
  t.clear();
}

void Database::FlushAll() {
  for (int i = 0; i < kNumDatabases; ++i) FlushDb(i);
}

size_t Database::ActiveExpireCycle() {
  static thread_local std::mt19937 rng(NowUs() & 0x7fffffff);
  size_t deleted = 0;
  constexpr size_t kSampleWindow = 200;
  for (int d = 0; d < kNumDatabases; ++d) {
    Table& t = dbs_[d];
    if (t.empty()) continue;
    size_t size = t.size();
    // Start from a pseudo-random offset (bounded, so the walk is cheap).
    size_t offset = rng() % (size < kSampleWindow ? size : kSampleWindow);
    auto it = t.begin();
    std::advance(it, offset);
    size_t scanned = 0;
    long long now = NowMs();
    while (it != t.end() && scanned < kSampleWindow) {
      ++scanned;
      if (it->second.expires_at_ms != 0 && now >= it->second.expires_at_ms) {
        MemSub(EntrySize(it->first, it->second));
        it = t.erase(it);
        ++deleted;
        ++total_expired_;
      } else {
        ++it;
      }
    }
  }
  return deleted;
}

}  // namespace tinyredis
