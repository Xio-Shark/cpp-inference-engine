#include "commands.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "resp.h"
#include "rdb.h"
#include "utils.h"
#include "value.h"

namespace tinyredis {

namespace {

using Args = const std::vector<std::string>&;
using Out = std::string&;

std::string ToLower(const std::string& s) {
  std::string r(s);
  for (auto& c : r)
    if (c >= 'A' && c <= 'Z') c += 32;
  return r;
}
std::string ToUpper(const std::string& s) {
  std::string r(s);
  for (auto& c : r)
    if (c >= 'a' && c <= 'z') c -= 32;
  return r;
}

// ---- error helpers (message text matches real Redis) ----
void ErrWrongArgs(Out out, const std::string& cmd_lower) {
  resp::AppendErrorReply(out, "ERR wrong number of arguments for '" +
                                  cmd_lower + "' command");
}
void ErrWrongType(Out out) {
  resp::AppendErrorReply(
      out, "WRONGTYPE Operation against a key holding the wrong kind of value");
}
void ErrSyntax(Out out) { resp::AppendErrorReply(out, "ERR syntax error"); }
void ErrNotInt(Out out) {
  resp::AppendErrorReply(out, "ERR value is not an integer or out of range");
}
void ErrNotFloat(Out out) {
  resp::AppendErrorReply(out, "ERR value is not a valid float");
}
void ErrHashNotInt(Out out) {
  resp::AppendErrorReply(out, "ERR hash value is not an integer");
}
void ErrIncrOverflow(Out out) {
  resp::AppendErrorReply(out, "ERR increment would produce NaN or Infinity");
}
void ErrHashNotFloat(Out out) {
  resp::AppendErrorReply(out, "ERR hash value is not a float");
}
void ErrUnknownCommand(Out out, const std::vector<std::string>& argv) {
  std::string msg = "ERR unknown command '" + argv[0] + "'";
  if (argv.size() > 1) {
    msg += ", with args beginning with: '" + argv[1] + "' ";
  }
  resp::AppendErrorReply(out, msg);
}

bool ParseIntStrict(const std::string& s, long long* out) {
  if (s.empty()) return false;
  errno = 0;
  char* end = nullptr;
  long long v = strtoll(s.c_str(), &end, 10);
  if (errno == ERANGE || end != s.c_str() + s.size()) return false;
  *out = v;
  return true;
}
bool ParseDoubleStrict(const std::string& s, double* out) {
  if (s.empty()) return false;
  errno = 0;
  char* end = nullptr;
  double v = strtod(s.c_str(), &end);
  if (end != s.c_str() + s.size()) return false;
  *out = v;
  return true;
}
bool ParseLongDoubleStrict(const std::string& s, long double* out) {
  if (s.empty()) return false;
  errno = 0;
  char* end = nullptr;
  long double v = strtold(s.c_str(), &end);
  if (end != s.c_str() + s.size()) return false;
  *out = v;
  return true;
}

// ---- typed lookup ----
enum class LookupResult { kMissing, kFound, kWrongType };

template <typename T>
LookupResult GetTyped(Database& db, int idx, const std::string& key, T** out) {
  Entry* e = db.Lookup(idx, key);
  if (!e) return LookupResult::kMissing;
  T* p = std::get_if<T>(&e->value);
  if (!p) return LookupResult::kWrongType;
  *out = p;
  return LookupResult::kFound;
}

// Sets or creates a string entry; returns the entry.
Entry& GetOrCreateStringEntry(Database& db, int idx, const std::string& key) {
  Entry* e = db.Lookup(idx, key);
  if (!e) {
    db.Set(idx, key, RawString{});
    e = db.Lookup(idx, key);
  }
  return *e;
}

bool KeyIsLiveExpired(const Entry& e) {
  return e.expires_at_ms != 0 && NowMs() >= e.expires_at_ms;
}

// ---- arity table ----
struct Arity {
  int min, max;  // max < 0 means unbounded
};

const std::unordered_map<std::string, Arity>& ArityTable() {
  static const std::unordered_map<std::string, Arity> table = {
      {"PING", {1, 2}},        {"ECHO", {2, 2}},
      {"SELECT", {2, 2}},      {"AUTH", {2, 2}},
      {"QUIT", {1, 1}},        {"HELLO", {1, -1}},
      {"COMMAND", {1, 2}},     {"INFO", {1, 2}},
      {"DBSIZE", {1, 1}},      {"FLUSHDB", {1, 2}},
      {"FLUSHALL", {1, 2}},    {"SAVE", {1, 1}},
      {"BGSAVE", {1, 1}},      {"LASTSAVE", {1, 1}},
      {"BGREWRITEAOF", {1, 1}}, {"WAIT", {3, 3}},
      {"TIME", {1, 1}},        {"ROLE", {1, 1}},
      {"SHUTDOWN", {1, 2}},    {"CONFIG", {3, 3}},
      {"CLIENT", {2, -1}},     {"MEMORY", {3, 3}},
      {"GET", {2, 2}},         {"SET", {3, -1}},
      {"SETNX", {3, 3}},       {"SETEX", {4, 4}},
      {"PSETEX", {4, 4}},       {"GETSET", {3, 3}},
      {"GETDEL", {2, 2}},      {"GETRANGE", {4, 4}},
      {"SETRANGE", {4, 4}},    {"STRLEN", {2, 2}},
      {"APPEND", {3, 3}},      {"INCR", {2, 2}},
      {"DECR", {2, 2}},        {"INCRBY", {3, 3}},
      {"DECRBY", {3, 3}},      {"INCRBYFLOAT", {3, 3}},
      {"MGET", {2, -1}},       {"MSET", {3, -1}},
      {"MSETNX", {3, -1}},     {"DEL", {2, -1}},
      {"UNLINK", {2, -1}},     {"EXISTS", {2, -1}},
      {"KEYS", {2, 2}},        {"RANDOMKEY", {1, 1}},
      {"TYPE", {2, 2}},        {"RENAME", {3, 3}},
      {"EXPIRE", {3, 3}},      {"PEXPIRE", {3, 3}},
      {"EXPIREAT", {3, 3}},    {"PEXPIREAT", {3, 3}},
      {"TTL", {2, 2}},         {"PTTL", {2, 2}},
      {"PERSIST", {2, 2}},     {"SCAN", {2, -1}},
      {"HSET", {4, -1}},       {"HMSET", {4, -1}},
      {"HGET", {3, 3}},        {"HMGET", {3, -1}},
      {"HGETALL", {2, 2}},     {"HDEL", {3, -1}},
      {"HEXISTS", {3, 3}},     {"HLEN", {2, 2}},
      {"HKEYS", {2, 2}},       {"HVALS", {2, 2}},
      {"HSETNX", {4, 4}},      {"HINCRBY", {4, 4}},
      {"HINCRBYFLOAT", {4, 4}}, {"HSTRLEN", {3, 3}},
      {"LPUSH", {3, -1}},      {"RPUSH", {3, -1}},
      {"LPOP", {2, 3}},        {"RPOP", {2, 3}},
      {"LLEN", {2, 2}},        {"LRANGE", {4, 4}},
      {"LINDEX", {3, 3}},      {"LSET", {4, 4}},
      {"LREM", {4, 4}},        {"LINSERT", {5, 5}},
      {"LTRIM", {4, 4}},       {"LMOVE", {5, 5}},
      {"RPOPLPUSH", {3, 3}},   {"SADD", {3, -1}},
      {"SREM", {3, -1}},       {"SMEMBERS", {2, 2}},
      {"SISMEMBER", {3, 3}},   {"SCARD", {2, 2}},
      {"SPOP", {2, 3}},        {"SRANDMEMBER", {2, 3}},
      {"SMOVE", {4, 4}},       {"SINTER", {2, -1}},
      {"SUNION", {2, -1}},     {"SDIFF", {2, -1}},
      {"SINTERSTORE", {3, -1}}, {"SUNIONSTORE", {3, -1}},
      {"SDIFFSTORE", {3, -1}}, {"ZADD", {4, -1}},
      {"ZSCORE", {3, 3}},      {"ZCARD", {2, 2}},
      {"ZRANGE", {3, -1}},     {"ZREVRANGE", {4, -1}},
      {"ZRANGEBYSCORE", {4, -1}}, {"ZREVRANGEBYSCORE", {4, -1}},
      {"ZRANK", {3, 3}},       {"ZREVRANK", {3, 3}},
      {"ZCOUNT", {4, 4}},      {"ZREM", {3, -1}},
      {"ZINCRBY", {4, 4}},     {"ZPOPMIN", {2, 3}},
      {"ZPOPMAX", {2, 3}},     {"ZREMRANGEBYRANK", {4, 4}},
      {"ZREMRANGEBYSCORE", {4, 4}}, {"MULTI", {1, 1}},
      {"EXEC", {1, 1}},        {"DISCARD", {1, 1}},
  };
  return table;
}

bool CommandExists(const std::string& upper) {
  return ArityTable().count(upper) > 0;
}
bool ArityOk(const std::string& upper, size_t argc) {
  auto it = ArityTable().find(upper);
  if (it == ArityTable().end()) return false;
  if ((int)argc < it->second.min) return false;
  if (it->second.max >= 0 && (int)argc > it->second.max) return false;
  return true;
}

bool IsWriteCommand(const std::string& upper) {
  static const std::unordered_set<std::string> writes = {
      "SET",    "SETNX",  "SETEX",     "PSETEX", "GETSET",  "SETRANGE",
      "APPEND", "INCR",   "DECR",      "INCRBY", "DECRBY",  "INCRBYFLOAT",
      "MSET",   "MSETNX", "DEL",       "UNLINK", "RENAME",  "EXPIRE",
      "PEXPIRE", "EXPIREAT", "PEXPIREAT", "PERSIST", "GETDEL",
      "HSET",   "HMSET",  "HDEL",      "HSETNX", "HINCRBY", "HINCRBYFLOAT",
      "LPUSH",  "RPUSH",  "LPOP",      "RPOP",   "LSET",   "LREM",
      "LINSERT", "LTRIM", "LMOVE",     "RPOPLPUSH",
      "SADD",   "SREM",   "SPOP",      "SMOVE",  "SINTERSTORE",
      "SUNIONSTORE", "SDIFFSTORE",
      "ZADD",   "ZREM",   "ZINCRBY",   "ZPOPMIN", "ZPOPMAX",
      "ZREMRANGEBYRANK", "ZREMRANGEBYSCORE",
      "FLUSHDB", "FLUSHALL",
  };
  return writes.count(upper) > 0;
}

// ---------------- score bounds (ZSET ranges) ----------------
struct ScoreBound {
  double v = 0.0;
  bool exclusive = false;
  bool ok = false;
};

ScoreBound ParseScoreBound(const std::string& s) {
  ScoreBound b;
  std::string t = s;
  if (!t.empty() && t[0] == '(') {
    b.exclusive = true;
    t = t.substr(1);
  }
  if (t == "-inf" || t == "-infinity") {
    b.v = -HUGE_VAL;
    b.ok = true;
    return b;
  }
  if (t == "+inf" || t == "inf" || t == "infinity") {
    b.v = HUGE_VAL;
    b.ok = true;
    return b;
  }
  double d = 0.0;
  if (!ParseDoubleStrict(t, &d)) return b;
  b.v = d;
  b.ok = true;
  return b;
}

zskiplist::ScoreRange ToRange(const ScoreBound& min, const ScoreBound& max) {
  zskiplist::ScoreRange r;
  r.min = min.v;
  r.max = max.v;
  r.min_exclusive = min.exclusive;
  r.max_exclusive = max.exclusive;
  return r;
}

std::string FormatDouble(double d) {
  char buf[64];
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  if (d == 0.0) return "0";
  // Shortest representation that round-trips (like Redis' fpconv dtoa).
  for (int prec = 15; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*g", prec, d);
    if (std::strtod(buf, nullptr) == d) return buf;
  }
  return buf;
}

std::string FormatLongDouble(long double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  if (d == 0.0) return "0";
  char buf[128];
  int l = std::snprintf(buf, sizeof(buf), "%.17Lf", d);
  if (l <= 0) return "0";
  if (std::strchr(buf, '.') != nullptr) {
    char* p = buf + l - 1;
    while (*p == '0') {
      *p = '\0';
      --p;
      --l;
    }
    if (*p == '.') {
      *p = '\0';
      --l;
    }
  }
  if (l == 2 && buf[0] == '-' && buf[1] == '0') {
    return "0";
  }
  return buf;
}

void AppendScoredPairs(Out out,
                       const std::vector<std::pair<double, std::string>>& v,
                       bool with_scores) {
  if (v.empty()) {
    resp::AppendEmptyArrayReply(out);
    return;
  }
  if (!with_scores) {
    resp::AppendArrayReply(out, (long long)v.size());
    for (const auto& p : v) resp::AppendBulkReply(out, p.second);
    return;
  }
  resp::AppendArrayReply(out, (long long)(v.size() * 2));
  for (const auto& p : v) {
    resp::AppendBulkReply(out, p.second);
    resp::AppendBulkReply(out, FormatDouble(p.first));
  }
}

// ---------------- string commands ----------------
void CmdSet(Database& db, ClientState& cs, Args argv, Out out) {
  const std::string& key = argv[1];
  const std::string& val = argv[2];
  long long ex_s = -1, px_ms = -1;
  bool nx = false, xx = false, keepttl = false, want_get = false;
  for (size_t i = 3; i < argv.size(); ++i) {
    std::string o = ToUpper(argv[i]);
    if (o == "NX") {
      nx = true;
    } else if (o == "XX") {
      xx = true;
    } else if (o == "KEEPTTL") {
      keepttl = true;
    } else if (o == "GET") {
      want_get = true;
    } else if (o == "EX" && i + 1 < argv.size()) {
      if (!ParseIntStrict(argv[++i], &ex_s) || ex_s <= 0) {
        ErrSyntax(out);
        return;
      }
    } else if (o == "PX" && i + 1 < argv.size()) {
      if (!ParseIntStrict(argv[++i], &px_ms) || px_ms <= 0) {
        ErrSyntax(out);
        return;
      }
    } else {
      ErrSyntax(out);
      return;
    }
  }
  if (nx && xx) {
    ErrSyntax(out);
    return;
  }
  Entry* e = db.Lookup(cs.db_index, key);
  if (want_get) {
    // Reply with the old value first (or error on wrong type).
    if (!e) {
      resp::AppendNullBulkReply(out);
    } else if (auto* rs = std::get_if<RawString>(&e->value)) {
      resp::AppendBulkReply(out, rs->s);
    } else {
      ErrWrongType(out);
      return;
    }
  }
  if (e && nx) {  // NX: key exists -> no-op
    if (!want_get) resp::AppendNullBulkReply(out);
    return;
  }
  if (!e && xx) {  // XX: key missing -> no-op
    if (!want_get) resp::AppendNullBulkReply(out);
    return;
  }
  long long keep_ttl = 0;
  if (e && keepttl) keep_ttl = e->expires_at_ms;
  long long expires = keep_ttl;
  if (ex_s > 0) expires = NowMs() + ex_s * 1000;
  if (px_ms > 0) expires = NowMs() + px_ms;
  db.Set(cs.db_index, key, RawString{val}, expires);
  if (!want_get) resp::AppendSimpleReply(out, "OK");
}

void CmdIncrBy(Database& db, ClientState& cs, Args argv, Out out,
               long long by, bool is_float) {
  const std::string& key = argv[1];
  Entry& e = GetOrCreateStringEntry(db, cs.db_index, key);
  auto* rs = std::get_if<RawString>(&e.value);
  if (!rs) {
    ErrWrongType(out);
    return;
  }
  if (is_float) {
    long double cur = 0.0, delta = 0.0;
    if (!rs->s.empty() && !ParseLongDoubleStrict(rs->s, &cur)) {
      ErrNotFloat(out);
      return;
    }
    if (rs->s.empty()) cur = 0.0;
    if (!ParseLongDoubleStrict(argv[2], &delta)) {
      ErrNotFloat(out);
      return;
    }
    long double res = cur + delta;
    if (std::isnan(res) || std::isinf(res)) {
      ErrIncrOverflow(out);
      return;
    }
    rs->s = FormatLongDouble(res);
    resp::AppendBulkReply(out, rs->s);
    return;
  }
  long long cur = 0;
  if (!rs->s.empty() && !ParseIntStrict(rs->s, &cur)) {
    ErrNotInt(out);
    return;
  }
  long long res;
  if (__builtin_add_overflow(cur, by, &res)) {
    ErrNotInt(out);
    return;
  }
  rs->s = std::to_string(res);
  resp::AppendIntReply(out, res);
}

// ---------------- hash commands ----------------
HashValue* HashOrCreate(Database& db, int idx, const std::string& key) {
  Entry* e = db.Lookup(idx, key);
  if (!e) {
    db.Set(idx, key, HashValue{});
    e = db.Lookup(idx, key);
  } else if (!std::get_if<HashValue>(&e->value)) {
    return nullptr;
  }
  return std::get_if<HashValue>(&e->value);
}

// ---------------- list commands ----------------
ListValue* ListCreate(Database& db, int idx, const std::string& key) {
  Entry* e = db.Lookup(idx, key);
  if (!e) {
    db.Set(idx, key, ListValue{});
    e = db.Lookup(idx, key);
  }
  return std::get_if<ListValue>(&e->value);
}

void CmdPush(Database& db, ClientState& cs, Args argv, Out out, bool left) {
  const std::string& key = argv[1];
  Entry* e = db.Lookup(cs.db_index, key);
  if (!e) {
    db.Set(cs.db_index, key, ListValue{});
    e = db.Lookup(cs.db_index, key);
  }
  auto* l = std::get_if<ListValue>(&e->value);
  if (!l) {
    ErrWrongType(out);
    return;
  }
  for (size_t i = 2; i < argv.size(); ++i) {
    if (left)
      l->l.push_front(argv[i]);
    else
      l->l.push_back(argv[i]);
  }
  resp::AppendIntReply(out, (long long)l->l.size());
}

void CmdPop(Database& db, ClientState& cs, Args argv, Out out, bool left) {
  const std::string& key = argv[1];
  long long count = -1;  // -1 = no count arg (single-pop semantics)
  if (argv.size() == 3) {
    if (!ParseIntStrict(argv[2], &count) || count < 0) {
      ErrNotInt(out);
      return;
    }
  }
  auto* l = (ListValue*)nullptr;
  auto st = GetTyped<ListValue>(db, cs.db_index, key, &l);
  if (st == LookupResult::kWrongType) {
    ErrWrongType(out);
    return;
  }
  if (st == LookupResult::kMissing || l->l.empty()) {
    resp::AppendNullBulkReply(out);
    return;
  }
  if (count < 0) {
    std::string v = left ? l->l.front() : l->l.back();
    if (left)
      l->l.pop_front();
    else
      l->l.pop_back();
    resp::AppendBulkReply(out, v);
    return;
  }
  long long n = std::min<long long>(count, (long long)l->l.size());
  std::vector<std::string> items;
  for (long long i = 0; i < n; ++i) {
    items.push_back(left ? l->l.front() : l->l.back());
    if (left)
      l->l.pop_front();
    else
      l->l.pop_back();
  }
  if (n == 0) {
    resp::AppendEmptyArrayReply(out);
    return;
  }
  resp::AppendBulkArray(out, items);
}

bool ListIndexRange(long long* start, long long* stop, long long llen) {
  if (*start < 0) *start += llen;
  if (*stop < 0) *stop += llen;
  if (*start < 0) *start = 0;
  if (*stop >= llen) *stop = llen - 1;
  if (*stop < *start) return false;  // empty window
  return true;
}

void CmdLMove(Database& db, ClientState& cs, Args argv, Out out) {
  const std::string& src = argv[1];
  const std::string& dst = argv[2];
  bool from_left = ToUpper(argv[3]) == "LEFT";
  bool to_left = ToUpper(argv[4]) == "LEFT";
  if (ToUpper(argv[3]) != "LEFT" && ToUpper(argv[3]) != "RIGHT") {
    ErrSyntax(out);
    return;
  }
  if (ToUpper(argv[4]) != "LEFT" && ToUpper(argv[4]) != "RIGHT") {
    ErrSyntax(out);
    return;
  }
  auto* sl = (ListValue*)nullptr;
  auto sst = GetTyped<ListValue>(db, cs.db_index, src, &sl);
  if (sst == LookupResult::kWrongType) {
    ErrWrongType(out);
    return;
  }
  auto* dl = (ListValue*)nullptr;
  auto dst_st = GetTyped<ListValue>(db, cs.db_index, dst, &dl);
  if (dst_st == LookupResult::kWrongType) {
    ErrWrongType(out);
    return;
  }
  if (sst == LookupResult::kMissing || sl->l.empty()) {
    resp::AppendNullBulkReply(out);
    return;
  }
  if (dst_st == LookupResult::kMissing) {
    dl = ListCreate(db, cs.db_index, dst);
    if (!dl) {
      ErrWrongType(out);
      return;
    }
  }
  std::string v;
  if (src == dst) {
    auto& l = sl->l;
    if (from_left && to_left) {  // rotate: no-op move
      v = l.front();
    } else if (!from_left && !to_left) {
      v = l.back();
    } else if (from_left) {  // LEFT -> RIGHT
      v = l.front();
      l.splice(l.end(), l, l.begin());
    } else {  // RIGHT -> LEFT
      v = l.back();
      l.splice(l.begin(), l, std::prev(l.end()));
    }
  } else {
    v = from_left ? sl->l.front() : sl->l.back();
    if (from_left)
      sl->l.pop_front();
    else
      sl->l.pop_back();
    if (to_left)
      dl->l.push_front(v);
    else
      dl->l.push_back(v);
  }
  resp::AppendBulkReply(out, v);
}

// ---------------- set commands ----------------
SetValue* SetCreate(Database& db, int idx, const std::string& key) {
  Entry* e = db.Lookup(idx, key);
  if (!e) {
    db.Set(idx, key, SetValue{});
    e = db.Lookup(idx, key);
  }
  return std::get_if<SetValue>(&e->value);
}

std::vector<std::string> SetMembersSorted(const SetValue& s) {
  std::vector<std::string> v(s.s.begin(), s.s.end());
  std::sort(v.begin(), v.end());
  return v;
}

enum class SetOp { kUnion, kInter, kDiff };

SetValue EvalSetOp(Database& db, int idx, Args argv, size_t skip, Out out,
                   bool* ok, SetOp op) {
  SetValue result;
  *ok = true;
  const SetValue* base = nullptr;
  std::vector<const SetValue*> others;
  for (size_t i = skip; i < argv.size(); ++i) {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, idx, argv[i], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      *ok = false;
      return result;
    }
    if (st == LookupResult::kMissing) {
      if (op == SetOp::kInter) return result;  // empty intersection
      continue;
    }
    if (!base) {
      base = sp;
    } else {
      others.push_back(sp);
    }
  }
  if (!base) return result;
  if (op == SetOp::kUnion) {
    for (const auto& m : base->s) result.s.insert(m);
    for (const auto* o : others)
      for (const auto& m : o->s) result.s.insert(m);
  } else if (op == SetOp::kInter) {
    for (const auto& m : base->s) {
      bool in_all = true;
      for (const auto* o : others)
        if (!o->s.count(m)) {
          in_all = false;
          break;
        }
      if (in_all) result.s.insert(m);
    }
  } else {  // kDiff: base minus all others
    for (const auto& m : base->s) {
      bool in_any = false;
      for (const auto* o : others)
        if (o->s.count(m)) {
          in_any = true;
          break;
        }
      if (!in_any) result.s.insert(m);
    }
  }
  return result;
}

void CmdSetOp(Database& db, ClientState& cs, Args argv, Out out, SetOp op) {
  bool ok = false;
  SetValue res = EvalSetOp(db, cs.db_index, argv, 1, out, &ok, op);
  if (!ok) return;
  resp::AppendBulkArray(out, SetMembersSorted(res));
}

void CmdSetOpStore(Database& db, ClientState& cs, Args argv, Out out,
                   SetOp op) {
  const std::string& dst = argv[1];
  bool ok = false;
  SetValue res = EvalSetOp(db, cs.db_index, argv, 2, out, &ok, op);
  if (!ok) return;
  db.Set(cs.db_index, dst, std::move(res));
  Entry* e = db.Lookup(cs.db_index, dst);
  auto* sp = std::get_if<SetValue>(&e->value);
  resp::AppendIntReply(out, (long long)sp->s.size());
}

// ---------------- zset commands ----------------
ZSetValue* ZSetCreate(Database& db, int idx, const std::string& key) {
  Entry* e = db.Lookup(idx, key);
  if (!e) {
    db.Set(idx, key, ZSetValue{});
    e = db.Lookup(idx, key);
  }
  return std::get_if<ZSetValue>(&e->value);
}

// ---------------- key / TTL ----------------
void CmdExpire(Database& db, ClientState& cs, Args argv, Out out,
               bool at, bool ms) {
  const std::string& key = argv[1];
  long long v = 0;
  if (!ParseIntStrict(argv[2], &v)) {
    ErrNotInt(out);
    return;
  }
  Entry* e = db.Lookup(cs.db_index, key);
  if (!e) {
    resp::AppendIntReply(out, 0);
    return;
  }
  long long expires = 0;
  if (at) {
    expires = ms ? v : v * 1000;
  } else {
    expires = NowMs() + (ms ? v : v * 1000);
  }
  e->expires_at_ms = expires;
  resp::AppendIntReply(out, 1);
}

// ---------------- server commands ----------------
std::string BuildInfo(Database& db, const ServerOptions& opts,
                      const ServerStats& stats, const std::string& section) {
  std::ostringstream oss;
  long long uptime_s =
      stats.start_time_us ? (NowUs() - stats.start_time_us) / 1000000 : 0;
  auto kv = [&](const std::string& k, const std::string& v) {
    oss << k << ":" << v << "\r\n";
  };
  auto want = [&](const char* name) {
    return section.empty() || section == "all" || section == "default" ||
           section == name;
  };
  if (want("Server")) {
    oss << "# Server\r\n";
    kv("redis_version", "7.4.0-tiny");
    kv("redis_mode", "standalone");
    kv("os", "Darwin tiny-redis-runtime");
    kv("run_id", "0000000000000000000000000000000000000000");
    kv("tcp_port", std::to_string(opts.port));
    kv("uptime_in_seconds", std::to_string(uptime_s));
  }
  if (want("Clients")) {
    oss << "# Clients\r\n";
    kv("connected_clients", std::to_string(stats.connected_clients));
  }
  if (want("Memory")) {
    oss << "# Memory\r\n";
    kv("used_memory", std::to_string(db.used_memory()));
    kv("used_memory_human", std::to_string(db.used_memory() / 1024) + "K");
    kv("maxmemory", std::to_string(db.max_memory()));
    kv("maxmemory_policy", "noeviction");
  }
  if (want("Persistence")) {
    oss << "# Persistence\r\n";
    kv("loading", "0");
    kv("rdb_last_save_time", std::to_string(rdb::last_save_time()));
  }
  if (want("Stats")) {
    oss << "# Stats\r\n";
    kv("total_connections_received", std::to_string(stats.total_connections));
    kv("total_commands_processed", std::to_string(stats.total_commands));
    kv("expired_keys", std::to_string(stats.expired_keys));
  }
  if (want("Replication")) {
    oss << "# Replication\r\n";
    kv("role", "master");
    kv("connected_slaves", "0");
  }
  if (want("Keyspace")) {
    oss << "# Keyspace\r\n";
    for (int i = 0; i < Database::kNumDatabases; ++i) {
      if (db.Size(i) == 0) continue;
      long long exp = 0;
      for (const auto& kve : db.table(i))
        if (kve.second.expires_at_ms != 0) ++exp;
      kv("db" + std::to_string(i),
         "keys=" + std::to_string(db.Size(i)) + ",expires=" +
             std::to_string(exp) + ",avg_ttl=0");
    }
  }
  return oss.str();
}

}  // namespace

// ---- public: glob matcher ----
bool GlobMatch(const std::string& pattern, const std::string& str) {
  // Iterative wildcard matcher with backtracking on '*'.
  size_t p = 0, s = 0, star_p = std::string::npos, star_s = 0;
  const size_t pn = pattern.size(), sn = str.size();
  while (s < sn) {
    if (p < pn && (pattern[p] == '?' || pattern[p] == str[s])) {
      ++p;
      ++s;
    } else if (p < pn && pattern[p] == '*') {
      star_p = p++;
      star_s = s;
    } else if (star_p != std::string::npos) {
      p = star_p + 1;
      s = ++star_s;
    } else {
      return false;
    }
  }
  while (p < pn && pattern[p] == '*') ++p;
  return p == pn;
}

void Dispatch(Database& db, ClientState& cs, ServerOptions& opts,
              ServerStats& stats, Args argv, Out out) {
  if (argv.empty()) return;
  const std::string upper = ToUpper(argv[0]);
  const std::string lower = ToLower(argv[0]);

  // MULTI queueing path.
  if (cs.in_multi && upper != "MULTI" && upper != "EXEC" &&
      upper != "DISCARD" && upper != "QUIT") {
    if (!CommandExists(upper)) {
      ErrUnknownCommand(out, argv);
      cs.multi_error = true;
      return;
    }
    if (!ArityOk(upper, argv.size())) {
      ErrWrongArgs(out, lower);
      cs.multi_error = true;
      return;
    }
    resp::AppendSimpleReply(out, "QUEUED");
    cs.multi_queue.push_back(argv);
    return;
  }

  if (!CommandExists(upper)) {
    ErrUnknownCommand(out, argv);
    return;
  }
  if (!ArityOk(upper, argv.size())) {
    ErrWrongArgs(out, lower);
    return;
  }

  // Maxmemory gate for write commands.
  if (opts.maxmemory > 0 && IsWriteCommand(upper) &&
      db.used_memory() > opts.maxmemory && upper != "FLUSHDB" &&
      upper != "FLUSHALL") {
    resp::AppendErrorReply(
        out, "OOM command not allowed when used memory > 'maxmemory'.");
    return;
  }

  if (upper == "PING") {
    if (argv.size() == 2)
      resp::AppendBulkReply(out, argv[1]);
    else
      resp::AppendSimpleReply(out, "PONG");
  } else if (upper == "ECHO") {
    resp::AppendBulkReply(out, argv[1]);
  } else if (upper == "SELECT") {
    long long idx = 0;
    if (!ParseIntStrict(argv[1], &idx) || idx < 0 ||
        idx >= opts.num_databases) {
      resp::AppendErrorReply(out, "ERR DB index is out of range");
      return;
    }
    cs.db_index = (int)idx;
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "AUTH") {
    resp::AppendErrorReply(out,
                           "ERR Client sent AUTH, but no password is set");
  } else if (upper == "QUIT") {
    cs.quit_requested = true;
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "HELLO") {
    long long proto = 2;
    if (argv.size() >= 2 &&
        (!ParseIntStrict(argv[1], &proto) || proto != 2)) {
      resp::AppendErrorReply(out, "NOPROTO unsupported protocol version");
      return;
    }
    resp::AppendArrayReply(out, 14);
    for (const char* f :
         {"server", "redis", "version", "7.4.0-tiny", "proto", "2", "id",
          "1", "mode", "standalone", "role", "master", "modules", ""}) {
      resp::AppendBulkReply(out, f);
    }
  } else if (upper == "COMMAND") {
    resp::AppendEmptyArrayReply(out);
  } else if (upper == "DBSIZE") {
    resp::AppendIntReply(out, db.Size(cs.db_index));
  } else if (upper == "FLUSHDB") {
    db.FlushDb(cs.db_index);
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "FLUSHALL") {
    db.FlushAll();
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "TIME") {
    long long us = NowUs();
    resp::AppendArrayReply(out, 2);
    resp::AppendBulkReply(out, std::to_string(us / 1000000));
    resp::AppendBulkReply(out, std::to_string(us % 1000000));
  } else if (upper == "ROLE") {
    resp::AppendArrayReply(out, 3);
    resp::AppendBulkReply(out, "master");
    resp::AppendIntReply(out, 0);
    resp::AppendArrayReply(out, 0);
  } else if (upper == "WAIT") {
    resp::AppendIntReply(out, 0);
  } else if (upper == "LASTSAVE") {
    resp::AppendIntReply(out, (long long)rdb::last_save_time());
  } else if (upper == "SAVE") {
    std::string err;
    std::string path = opts.dir + "/" + opts.dbfilename;
    if (rdb::SaveToFile(path, db, &err))
      resp::AppendSimpleReply(out, "OK");
    else
      resp::AppendErrorReply(out, err);
  } else if (upper == "BGSAVE") {
    std::string err;
    std::string path = opts.dir + "/" + opts.dbfilename;
    if (rdb::BgSaveFork(path, db, &err) > 0)
      resp::AppendSimpleReply(out, "Background saving started");
    else
      resp::AppendErrorReply(out, err);
  } else if (upper == "BGREWRITEAOF") {
    resp::AppendSimpleReply(out, "Background append only file rewriting "
                                 "started");
  } else if (upper == "INFO") {
    std::string sec =
        argv.size() == 2 ? ToLower(argv[1]) : std::string();
    resp::AppendBulkReply(out, BuildInfo(db, opts, stats, sec));
  } else if (upper == "SHUTDOWN") {
    bool nosave = argv.size() == 2 && ToUpper(argv[1]) == "NOSAVE";
    if (argv.size() == 2 && !nosave && ToUpper(argv[1]) != "SAVE") {
      ErrSyntax(out);
      return;
    }
    stats.shutdown_nosave = nosave;
    stats.shutdown_requested = true;
    // No reply: the server closes connections.
  } else if (upper == "CONFIG") {
    std::string sub = ToUpper(argv[1]);
    if (sub == "GET") {
      std::string param = ToLower(argv[2]);
      std::vector<std::string> pairs;
      if (param == "maxmemory") {
        pairs = {"maxmemory", std::to_string(db.max_memory())};
      } else if (param == "port") {
        pairs = {"port", std::to_string(opts.port)};
      } else if (param == "databases") {
        pairs = {"databases", std::to_string(opts.num_databases)};
      } else if (param == "dir") {
        pairs = {"dir", opts.dir};
      } else if (param == "dbfilename") {
        pairs = {"dbfilename", opts.dbfilename};
      } else if (param == "save") {
        pairs = {"save", ""};
      } else if (param == "timeout") {
        pairs = {"timeout", "0"};
      } else {
        resp::AppendErrorReply(out, "ERR Unknown option or number of "
                                    "arguments for CONFIG GET -- '" +
                                        param + "'");
        return;
      }
      resp::AppendArrayReply(out, (long long)pairs.size());
      for (const auto& p : pairs) resp::AppendBulkReply(out, p);
    } else if (sub == "SET") {
      std::string param = ToLower(argv[1 + 1]);
      if (param == "maxmemory") {
        long long cap = 0;
        if (!ParseIntStrict(argv[3], &cap) || cap < 0) {
          ErrNotInt(out);
          return;
        }
        db.set_max_memory((size_t)cap);
        resp::AppendSimpleReply(out, "OK");
      } else {
        resp::AppendErrorReply(out, "ERR Unsupported CONFIG parameter: " +
                                        param);
      }
    } else {
      ErrSyntax(out);
    }
  } else if (upper == "CLIENT") {
    std::string sub = ToUpper(argv[1]);
    if (sub == "ID") {
      resp::AppendIntReply(out, (long long)cs.id);
    } else if (sub == "GETNAME") {
      if (cs.name.empty())
        resp::AppendNullBulkReply(out);
      else
        resp::AppendBulkReply(out, cs.name);
    } else if (sub == "SETNAME") {
      cs.name = argv[2];
      resp::AppendSimpleReply(out, "OK");
    } else if (sub == "INFO") {
      std::ostringstream oss;
      oss << "id=" << cs.id << " addr=127.0.0.1:0 laddr=127.0.0.1:" <<
             opts.port << " name=" << cs.name << " db=" << cs.db_index;
      resp::AppendBulkReply(out, oss.str());
    } else if (sub == "LIST") {
      std::ostringstream oss;
      oss << "id=" << cs.id << " addr=127.0.0.1:0 fd=-1 name=" << cs.name
          << " db=" << cs.db_index << " cmd=client";
      resp::AppendBulkReply(out, oss.str());
    } else {
      resp::AppendErrorReply(out, "ERR Unknown CLIENT subcommand or wrong "
                                  "number of arguments for '" +
                                      ToLower(argv[1]) + "'");
    }
  } else if (upper == "MEMORY") {
    if (ToUpper(argv[1]) == "USAGE") {
      Entry* e = db.Lookup(cs.db_index, argv[2]);
      if (!e) {
        resp::AppendNullBulkReply(out);
      } else {
        resp::AppendIntReply(out,
                              (long long)(48 + argv[2].size() +
                                           Database::ValueSize(e->value)));
      }
    } else {
      resp::AppendErrorReply(out, "ERR syntax error");
    }
  }
  // ---------- string ----------
  else if (upper == "GET") {
    auto* rs = (RawString*)nullptr;
    auto st = GetTyped<RawString>(db, cs.db_index, argv[1], &rs);
    if (st == LookupResult::kWrongType)
      ErrWrongType(out);
    else if (st == LookupResult::kMissing)
      resp::AppendNullBulkReply(out);
    else
      resp::AppendBulkReply(out, rs->s);
  } else if (upper == "SET") {
    CmdSet(db, cs, argv, out);
  } else if (upper == "SETNX") {
    if (db.Lookup(cs.db_index, argv[1])) {
      resp::AppendIntReply(out, 0);
      return;
    }
    db.Set(cs.db_index, argv[1], RawString{argv[2]});
    resp::AppendIntReply(out, 1);
  } else if (upper == "SETEX" || upper == "PSETEX") {
    long long v = 0;
    if (!ParseIntStrict(argv[2], &v) || v <= 0) {
      ErrNotInt(out);
      return;
    }
    long long expires =
        NowMs() + (upper == "SETEX" ? v * 1000 : v);
    db.Set(cs.db_index, argv[1], RawString{argv[3]}, expires);
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "GETSET") {
    auto* rs = (RawString*)nullptr;
    auto st = GetTyped<RawString>(db, cs.db_index, argv[1], &rs);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    std::string old = (st == LookupResult::kFound) ? rs->s : "";
    if (st == LookupResult::kMissing) {
      resp::AppendNullBulkReply(out);
    } else {
      resp::AppendBulkReply(out, old);
    }
    db.Set(cs.db_index, argv[1], RawString{argv[2]});
  } else if (upper == "GETDEL") {
    auto* rs = (RawString*)nullptr;
    auto st = GetTyped<RawString>(db, cs.db_index, argv[1], &rs);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    if (st == LookupResult::kMissing) {
      resp::AppendNullBulkReply(out);
      return;
    }
    resp::AppendBulkReply(out, rs->s);
    db.Delete(cs.db_index, argv[1]);
  } else if (upper == "GETRANGE") {
    auto* rs = (RawString*)nullptr;
    auto st = GetTyped<RawString>(db, cs.db_index, argv[1], &rs);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    std::string s = (st == LookupResult::kFound) ? rs->s : "";
    long long len = (long long)s.size();
    long long start = 0, end = 0;
    if (!ParseIntStrict(argv[2], &start) || !ParseIntStrict(argv[3], &end)) {
      ErrNotInt(out);
      return;
    }
    if (len == 0) {
      resp::AppendBulkReply(out, "");
      return;
    }
    if (start < 0) start += len;
    if (end < 0) end += len;
    if (start < 0) start = 0;
    if (end >= len) end = len - 1;
    if (start > end || start >= len || end < 0) {
      resp::AppendBulkReply(out, "");
      return;
    }
    resp::AppendBulkReply(out, s.substr(start, end - start + 1));
  } else if (upper == "SETRANGE") {
    long long offset = 0;
    if (!ParseIntStrict(argv[2], &offset) || offset < 0) {
      resp::AppendErrorReply(out, "ERR offset is out of range");
      return;
    }
    if (offset + (long long)argv[3].size() > 512LL * 1024 * 1024) {
      resp::AppendErrorReply(
          out, "ERR string exceeds maximum allowed size (proto-max-bulk-len)");
      return;
    }
    Entry& e = GetOrCreateStringEntry(db, cs.db_index, argv[1]);
    auto* rs = std::get_if<RawString>(&e.value);
    if (!rs) {
      ErrWrongType(out);
      return;
    }
    if (rs->s.size() < (size_t)offset + argv[3].size()) {
      rs->s.resize((size_t)offset + argv[3].size(), '\0');
    }
    rs->s.replace((size_t)offset, argv[3].size(), argv[3]);
    resp::AppendIntReply(out, (long long)rs->s.size());
  } else if (upper == "STRLEN") {
    auto* rs = (RawString*)nullptr;
    auto st = GetTyped<RawString>(db, cs.db_index, argv[1], &rs);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out, st == LookupResult::kFound
                                  ? (long long)rs->s.size()
                                  : 0);
  } else if (upper == "APPEND") {
    Entry& e = GetOrCreateStringEntry(db, cs.db_index, argv[1]);
    auto* rs = std::get_if<RawString>(&e.value);
    if (!rs) {
      ErrWrongType(out);
      return;
    }
    rs->s += argv[2];
    resp::AppendIntReply(out, (long long)rs->s.size());
  } else if (upper == "INCR") {
    std::vector<std::string> a = {argv[0], argv[1], "1"};
    CmdIncrBy(db, cs, a, out, 1, false);
  } else if (upper == "DECR") {
    std::vector<std::string> a = {argv[0], argv[1], "-1"};
    CmdIncrBy(db, cs, a, out, -1, false);
  } else if (upper == "INCRBY") {
    long long by = 0;
    if (!ParseIntStrict(argv[2], &by)) {
      ErrNotInt(out);
      return;
    }
    CmdIncrBy(db, cs, argv, out, by, false);
  } else if (upper == "DECRBY") {
    long long by = 0;
    if (!ParseIntStrict(argv[2], &by)) {
      ErrNotInt(out);
      return;
    }
    if (__builtin_sub_overflow(0LL, by, &by)) {
      ErrNotInt(out);
      return;
    }
    CmdIncrBy(db, cs, argv, out, by, false);
  } else if (upper == "INCRBYFLOAT") {
    CmdIncrBy(db, cs, argv, out, 0, true);
  } else if (upper == "MGET") {
    resp::AppendArrayReply(out, (long long)(argv.size() - 1));
    for (size_t i = 1; i < argv.size(); ++i) {
      auto* rs = (RawString*)nullptr;
      auto st = GetTyped<RawString>(db, cs.db_index, argv[i], &rs);
      if (st == LookupResult::kFound)
        resp::AppendBulkReply(out, rs->s);
      else
        resp::AppendNullBulkReply(out);
    }
  } else if (upper == "MSET" || upper == "MSETNX") {
    if ((argv.size() - 1) % 2 != 0) {
      ErrWrongArgs(out, lower);
      return;
    }
    if (upper == "MSETNX") {
      for (size_t i = 1; i < argv.size(); i += 2) {
        if (db.Lookup(cs.db_index, argv[i])) {
          resp::AppendIntReply(out, 0);
          return;
        }
      }
    }
    for (size_t i = 1; i < argv.size(); i += 2) {
      db.Set(cs.db_index, argv[i], RawString{argv[i + 1]});
    }
    if (upper == "MSET")
      resp::AppendSimpleReply(out, "OK");
    else
      resp::AppendIntReply(out, 1);
  }
  // ---------- keys / TTL ----------
  else if (upper == "DEL" || upper == "UNLINK") {
    long long n = 0;
    for (size_t i = 1; i < argv.size(); ++i)
      n += db.Delete(cs.db_index, argv[i]) ? 1 : 0;
    resp::AppendIntReply(out, n);
  } else if (upper == "EXISTS") {
    long long n = 0;
    for (size_t i = 1; i < argv.size(); ++i)
      if (db.Lookup(cs.db_index, argv[i])) ++n;
    resp::AppendIntReply(out, n);
  } else if (upper == "KEYS") {
    std::vector<std::string> keys;
    std::vector<std::string> stale;
    for (const auto& kv : db.table(cs.db_index)) {
      if (KeyIsLiveExpired(kv.second)) {
        stale.push_back(kv.first);
        continue;
      }
      if (GlobMatch(argv[1], kv.first)) keys.push_back(kv.first);
    }
    for (const auto& k : stale) {
      db.Delete(cs.db_index, k);
      ++stats.expired_keys;
    }
    // Sorted for determinism (Redis returns hash-table order).
    std::sort(keys.begin(), keys.end());
    resp::AppendBulkArray(out, keys);
  } else if (upper == "SCAN") {
    long long cursor = 0;
    if (!ParseIntStrict(argv[1], &cursor) || cursor != 0) {
      resp::AppendErrorReply(out, "ERR invalid cursor");
      return;
    }
    std::string pattern = "*";
    long long scan_count = -1;  // -1: no COUNT given
    for (size_t i = 2; i < argv.size(); ++i) {
      std::string o = ToUpper(argv[i]);
      if (o == "MATCH" && i + 1 < argv.size()) {
        pattern = argv[++i];
      } else if (o == "COUNT" && i + 1 < argv.size()) {
        long long cnt = 0;
        if (!ParseIntStrict(argv[++i], &cnt) || cnt < 0) {
          ErrNotInt(out);
          return;
        }
        // Simplification: a single full pass; COUNT bounds the reply.
        scan_count = cnt;
      } else {
        ErrSyntax(out);
        return;
      }
    }
    std::vector<std::string> keys;
    for (const auto& kv : db.table(cs.db_index)) {
      if (KeyIsLiveExpired(kv.second)) continue;
      if (GlobMatch(pattern, kv.first)) keys.push_back(kv.first);
      if (scan_count >= 0 && (long long)keys.size() >= scan_count) break;
    }
    // Sorted for determinism (Redis returns hash-table order).
    std::sort(keys.begin(), keys.end());
    resp::AppendArrayReply(out, 2);
    resp::AppendBulkReply(out, "0");
    resp::AppendBulkArray(out, keys);
  } else if (upper == "RANDOMKEY") {
    Table& t = db.table(cs.db_index);
    if (t.empty()) {
      resp::AppendNullBulkReply(out);
      return;
    }
    static std::mt19937 rng(NowUs() & 0x7fffffff);
    auto it = t.begin();
    std::advance(it, rng() % t.size());
    if (KeyIsLiveExpired(it->second)) {
      std::string k = it->first;
      db.Delete(cs.db_index, k);
      ++stats.expired_keys;
      resp::AppendNullBulkReply(out);
    } else {
      resp::AppendBulkReply(out, it->first);
    }
  } else if (upper == "TYPE") {
    Entry* e = db.Lookup(cs.db_index, argv[1]);
    if (!e) {
      resp::AppendSimpleReply(out, "none");
      return;
    }
    resp::AppendSimpleReply(out, ValueTypeName(TypeOf(e->value)));
  } else if (upper == "RENAME") {
    const std::string& oldk = argv[1];
    const std::string& newk = argv[2];
    Entry* e = db.Lookup(cs.db_index, oldk);
    if (!e) {
      resp::AppendErrorReply(out, "ERR no such key");
      return;
    }
    if (oldk == newk) {
      resp::AppendSimpleReply(out, "OK");
      return;
    }
    Entry copy = std::move(*e);
    db.Delete(cs.db_index, oldk);
    db.Set(cs.db_index, newk, std::move(copy.value), copy.expires_at_ms);
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "EXPIRE" || upper == "PEXPIRE" ||
             upper == "EXPIREAT" || upper == "PEXPIREAT") {
    bool at = (upper == "EXPIREAT" || upper == "PEXPIREAT");
    bool ms = (upper == "PEXPIRE" || upper == "PEXPIREAT");
    CmdExpire(db, cs, argv, out, at, ms);
  } else if (upper == "TTL" || upper == "PTTL") {
    Entry* e = db.Lookup(cs.db_index, argv[1]);
    if (!e) {
      resp::AppendIntReply(out, -2);
      return;
    }
    if (e->expires_at_ms == 0) {
      resp::AppendIntReply(out, -1);
      return;
    }
    long long rem = e->expires_at_ms - NowMs();
    if (rem <= 0) {
      resp::AppendIntReply(out, -2);
      return;
    }
    resp::AppendIntReply(out, upper == "TTL" ? (rem + 999) / 1000 : rem);
  } else if (upper == "PERSIST") {
    Entry* e = db.Lookup(cs.db_index, argv[1]);
    if (!e || e->expires_at_ms == 0) {
      resp::AppendIntReply(out, 0);
      return;
    }
    e->expires_at_ms = 0;
    resp::AppendIntReply(out, 1);
  }
  // ---------- hash ----------
  else if (upper == "HSET" || upper == "HMSET") {
    if ((argv.size() - 2) % 2 != 0) {
      ErrWrongArgs(out, lower);
      return;
    }
    auto* h = HashOrCreate(db, cs.db_index, argv[1]);
    if (!h) {
      ErrWrongType(out);
      return;
    }
    long long added = 0;
    for (size_t i = 2; i < argv.size(); i += 2) {
      if (!h->h.count(argv[i])) ++added;
      h->h[argv[i]] = argv[i + 1];
    }
    if (upper == "HMSET")
      resp::AppendSimpleReply(out, "OK");
    else
      resp::AppendIntReply(out, added);
  } else if (upper == "HGET") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    auto it = st == LookupResult::kFound ? h->h.find(argv[2]) : h->h.end();
    if (st != LookupResult::kFound || it == h->h.end()) {
      resp::AppendNullBulkReply(out);
    } else {
      resp::AppendBulkReply(out, it->second);
    }
  } else if (upper == "HMGET") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendArrayReply(out, (long long)(argv.size() - 2));
    for (size_t i = 2; i < argv.size(); ++i) {
      auto it = st == LookupResult::kFound ? h->h.find(argv[i]) : h->h.end();
      if (it == h->h.end())
        resp::AppendNullBulkReply(out);
      else
        resp::AppendBulkReply(out, it->second);
    }
  } else if (upper == "HGETALL" || upper == "HKEYS" || upper == "HVALS") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    std::vector<std::string> keys, vals;
    for (const auto& kv : h->h) {
      keys.push_back(kv.first);
      vals.push_back(kv.second);
    }
    std::sort(keys.begin(), keys.end());
    std::sort(vals.begin(), vals.end());
    if (upper == "HGETALL") {
      resp::AppendArrayReply(out, (long long)(keys.size() * 2));
      for (size_t i = 0; i < keys.size(); ++i) {
        resp::AppendBulkReply(out, keys[i]);
        resp::AppendBulkReply(out, h->h[keys[i]]);
      }
    } else if (upper == "HKEYS") {
      resp::AppendBulkArray(out, keys);
    } else {
      resp::AppendBulkArray(out, vals);
    }
  } else if (upper == "HDEL") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long n = 0;
    for (size_t i = 2; i < argv.size(); ++i) n += h->h.erase(argv[i]);
    resp::AppendIntReply(out, n);
  } else if (upper == "HEXISTS") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out,
                         st == LookupResult::kFound && h->h.count(argv[2])
                             ? 1
                             : 0);
  } else if (upper == "HLEN") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out,
                         st == LookupResult::kFound ? (long long)h->h.size()
                                                    : 0);
  } else if (upper == "HSETNX") {
    auto* h = HashOrCreate(db, cs.db_index, argv[1]);
    if (!h) {
      ErrWrongType(out);
      return;
    }
    if (h->h.count(argv[2])) {
      resp::AppendIntReply(out, 0);
      return;
    }
    h->h[argv[2]] = argv[3];
    resp::AppendIntReply(out, 1);
  } else if (upper == "HSTRLEN") {
    auto* h = (HashValue*)nullptr;
    auto st = GetTyped<HashValue>(db, cs.db_index, argv[1], &h);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    auto it = st == LookupResult::kFound ? h->h.find(argv[2]) : h->h.end();
    resp::AppendIntReply(out, it == h->h.end() ? 0
                                               : (long long)it->second.size());
  } else if (upper == "HINCRBY") {
    long long by = 0;
    if (!ParseIntStrict(argv[3], &by)) {
      ErrNotInt(out);
      return;
    }
    auto* h = HashOrCreate(db, cs.db_index, argv[1]);
    if (!h) {
      ErrWrongType(out);
      return;
    }
    auto it = h->h.find(argv[2]);
    if (it == h->h.end()) {
      h->h[argv[2]] = std::to_string(by);
      resp::AppendIntReply(out, by);
      return;
    }
    long long cur = 0;
    if (!ParseIntStrict(it->second, &cur)) {
      ErrHashNotInt(out);
      return;
    }
    long long res;
    if (__builtin_add_overflow(cur, by, &res)) {
      ErrNotInt(out);
      return;
    }
    it->second = std::to_string(res);
    resp::AppendIntReply(out, res);
  } else if (upper == "HINCRBYFLOAT") {
    long double by = 0.0;
    if (!ParseLongDoubleStrict(argv[3], &by)) {
      ErrNotFloat(out);
      return;
    }
    auto* h = HashOrCreate(db, cs.db_index, argv[1]);
    if (!h) {
      ErrWrongType(out);
      return;
    }
    auto it = h->h.find(argv[2]);
    long double cur = 0.0;
    if (it != h->h.end() && !ParseLongDoubleStrict(it->second, &cur)) {
      ErrHashNotFloat(out);
      return;
    }
    long double res = (it == h->h.end() ? 0.0 : cur) + by;
    if (std::isnan(res) || std::isinf(res)) {
      ErrIncrOverflow(out);
      return;
    }
    std::string s = FormatLongDouble(res);
    if (it == h->h.end())
      h->h[argv[2]] = s;
    else
      it->second = s;
    resp::AppendBulkReply(out, s);
  }
  // ---------- list ----------
  else if (upper == "LPUSH" || upper == "RPUSH") {
    CmdPush(db, cs, argv, out, upper == "LPUSH");
  } else if (upper == "LPOP" || upper == "RPOP") {
    CmdPop(db, cs, argv, out, upper == "LPOP");
  } else if (upper == "LLEN") {
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out, st == LookupResult::kFound
                                  ? (long long)l->l.size()
                                  : 0);
  } else if (upper == "LRANGE") {
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long start = 0, stop = 0;
    if (!ParseIntStrict(argv[2], &start) || !ParseIntStrict(argv[3], &stop)) {
      ErrNotInt(out);
      return;
    }
    long long llen = st == LookupResult::kFound ? (long long)l->l.size() : 0;
    if (!ListIndexRange(&start, &stop, llen)) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    std::vector<std::string> items;
    long long i = 0;
    for (const auto& v : l->l) {
      if (i > stop) break;
      if (i >= start) items.push_back(v);
      ++i;
    }
    resp::AppendBulkArray(out, items);
  } else if (upper == "LINDEX") {
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long idx = 0;
    if (!ParseIntStrict(argv[2], &idx)) {
      ErrNotInt(out);
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendNullBulkReply(out);
      return;
    }
    long long llen = (long long)l->l.size();
    if (idx < 0) idx += llen;
    if (idx < 0 || idx >= llen) {
      resp::AppendNullBulkReply(out);
      return;
    }
    auto it = l->l.begin();
    std::advance(it, (size_t)idx);
    resp::AppendBulkReply(out, *it);
  } else if (upper == "LSET") {
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long idx = 0;
    if (!ParseIntStrict(argv[2], &idx)) {
      ErrNotInt(out);
      return;
    }
    if (st != LookupResult::kFound || l->l.empty()) {
      resp::AppendErrorReply(out, "ERR no such key");
      return;
    }
    long long llen = (long long)l->l.size();
    if (idx < 0) idx += llen;
    if (idx < 0 || idx >= llen) {
      resp::AppendErrorReply(out, "ERR index out of range");
      return;
    }
    auto it = l->l.begin();
    std::advance(it, (size_t)idx);
    *it = argv[3];
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "LREM") {
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long count = 0;
    if (!ParseIntStrict(argv[2], &count)) {
      ErrNotInt(out);
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendIntReply(out, 0);
      return;
    }
    long long removed = 0;
    if (count >= 0) {
      for (auto it = l->l.begin(); it != l->l.end() &&
                                  (count == 0 || removed < count);) {
        if (*it == argv[3]) {
          it = l->l.erase(it);
          ++removed;
        } else {
          ++it;
        }
      }
    } else {
      // Negative count: remove from the tail. Collect matching iterators
      // forward, then erase the last |count| of them.
      std::vector<std::list<std::string>::iterator> hits;
      for (auto it = l->l.begin(); it != l->l.end(); ++it)
        if (*it == argv[3]) hits.push_back(it);
      long long want = std::min<long long>((long long)hits.size(), -count);
      for (long long k = (long long)hits.size() - 1;
           k >= (long long)hits.size() - want; --k) {
        l->l.erase(hits[k]);
        ++removed;
      }
    }
    resp::AppendIntReply(out, removed);
  } else if (upper == "LINSERT") {
    bool before = ToUpper(argv[2]) == "BEFORE";
    if (!before && ToUpper(argv[2]) != "AFTER") {
      ErrSyntax(out);
      return;
    }
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendIntReply(out, 0);
      return;
    }
    for (auto it = l->l.begin(); it != l->l.end(); ++it) {
      if (*it == argv[3]) {
        if (before)
          l->l.insert(it, argv[4]);
        else
          l->l.insert(std::next(it), argv[4]);
        resp::AppendIntReply(out, (long long)l->l.size());
        return;
      }
    }
    resp::AppendIntReply(out, -1);
  } else if (upper == "LTRIM") {
    auto* l = (ListValue*)nullptr;
    auto st = GetTyped<ListValue>(db, cs.db_index, argv[1], &l);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long start = 0, stop = 0;
    if (!ParseIntStrict(argv[2], &start) || !ParseIntStrict(argv[3], &stop)) {
      ErrNotInt(out);
      return;
    }
    if (st != LookupResult::kFound || l->l.empty()) {
      resp::AppendSimpleReply(out, "OK");
      return;
    }
    long long llen = (long long)l->l.size();
    if (!ListIndexRange(&start, &stop, llen)) {
      db.Delete(cs.db_index, argv[1]);
      resp::AppendSimpleReply(out, "OK");
      return;
    }
    long long li = 0;
    for (auto it = l->l.begin(); it != l->l.end();) {
      if (li < start || li > stop) {
        it = l->l.erase(it);
      } else {
        ++it;
      }
      ++li;
    }
    if (l->l.empty()) db.Delete(cs.db_index, argv[1]);
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "RPOPLPUSH") {
    std::vector<std::string> a = {"LMOVE", argv[1], argv[2], "RIGHT", "LEFT"};
    CmdLMove(db, cs, a, out);
  } else if (upper == "LMOVE") {
    CmdLMove(db, cs, argv, out);
  }
  // ---------- set ----------
  else if (upper == "SADD") {
    auto* sp = SetCreate(db, cs.db_index, argv[1]);
    if (!sp) {
      ErrWrongType(out);
      return;
    }
    long long added = 0;
    for (size_t i = 2; i < argv.size(); ++i)
      if (sp->s.insert(argv[i]).second) ++added;
    resp::AppendIntReply(out, added);
  } else if (upper == "SREM") {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, cs.db_index, argv[1], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long n = 0;
    for (size_t i = 2; i < argv.size(); ++i) n += sp->s.erase(argv[i]);
    resp::AppendIntReply(out, n);
  } else if (upper == "SMEMBERS") {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, cs.db_index, argv[1], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendBulkArray(out, st == LookupResult::kFound
                                     ? SetMembersSorted(*sp)
                                     : std::vector<std::string>{});
  } else if (upper == "SISMEMBER") {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, cs.db_index, argv[1], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out,
                         st == LookupResult::kFound && sp->s.count(argv[2])
                             ? 1
                             : 0);
  } else if (upper == "SCARD") {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, cs.db_index, argv[1], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out, st == LookupResult::kFound
                                  ? (long long)sp->s.size()
                                  : 0);
  } else if (upper == "SPOP") {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, cs.db_index, argv[1], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long count = -1;
    if (argv.size() == 3) {
      if (!ParseIntStrict(argv[2], &count) || count < 0) {
        ErrNotInt(out);
        return;
      }
    }
    if (st == LookupResult::kMissing || sp->s.empty()) {
      if (count < 0)
        resp::AppendNullBulkReply(out);
      else
        resp::AppendNullBulkReply(out);
      return;
    }
    static std::mt19937 rng(NowUs() & 0x7fffffff);
    auto random_member = [&]() -> std::string {
      auto it = sp->s.begin();
      std::advance(it, rng() % sp->s.size());
      return *it;
    };
    if (count < 0) {
      std::string m = random_member();
      sp->s.erase(m);
      resp::AppendBulkReply(out, m);
      return;
    }
    if (count == 0) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    long long n = std::min<long long>(count, (long long)sp->s.size());
    std::vector<std::string> items;
    for (long long i = 0; i < n; ++i) {
      std::string m = random_member();
      sp->s.erase(m);
      items.push_back(m);
    }
    resp::AppendBulkArray(out, items);
  } else if (upper == "SRANDMEMBER") {
    auto* sp = (SetValue*)nullptr;
    auto st = GetTyped<SetValue>(db, cs.db_index, argv[1], &sp);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long count = -1;
    if (argv.size() == 3) {
      if (!ParseIntStrict(argv[2], &count)) {
        ErrNotInt(out);
        return;
      }
    }
    if (st == LookupResult::kMissing || sp->s.empty()) {
      if (count < 0 || argv.size() == 2)
        resp::AppendNullBulkReply(out);
      else
        resp::AppendEmptyArrayReply(out);
      return;
    }
    static std::mt19937 rng2(NowUs() & 0x7fffffff);
    auto random_member = [&]() {
      auto it = sp->s.begin();
      std::advance(it, rng2() % sp->s.size());
      return *it;
    };
    if (argv.size() == 2) {
      resp::AppendBulkReply(out, random_member());
      return;
    }
    if (count >= 0) {
      long long n = std::min<long long>(count, (long long)sp->s.size());
      std::vector<std::string> pool(sp->s.begin(), sp->s.end());
      std::shuffle(pool.begin(), pool.end(), rng2);
      pool.resize((size_t)n);
      std::sort(pool.begin(), pool.end());
      resp::AppendBulkArray(out, pool);
      return;
    }
    // count < 0: repeats allowed.
    std::vector<std::string> items;
    for (long long i = 0; i < -count; ++i) items.push_back(random_member());
    std::sort(items.begin(), items.end());
    resp::AppendBulkArray(out, items);
  } else if (upper == "SMOVE") {
    auto* src = (SetValue*)nullptr;
    auto sst = GetTyped<SetValue>(db, cs.db_index, argv[1], &src);
    if (sst == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    auto* dst = (SetValue*)nullptr;
    auto dst_st = GetTyped<SetValue>(db, cs.db_index, argv[2], &dst);
    if (dst_st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    if (sst != LookupResult::kFound || !src->s.count(argv[3])) {
      resp::AppendIntReply(out, 0);
      return;
    }
    if (argv[1] == argv[2]) {
      resp::AppendIntReply(out, 1);
      return;
    }
    if (dst_st == LookupResult::kMissing) {
      dst = SetCreate(db, cs.db_index, argv[2]);
    }
    src->s.erase(argv[3]);
    dst->s.insert(argv[3]);
    resp::AppendIntReply(out, 1);
  } else if (upper == "SINTER") {
    CmdSetOp(db, cs, argv, out, SetOp::kInter);
  } else if (upper == "SUNION") {
    CmdSetOp(db, cs, argv, out, SetOp::kUnion);
  } else if (upper == "SDIFF") {
    CmdSetOp(db, cs, argv, out, SetOp::kDiff);
  } else if (upper == "SINTERSTORE") {
    CmdSetOpStore(db, cs, argv, out, SetOp::kInter);
  } else if (upper == "SUNIONSTORE") {
    CmdSetOpStore(db, cs, argv, out, SetOp::kUnion);
  } else if (upper == "SDIFFSTORE") {
    CmdSetOpStore(db, cs, argv, out, SetOp::kDiff);
  }
  // ---------- zset ----------
  else if (upper == "ZADD") {
    bool nx = false, xx = false, ch = false, incr = false;
    size_t i = 2;
    for (; i < argv.size(); ++i) {
      std::string o = ToUpper(argv[i]);
      if (o == "NX")
        nx = true;
      else if (o == "XX")
        xx = true;
      else if (o == "CH")
        ch = true;
      else if (o == "INCR")
        incr = true;
      else
        break;
    }
    if (nx && xx) {
      ErrSyntax(out);
      return;
    }
    size_t rest = argv.size() - i;
    if (incr && rest != 2) {
      ErrSyntax(out);
      return;
    }
    if (rest == 0 || rest % 2 != 0) {
      ErrWrongArgs(out, lower);
      return;
    }
    auto* z = ZSetCreate(db, cs.db_index, argv[1]);
    if (!z) {
      ErrWrongType(out);
      return;
    }
    long long added = 0, changed = 0;
    std::string incr_result;
    bool incr_nil = false;
    for (; i < argv.size(); i += 2) {
      double score = 0.0;
      if (!ParseDoubleStrict(argv[i], &score) || std::isnan(score)) {
        ErrNotFloat(out);
        return;
      }
      const std::string& member = argv[i + 1];
      auto it = z->dict.find(member);
      bool exists = it != z->dict.end();
      if (incr) {
        if ((nx && !exists) || (xx && exists)) {
          incr_nil = true;
          break;
        }
        double res = (exists ? it->second : 0.0) + score;
        if (std::isnan(res) || std::isinf(res)) {
          ErrIncrOverflow(out);
          return;
        }
        if (exists) {
          z->sl.Erase(it->second, member);
          z->sl.Insert(res, member);
          it->second = res;
        } else {
          z->dict[member] = res;
          z->sl.Insert(res, member);
        }
        incr_result = FormatDouble(res);
        break;  // INCR allows exactly one pair.
      }
      if (nx && exists) continue;
      if (xx && !exists) continue;
      if (exists) {
        if (it->second != score) {
          z->sl.Erase(it->second, member);
          z->sl.Insert(score, member);
          it->second = score;
          ++changed;
        }
      } else {
        z->dict[member] = score;
        z->sl.Insert(score, member);
        ++added;
        ++changed;
      }
    }
    if (incr) {
      if (incr_nil)
        resp::AppendNullBulkReply(out);
      else
        resp::AppendBulkReply(out, incr_result);
      return;
    }
    resp::AppendIntReply(out, ch ? changed : added);
  } else if (upper == "ZSCORE") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    auto it = st == LookupResult::kFound ? z->dict.find(argv[2]) : z->dict.end();
    if (it == z->dict.end())
      resp::AppendNullBulkReply(out);
    else
      resp::AppendBulkReply(out, FormatDouble(it->second));
  } else if (upper == "ZCARD") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    resp::AppendIntReply(out, st == LookupResult::kFound
                                  ? (long long)z->dict.size()
                                  : 0);
  } else if (upper == "ZRANGE" || upper == "ZREVRANGE") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long start = 0, stop = 0;
    if (!ParseIntStrict(argv[2], &start) || !ParseIntStrict(argv[3], &stop)) {
      ErrNotInt(out);
      return;
    }
    bool with_scores = false, rev = (upper == "ZREVRANGE");
    for (size_t i = 4; i < argv.size(); ++i) {
      std::string o = ToUpper(argv[i]);
      if (o == "WITHSCORES")
        with_scores = true;
      else if (o == "REV" && upper == "ZRANGE")
        rev = true;
      else {
        ErrSyntax(out);
        return;
      }
    }
    if (st != LookupResult::kFound) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    AppendScoredPairs(out, z->sl.RangeByRank(start, stop, rev), with_scores);
  } else if (upper == "ZRANGEBYSCORE" || upper == "ZREVRANGEBYSCORE") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    // NB: ZREVRANGEBYSCORE takes MAX first, then MIN.
    bool max_first = (upper == "ZREVRANGEBYSCORE");
    ScoreBound minb = ParseScoreBound(argv[max_first ? 3 : 2]);
    ScoreBound maxb = ParseScoreBound(argv[max_first ? 2 : 3]);
    if (!minb.ok || !maxb.ok) {
      resp::AppendErrorReply(out, "ERR min or max is not a float");
      return;
    }
    bool with_scores = false;
    long long offset = 0, count = -1;
    for (size_t i = 4; i < argv.size(); ++i) {
      std::string o = ToUpper(argv[i]);
      if (o == "WITHSCORES") {
        with_scores = true;
      } else if (o == "LIMIT" && i + 2 < argv.size()) {
        if (!ParseIntStrict(argv[i + 1], &offset) ||
            !ParseIntStrict(argv[i + 2], &count)) {
          ErrNotInt(out);
          return;
        }
        i += 2;
      } else {
        ErrSyntax(out);
        return;
      }
    }
    if (st != LookupResult::kFound) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    auto range = ToRange(minb, maxb);
    AppendScoredPairs(out,
                      z->sl.RangeByScore(range,
                                         upper == "ZREVRANGEBYSCORE",
                                         offset, count),
                      with_scores);
  } else if (upper == "ZRANK" || upper == "ZREVRANK") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendNullBulkReply(out);
      return;
    }
    auto it = z->dict.find(argv[2]);
    if (it == z->dict.end()) {
      resp::AppendNullBulkReply(out);
      return;
    }
    size_t r = upper == "ZRANK" ? z->sl.Rank(it->second, argv[2])
                                : z->sl.RevRank(it->second, argv[2]);
    if (r == SIZE_MAX) {
      resp::AppendNullBulkReply(out);
    } else {
      resp::AppendIntReply(out, (long long)r);
    }
  } else if (upper == "ZCOUNT") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    ScoreBound minb = ParseScoreBound(argv[2]);
    ScoreBound maxb = ParseScoreBound(argv[3]);
    if (!minb.ok || !maxb.ok) {
      resp::AppendErrorReply(out, "ERR min or max is not a float");
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendIntReply(out, 0);
      return;
    }
    resp::AppendIntReply(out, (long long)z->sl.CountInRange(ToRange(minb, maxb)));
  } else if (upper == "ZREM") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long n = 0;
    for (size_t i = 2; i < argv.size(); ++i) {
      auto it = st == LookupResult::kFound ? z->dict.find(argv[i])
                                           : z->dict.end();
      if (it != z->dict.end()) {
        z->sl.Erase(it->second, argv[i]);
        z->dict.erase(it);
        ++n;
      }
    }
    resp::AppendIntReply(out, n);
  } else if (upper == "ZINCRBY") {
    double by = 0.0;
    if (!ParseDoubleStrict(argv[2], &by) || std::isnan(by)) {
      ErrNotFloat(out);
      return;
    }
    auto* z = ZSetCreate(db, cs.db_index, argv[1]);
    if (!z) {
      ErrWrongType(out);
      return;
    }
    auto it = z->dict.find(argv[3]);
    double res = (it != z->dict.end() ? it->second : 0.0) + by;
    if (std::isnan(res) || std::isinf(res)) {
      ErrIncrOverflow(out);
      return;
    }
    if (it != z->dict.end()) {
      z->sl.Erase(it->second, argv[3]);
      z->sl.Insert(res, argv[3]);
      it->second = res;
    } else {
      z->dict[argv[3]] = res;
      z->sl.Insert(res, argv[3]);
    }
    resp::AppendBulkReply(out, FormatDouble(res));
  } else if (upper == "ZPOPMIN" || upper == "ZPOPMAX") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long count = 1;
    if (argv.size() == 3) {
      if (!ParseIntStrict(argv[2], &count) || count < 0) {
        ErrNotInt(out);
        return;
      }
    }
    if (st != LookupResult::kFound) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    bool min_mode = (upper == "ZPOPMIN");
    if (count == 0) {
      resp::AppendEmptyArrayReply(out);
      return;
    }
    std::vector<std::string> flat;
    for (long long i = 0; i < count; ++i) {
      auto popped = min_mode ? z->sl.PopMin() : z->sl.PopMax();
      if (!popped) break;
      z->dict.erase(popped->second);
      flat.push_back(popped->second);
      flat.push_back(FormatDouble(popped->first));
    }
    resp::AppendBulkArray(out, flat);
  } else if (upper == "ZREMRANGEBYRANK") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    long long start = 0, stop = 0;
    if (!ParseIntStrict(argv[2], &start) || !ParseIntStrict(argv[3], &stop)) {
      ErrNotInt(out);
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendIntReply(out, 0);
      return;
    }
    auto victims = z->sl.RangeByRank(start, stop, false);
    for (const auto& p : victims) {
      z->sl.Erase(p.first, p.second);
      z->dict.erase(p.second);
    }
    resp::AppendIntReply(out, (long long)victims.size());
  } else if (upper == "ZREMRANGEBYSCORE") {
    auto* z = (ZSetValue*)nullptr;
    auto st = GetTyped<ZSetValue>(db, cs.db_index, argv[1], &z);
    if (st == LookupResult::kWrongType) {
      ErrWrongType(out);
      return;
    }
    ScoreBound minb = ParseScoreBound(argv[2]);
    ScoreBound maxb = ParseScoreBound(argv[3]);
    if (!minb.ok || !maxb.ok) {
      resp::AppendErrorReply(out, "ERR min or max is not a float");
      return;
    }
    if (st != LookupResult::kFound) {
      resp::AppendIntReply(out, 0);
      return;
    }
    auto victims = z->sl.RangeByScore(ToRange(minb, maxb), false, 0, -1);
    for (const auto& p : victims) {
      z->sl.Erase(p.first, p.second);
      z->dict.erase(p.second);
    }
    resp::AppendIntReply(out, (long long)victims.size());
  }
  // ---------- transactions ----------
  else if (upper == "MULTI") {
    if (cs.in_multi) {
      resp::AppendErrorReply(out, "ERR MULTI calls can not be nested");
      return;
    }
    cs.in_multi = true;
    cs.multi_error = false;
    cs.multi_queue.clear();
    resp::AppendSimpleReply(out, "OK");
  } else if (upper == "EXEC") {
    if (!cs.in_multi) {
      resp::AppendErrorReply(out, "ERR EXEC without MULTI");
      return;
    }
    if (cs.multi_error) {
      cs.in_multi = false;
      cs.multi_queue.clear();
      resp::AppendErrorReply(
          out, "EXECABORT Transaction discarded because of previous errors.");
      return;
    }
    auto queue = std::move(cs.multi_queue);
    cs.multi_queue.clear();
    cs.in_multi = false;
    resp::AppendArrayReply(out, (long long)queue.size());
    for (const auto& cmd_argv : queue) {
      Dispatch(db, cs, opts, stats, cmd_argv, out);
    }
  } else if (upper == "DISCARD") {
    if (!cs.in_multi) {
      resp::AppendErrorReply(out, "ERR DISCARD without MULTI");
      return;
    }
    cs.in_multi = false;
    cs.multi_error = false;
    cs.multi_queue.clear();
    resp::AppendSimpleReply(out, "OK");
  } else {
    ErrUnknownCommand(out, argv);
  }
}

const char* ValueTypeName(ValueType t) {
  switch (t) {
    case ValueType::kString:
      return "string";
    case ValueType::kHash:
      return "hash";
    case ValueType::kList:
      return "list";
    case ValueType::kSet:
      return "set";
    case ValueType::kZSet:
      return "zset";
  }
  return "none";
}

ValueType TypeOf(const Value& v) {
  switch (v.index()) {
    case 0:
      return ValueType::kString;
    case 1:
      return ValueType::kHash;
    case 2:
      return ValueType::kList;
    case 3:
      return ValueType::kSet;
    default:
      return ValueType::kZSet;
  }
}

}  // namespace tinyredis
