// Command semantics tests: exercise Dispatch directly against exact
// RESP2 replies (no sockets involved).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "commands.h"

using namespace tinyredis;

static int failures = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      ++failures;                                                \
      std::fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond, \
                   __LINE__);                                    \
    }                                                            \
  } while (0)

struct Env {
  Database db;
  ClientState cs;
  ServerOptions opts;
  ServerStats stats;
  Env() {
    opts.port = 0;
    stats.connected_clients = 1;
  }
};

static std::string Exec(Env& e, std::vector<std::string> argv) {
  std::string out;
  Dispatch(e.db, e.cs, e.opts, e.stats, argv, out);
  return out;
}
static std::vector<std::string> R(const char* name,
                                  std::vector<std::string> rest = {}) {
  std::vector<std::string> argv{name};
  for (auto& r : rest) argv.push_back(r);
  return argv;
}

int main() {
  Env e;
  // Unknown command + arity.
  CHECK(Exec(e, R("NOSUCH", {"x", "y"})) ==
        "-ERR unknown command 'NOSUCH', with args beginning with: 'x' \r\n");
  CHECK(Exec(e, R("GET")) == "-ERR wrong number of arguments for 'get' command\r\n");
  CHECK(Exec(e, R("GET", {"a", "b"})) == "-ERR wrong number of arguments for 'get' command\r\n");

  // Strings.
  CHECK(Exec(e, R("SET", {"k", "v"})) == "+OK\r\n");
  CHECK(Exec(e, R("GET", {"k"})) == "$1\r\nv\r\n");
  CHECK(Exec(e, R("GET", {"missing"})) == "$-1\r\n");
  CHECK(Exec(e, R("APPEND", {"k", "v"})) == ":2\r\n");
  CHECK(Exec(e, R("STRLEN", {"k"})) == ":2\r\n");
  CHECK(Exec(e, R("GETRANGE", {"k", "0", "0"})) == "$1\r\nv\r\n");
  CHECK(Exec(e, R("SETRANGE", {"k", "1", "XY"})) == ":3\r\n");
  CHECK(Exec(e, R("GET", {"k"})) == "$3\r\nvXY\r\n");
  CHECK(Exec(e, R("SET", {"k2", "10", "EX", "100"})) == "+OK\r\n");
  CHECK(Exec(e, R("SET", {"k2", "11", "KEEPTTL"})) == "+OK\r\n");
  CHECK(Exec(e, R("SET", {"k3", "x", "NX", "GET"})) == "$-1\r\n");  // sets k3
  CHECK(Exec(e, R("SET", {"k3", "y", "XX"})) == "+OK\r\n");  // exists: set ok
  CHECK(Exec(e, R("SET", {"k3", "z", "NX"})) == "$-1\r\n");  // exists: no-op nil
  // Wrongtype.
  CHECK(Exec(e, R("HSET", {"k", "f", "v"})) ==
        "-WRONGTYPE Operation against a key holding the wrong kind of value\r\n");
  // INCR family.
  Env e2;
  CHECK(Exec(e2, R("INCR", {"c"})) == ":1\r\n");
  CHECK(Exec(e2, R("INCRBY", {"c", "10"})) == ":11\r\n");
  CHECK(Exec(e2, R("DECRBY", {"c", "1"})) == ":10\r\n");
  CHECK(Exec(e2, R("DECR", {"c"})) == ":9\r\n");
  CHECK(Exec(e2, R("SET", {"f", "1.5"})) == "+OK\r\n");
  CHECK(Exec(e2, R("INCRBYFLOAT", {"f", "0.1"})) ==
        "$19\r\n1.60000000000000009\r\n");
  CHECK(Exec(e2, R("INCRBY", {"f", "1"})) ==
        "-ERR value is not an integer or out of range\r\n");
  CHECK(Exec(e2, R("SET", {"f", "9223372036854775807"})) == "+OK\r\n");
  CHECK(Exec(e2, R("INCR", {"f"})) ==
        "-ERR value is not an integer or out of range\r\n");
  // MSET/MGET/MSETNX.
  CHECK(Exec(e2, R("MSET", {"a", "1", "b", "2"})) == "+OK\r\n");
  CHECK(Exec(e2, R("MGET", {"a", "b", "zz"})) ==
        "*3\r\n$1\r\n1\r\n$1\r\n2\r\n$-1\r\n");
  CHECK(Exec(e2, R("MSETNX", {"a", "9", "c", "9"})) == ":0\r\n");
  CHECK(Exec(e2, R("GET", {"a"})) == "$1\r\n1\r\n");
  // Keys/TTL.
  CHECK(Exec(e2, R("EXPIRE", {"a", "100"})) == ":1\r\n");
  long long ttl = 0;
  {
    std::string r = Exec(e2, R("TTL", {"a"}));
    CHECK(r.size() > 3 && r.substr(0, 1) == ":");
    ttl = std::atoll(r.c_str() + 1);
    CHECK(ttl == 100 || ttl == 99);
  }
  CHECK(Exec(e2, R("TTL", {"b"})) == ":-1\r\n");
  CHECK(Exec(e2, R("PERSIST", {"a"})) == ":1\r\n");
  CHECK(Exec(e2, R("TTL", {"a"})) == ":-1\r\n");
  CHECK(Exec(e2, R("EXPIRE", {"nope", "10"})) == ":0\r\n");
  CHECK(Exec(e2, R("EXISTS", {"a", "b", "a"})) == ":3\r\n");
  CHECK(Exec(e2, R("DEL", {"a", "b", "nope"})) == ":2\r\n");
  CHECK(Exec(e2, R("TYPE", {"b"})) == "+none\r\n");
  CHECK(Exec(e2, R("KEYS", {"*"})) == "*2\r\n$1\r\nc\r\n$1\r\nf\r\n");
  Exec(e2, R("SET", {"user:1", "a"}));
  Exec(e2, R("SET", {"user:2", "b"}));
  CHECK(Exec(e2, R("KEYS", {"user:*"})) == "*2\r\n$6\r\nuser:1\r\n$6\r\nuser:2\r\n");
  CHECK(Exec(e2, R("RENAME", {"user:1", "user:9"})) == "+OK\r\n");
  CHECK(Exec(e2, R("RENAME", {"nope", "x"})) == "-ERR no such key\r\n");
  // SCAN simplified.
  CHECK(Exec(e2, R("SCAN", {"0", "MATCH", "user:*"})) ==
        "*2\r\n$1\r\n0\r\n*2\r\n$6\r\nuser:2\r\n$6\r\nuser:9\r\n");

  // Hash.
  Env h;
  CHECK(Exec(h, R("HSET", {"hk", "f1", "v1", "f2", "v2"})) == ":2\r\n");
  CHECK(Exec(h, R("HSET", {"hk", "f2", "v2b"})) == ":0\r\n");
  CHECK(Exec(h, R("HGET", {"hk", "f2"})) == "$3\r\nv2b\r\n");
  CHECK(Exec(h, R("HGET", {"hk", "zz"})) == "$-1\r\n");
  CHECK(Exec(h, R("HGETALL", {"hk"})) ==
        "*4\r\n$2\r\nf1\r\n$2\r\nv1\r\n$2\r\nf2\r\n$3\r\nv2b\r\n");
  CHECK(Exec(h, R("HKEYS", {"hk"})) == "*2\r\n$2\r\nf1\r\n$2\r\nf2\r\n");
  CHECK(Exec(h, R("HVALS", {"hk"})) == "*2\r\n$2\r\nv1\r\n$3\r\nv2b\r\n");
  CHECK(Exec(h, R("HLEN", {"hk"})) == ":2\r\n");
  CHECK(Exec(h, R("HSTRLEN", {"hk", "f2"})) == ":3\r\n");
  CHECK(Exec(h, R("HINCRBY", {"hk", "n", "5"})) == ":5\r\n");
  CHECK(Exec(h, R("HINCRBYFLOAT", {"hk", "n", "0.5"})) == "$3\r\n5.5\r\n");
  CHECK(Exec(h, R("HSETNX", {"hk", "n", "1"})) == ":0\r\n");
  CHECK(Exec(h, R("HDEL", {"hk", "n", "f1"})) == ":2\r\n");
  CHECK(Exec(h, R("HEXISTS", {"hk", "f1"})) == ":0\r\n");

  // List.
  Env l;
  CHECK(Exec(l, R("RPUSH", {"lk", "a", "b", "c"})) == ":3\r\n");
  CHECK(Exec(l, R("LPUSH", {"lk", "z"})) == ":4\r\n");
  CHECK(Exec(l, R("LRANGE", {"lk", "0", "-1"})) ==
        "*4\r\n$1\r\nz\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n");
  CHECK(Exec(l, R("LLEN", {"lk"})) == ":4\r\n");
  CHECK(Exec(l, R("LINDEX", {"lk", "1"})) == "$1\r\na\r\n");
  CHECK(Exec(l, R("LINDEX", {"lk", "-1"})) == "$1\r\nc\r\n");
  CHECK(Exec(l, R("LSET", {"lk", "1", "A"})) == "+OK\r\n");
  CHECK(Exec(l, R("LINDEX", {"lk", "1"})) == "$1\r\nA\r\n");
  CHECK(Exec(l, R("LINSERT", {"lk", "BEFORE", "A", "aa"})) == ":5\r\n");
  CHECK(Exec(l, R("LINSERT", {"lk", "AFTER", "zz", "x"})) == ":-1\r\n");
  CHECK(Exec(l, R("LREM", {"lk", "1", "aa"})) == ":1\r\n");
  CHECK(Exec(l, R("LPOP", {"lk"})) == "$1\r\nz\r\n");
  CHECK(Exec(l, R("RPOP", {"lk", "1"})) == "*1\r\n$1\r\nc\r\n");
  CHECK(Exec(l, R("LPOP", {"lk", "10"})) ==
        "*2\r\n$1\r\nA\r\n$1\r\nb\r\n");
  CHECK(Exec(l, R("LPOP", {"lk"})) == "$-1\r\n");
  Exec(l, R("RPUSH", {"src", "1", "2", "3"}));
  CHECK(Exec(l, R("RPOPLPUSH", {"src", "dst"})) == "$1\r\n3\r\n");
  CHECK(Exec(l, R("LRANGE", {"dst", "0", "-1"})) == "*1\r\n$1\r\n3\r\n");
  CHECK(Exec(l, R("LMOVE", {"src", "dst", "LEFT", "RIGHT"})) == "$1\r\n1\r\n");
  CHECK(Exec(l, R("LTRIM", {"dst", "1", "2"})) == "+OK\r\n");
  CHECK(Exec(l, R("LRANGE", {"dst", "0", "-1"})) == "*1\r\n$1\r\n1\r\n");

  // Set (tiny's members are emitted sorted for determinism).
  Env s;
  CHECK(Exec(s, R("SADD", {"sk", "b", "a", "c", "a"})) == ":3\r\n");
  CHECK(Exec(s, R("SMEMBERS", {"sk"})) ==
        "*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n");
  CHECK(Exec(s, R("SISMEMBER", {"sk", "a"})) == ":1\r\n");
  CHECK(Exec(s, R("SCARD", {"sk"})) == ":3\r\n");
  CHECK(Exec(s, R("SADD", {"sk2", "a", "d"})) == ":2\r\n");
  CHECK(Exec(s, R("SUNION", {"sk", "sk2", "missing"})) ==
        "*4\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n$1\r\nd\r\n");
  CHECK(Exec(s, R("SINTER", {"sk", "sk2"})) == "*1\r\n$1\r\na\r\n");
  CHECK(Exec(s, R("SDIFF", {"sk", "sk2"})) ==
        "*2\r\n$1\r\nb\r\n$1\r\nc\r\n");
  CHECK(Exec(s, R("SINTERSTORE", {"dst", "sk", "sk2"})) == ":1\r\n");
  CHECK(Exec(s, R("SMOVE", {"sk", "sk2", "c"})) == ":1\r\n");
  CHECK(Exec(s, R("SISMEMBER", {"sk2", "c"})) == ":1\r\n");
  CHECK(Exec(s, R("SREM", {"sk2", "c", "nope"})) == ":1\r\n");
  CHECK(Exec(s, R("SISMEMBER", {"sk2", "c"})) == ":0\r\n");

  // ZSet.
  Env z;
  CHECK(Exec(z, R("ZADD", {"zk", "1", "a", "2", "b", "3", "c", "3.5", "d"})) == ":4\r\n");
  CHECK(Exec(z, R("ZSCORE", {"zk", "b"})) == "$1\r\n2\r\n");
  CHECK(Exec(z, R("ZSCORE", {"zk", "zz"})) == "$-1\r\n");
  CHECK(Exec(z, R("ZCARD", {"zk"})) == ":4\r\n");
  CHECK(Exec(z, R("ZRANGE", {"zk", "0", "-1"})) ==
        "*4\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n$1\r\nd\r\n");
  CHECK(Exec(z, R("ZRANGE", {"zk", "0", "-1", "WITHSCORES"})) ==
        "*8\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n$1\r\nc\r\n$1\r\n3\r\n"
        "$1\r\nd\r\n$3\r\n3.5\r\n");
  CHECK(Exec(z, R("ZRANGE", {"zk", "0", "-1", "REV"})) ==
        "*4\r\n$1\r\nd\r\n$1\r\nc\r\n$1\r\nb\r\n$1\r\na\r\n");
  CHECK(Exec(z, R("ZREVRANGE", {"zk", "0", "0", "WITHSCORES"})) ==
        "*2\r\n$1\r\nd\r\n$3\r\n3.5\r\n");
  CHECK(Exec(z, R("ZRANK", {"zk", "c"})) == ":2\r\n");
  CHECK(Exec(z, R("ZREVRANK", {"zk", "c"})) == ":1\r\n");
  CHECK(Exec(z, R("ZRANK", {"zk", "zz"})) == "$-1\r\n");
  CHECK(Exec(z, R("ZINCRBY", {"zk", "10", "a"})) == "$2\r\n11\r\n");
  CHECK(Exec(z, R("ZCOUNT", {"zk", "2", "11"})) == ":4\r\n");
  CHECK(Exec(z, R("ZRANGEBYSCORE", {"zk", "(2", "+inf", "LIMIT", "0", "2"})) ==
        "*2\r\n$1\r\nc\r\n$1\r\nd\r\n");
  CHECK(Exec(z, R("ZREVRANGEBYSCORE", {"zk", "+inf", "(2"})) ==
        "*3\r\n$1\r\na\r\n$1\r\nd\r\n$1\r\nc\r\n");
  CHECK(Exec(z, R("ZADD", {"zk", "NX", "99", "a"})) == ":0\r\n");
  CHECK(Exec(z, R("ZADD", {"zk", "XX", "99", "new"})) == ":0\r\n");
  CHECK(Exec(z, R("ZADD", {"zk", "CH", "5", "a", "4", "e"})) == ":2\r\n");
  CHECK(Exec(z, R("ZADD", {"zk", "INCR", "1", "a"})) == "$1\r\n6\r\n");
  CHECK(Exec(z, R("ZPOPMIN", {"zk"})) == "*2\r\n$1\r\nb\r\n$1\r\n2\r\n");
  CHECK(Exec(z, R("ZPOPMAX", {"zk", "2"})) ==
        "*4\r\n$1\r\na\r\n$1\r\n6\r\n$1\r\ne\r\n$1\r\n4\r\n");
  Exec(z, R("ZADD", {"zk", "1", "x", "2", "y", "3", "w"}));
  CHECK(Exec(z, R("ZREMRANGEBYRANK", {"zk", "0", "0"})) == ":1\r\n");
  CHECK(Exec(z, R("ZREMRANGEBYSCORE", {"zk", "-inf", "2"})) == ":1\r\n");
  CHECK(Exec(z, R("ZREM", {"zk", "a", "nope"})) == ":0\r\n");
  CHECK(Exec(z, R("ZCARD", {"zk"})) == ":3\r\n");

  // Transactions.
  Env t;
  CHECK(Exec(t, R("MULTI")) == "+OK\r\n");
  CHECK(Exec(t, R("SET", {"tk", "1"})) == "+QUEUED\r\n");
  CHECK(Exec(t, R("INCR", {"tk"})) == "+QUEUED\r\n");
  CHECK(Exec(t, R("EXEC")) == "*2\r\n+OK\r\n:2\r\n");
  CHECK(Exec(t, R("EXEC")) == "-ERR EXEC without MULTI\r\n");
  CHECK(Exec(t, R("MULTI")) == "+OK\r\n");
  CHECK(Exec(t, R("BOGUS")) == "-ERR unknown command 'BOGUS'\r\n");
  CHECK(Exec(t, R("SET", {"tk2", "2"})) == "+QUEUED\r\n");
  CHECK(Exec(t, R("EXEC")) ==
        "-EXECABORT Transaction discarded because of previous errors.\r\n");
  CHECK(Exec(t, R("GET", {"tk2"})) == "$-1\r\n");
  CHECK(Exec(t, R("MULTI")) == "+OK\r\n");
  CHECK(Exec(t, R("MULTI")) == "-ERR MULTI calls can not be nested\r\n");
  CHECK(Exec(t, R("DISCARD")) == "+OK\r\n");

  // Databases.
  Env d;
  Exec(d, R("SET", {"x", "1"}));
  CHECK(Exec(d, R("DBSIZE")) == ":1\r\n");
  CHECK(Exec(d, R("SELECT", {"1"})) == "+OK\r\n");
  CHECK(Exec(d, R("DBSIZE")) == ":0\r\n");
  Exec(d, R("SET", {"y", "2"}));
  CHECK(Exec(d, R("SELECT", {"16"})) == "-ERR DB index is out of range\r\n");
  CHECK(Exec(d, R("FLUSHDB")) == "+OK\r\n");
  CHECK(Exec(d, R("DBSIZE")) == ":0\r\n");
  CHECK(Exec(d, R("FLUSHALL")) == "+OK\r\n");
  CHECK(Exec(d, R("TYPE", {"x"})) == "+none\r\n");
  // Server introspection.
  std::string info = Exec(d, R("INFO"));
  CHECK(info.find("redis_version:7.4.0-tiny") != std::string::npos);
  CHECK(info.find("role:master") != std::string::npos);
  CHECK(Exec(d, R("CONFIG", {"GET", "port"})) == "*2\r\n$4\r\nport\r\n$1\r\n0\r\n");
  CHECK(Exec(d, R("PING")) == "+PONG\r\n");
  CHECK(Exec(d, R("PING", {"hi"})) == "$2\r\nhi\r\n");
  CHECK(Exec(d, R("ECHO", {"yo"})) == "$2\r\nyo\r\n");

  if (failures == 0) {
    std::printf("test_commands: OK\n");
    return 0;
  }
  std::printf("test_commands: %d FAILURES\n", failures);
  return 1;
}
