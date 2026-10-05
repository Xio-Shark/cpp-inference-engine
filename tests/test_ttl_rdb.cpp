// TTL (lazy + active expiry) and RDB persistence round-trip tests.
#include <cstdio>
#include <string>
#include <vector>

#include "commands.h"
#include "rdb.h"
#include "utils.h"

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

int main() {
  // ---- lazy expiry ----
  {
    Database db;
    db.Set(0, "gone", RawString{"v"}, NowMs() - 1000);
    CHECK(db.Lookup(0, "gone") == nullptr);
    CHECK(db.Size(0) == 0);
    // Not-yet-expired is visible.
    db.Set(0, "soon", RawString{"v"}, NowMs() + 100000);
    CHECK(db.Lookup(0, "soon") != nullptr);
    // Replace preserves TTL only when given; Set resets it.
    db.Set(0, "soon2", RawString{"v"}, NowMs() + 100000);
    db.Set(0, "soon2", RawString{"w"});  // TTL cleared
    Entry* e = db.Lookup(0, "soon2");
    CHECK(e && e->expires_at_ms == 0);
  }
  // ---- active expire cycle ----
  {
    Database db;
    for (int i = 0; i < 50; ++i) {
      db.Set(0, "k" + std::to_string(i), RawString{"v"},
             NowMs() - 1000);
    }
    db.Set(0, "fresh", RawString{"v"});
    size_t deleted = db.ActiveExpireCycle();
    CHECK(deleted > 0);
    CHECK(db.Lookup(0, "fresh") != nullptr);
    // Repeated cycles eventually clean everything (window 200/cycle).
    for (int i = 0; i < 5; ++i) db.ActiveExpireCycle();
    CHECK(db.Size(0) == 1);  // only "fresh" remains
  }
  // ---- TTL semantics through commands ----
  {
    Database db;
    ClientState cs;
    ServerOptions opts;
    ServerStats stats;
    auto exec = [&](std::vector<std::string> argv) {
      std::string out;
      Dispatch(db, cs, opts, stats, argv, out);
      return out;
    };
    exec({"SET", "a", "1"});
    CHECK(exec({"TTL", "a"}) == ":-1\r\n");
    CHECK(exec({"EXPIRE", "a", "100"}) == ":1\r\n");
    std::string ttl = exec({"TTL", "a"});
    CHECK(ttl == ":100\r\n" || ttl == ":99\r\n");
    CHECK(exec({"PERSIST", "a"}) == ":1\r\n");
    CHECK(exec({"PTTL", "a"}) == ":-1\r\n");
    CHECK(exec({"PEXPIRE", "a", "1"}) == ":1\r\n");
    // Wait for the millisecond expiry.
    long long deadline = NowMs() + 50;
    while (NowMs() < deadline) {
    }
    CHECK(exec({"GET", "a"}) == "$-1\r\n");
    CHECK(exec({"TTL", "a"}) == ":-2\r\n");
  }
  // ---- RDB round trip ----
  {
    Database db;
    db.Set(0, "str", RawString{"hello"});
    db.Set(0, "str-ttl", RawString{"bye"}, NowMs() + 100000);
    {
      HashValue h;
      h.h = {{"f1", "v1"}, {"f2", "v2"}};
      db.Set(0, "h", std::move(h));
    }
    {
      ListValue l;
      l.l = {"a", "b", "c"};
      db.Set(0, "l", std::move(l));
    }
    {
      SetValue s;
      s.s = {"x", "y"};
      db.Set(0, "s", std::move(s));
    }
    {
      ZSetValue z;
      z.dict = {{"m1", 1.5}, {"m2", -2.25}};
      z.sl.Insert(1.5, "m1");
      z.sl.Insert(-2.25, "m2");
      db.Set(0, "z", std::move(z));
    }
    db.Set(3, "other-db", RawString{"v3"});

    std::string path = "/tmp/tiny_redis_test_dump.rdb";
    std::string err;
    CHECK(rdb::SaveToFile(path, db, &err));
    Database loaded;
    CHECK(rdb::LoadFromFile(path, loaded, &err));
    CHECK(loaded.Lookup(0, "str") != nullptr);
    Entry* e = loaded.Lookup(0, "str");
    CHECK(std::get<RawString>(e->value).s == "hello");
    e = loaded.Lookup(0, "str-ttl");
    CHECK(e && e->expires_at_ms != 0);
    e = loaded.Lookup(0, "h");
    CHECK(e && std::get<HashValue>(e->value).h.at("f1") == "v1");
    e = loaded.Lookup(0, "l");
    CHECK(e && std::get<ListValue>(e->value).l.size() == 3);
    e = loaded.Lookup(0, "s");
    CHECK(e && std::get<SetValue>(e->value).s.count("x"));
    e = loaded.Lookup(0, "z");
    CHECK(e && std::get<ZSetValue>(e->value).dict.at("m1") == 1.5);
    e = loaded.Lookup(3, "other-db");
    CHECK(e && std::get<RawString>(e->value).s == "v3");
    CHECK(loaded.Lookup(1, "str") == nullptr);
    std::remove(path.c_str());
    // Corruption is detected.
    CHECK(rdb::SaveToFile(path, db, &err));
    {
      FILE* f = std::fopen(path.c_str(), "r+b");
      CHECK(f != nullptr);
      std::fseek(f, 20, SEEK_SET);
      std::fputc('X', f);
      std::fclose(f);
    }
    Database bad;
    CHECK(!rdb::LoadFromFile(path, bad, &err));
    std::remove(path.c_str());
  }
  if (failures == 0) {
    std::printf("test_ttl_rdb: OK\n");
    return 0;
  }
  std::printf("test_ttl_rdb: %d FAILURES\n", failures);
  return 1;
}
