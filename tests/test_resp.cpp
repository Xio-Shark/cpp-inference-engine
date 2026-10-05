// RESP2 parser + encoder tests.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "resp.h"

using namespace tinyredis::resp;

static int failures = 0;
#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      ++failures;                                                \
      std::fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond, \
                   __LINE__);                                    \
    }                                                            \
  } while (0)

#define FEED(p, s) p.Feed((s), std::strlen(s))

int main() {
  // Inline commands.
  {
    RequestParser p;
    auto cmds = FEED(p, "PING\r\n");
    CHECK(cmds.size() == 1 && cmds[0][0] == "PING");
  }
  {
    RequestParser p;
    auto cmds = FEED(p, "SET a b\r\nGET a\r\n");
    CHECK(cmds.size() == 2);
    CHECK(cmds[0].size() == 3 && cmds[0][0] == "SET");
    CHECK(cmds[1].size() == 2 && cmds[1][0] == "GET");
  }
  {
    // Inline with LF-only terminators.
    RequestParser p;
    auto cmds = FEED(p, "PING\n");
    CHECK(cmds.size() == 1 && cmds[0][0] == "PING");
  }
  {
    // Inline with extra whitespace.
    RequestParser p;
    auto cmds = FEED(p, "  SET   a   b  \r\n");
    CHECK(cmds.size() == 1);
    CHECK(cmds[0] == std::vector<std::string>({"SET", "a", "b"}));
  }
  // Multibulk.
  {
    RequestParser p;
    auto cmds = FEED(p, "*3\r\n$3\r\nSET\r\n$1\r\na\r\n$1\r\nb\r\n");
    CHECK(cmds.size() == 1);
    CHECK(cmds[0] == std::vector<std::string>({"SET", "a", "b"}));
  }
  {
    // Empty bulk inside multibulk is preserved.
    RequestParser p;
    auto cmds = FEED(p, "*2\r\n$3\r\nGET\r\n$0\r\n\r\n");
    CHECK(cmds.size() == 1);
    CHECK(cmds[0] == std::vector<std::string>({"GET", ""}));
  }
  {
    // Binary-safe payload (embedded CRLF and NUL).
    RequestParser p;
    const char raw[] = "*1\r\n$4\r\na\r\nb\r\n";
    // length prefix must be the real payload length (4), so craft manually:
    std::string wire = "*1\r\n$4\r\n";
    wire += "a\r\n";
    wire += "b";
    wire += "\r\n";
    auto cmds = p.Feed(wire.data(), wire.size());
    CHECK(cmds.size() == 1);
    CHECK(cmds[0][0] == std::string("a\r\nb"));
    (void)raw;
  }
  // Chunked delivery: parser must not consume partial data.
  {
    RequestParser p;
    const char raw[] = "*2\r\n$3\r\nGET\r\n$1\r\na\r\n";
    size_t len = std::strlen(raw);
    for (size_t i = 0; i < len; ++i) {
      auto cmds = p.Feed(raw + i, 1);
      // The command only completes on the last byte.
      CHECK(cmds.size() == (i == len - 1 ? 1 : 0));
    }
  }
  // Pipelined multibulk commands in one buffer.
  {
    RequestParser p;
    auto cmds = FEED(p, "*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nPING\r\n");
    CHECK(cmds.size() == 2);
  }
  // Protocol errors.
  {
    RequestParser p;
    FEED(p, "*2\r\n$abc\r\n");
    CHECK(p.has_error());
    p.reset();
    auto cmds = FEED(p, "*1\r\n$4\r\nPING\r\n");
    CHECK(!p.has_error() && cmds.size() == 1);
  }
  {
    RequestParser p;
    FEED(p, "*99999999\r\n");
    CHECK(p.has_error());
  }
  {
    // Bulk payload not followed by CRLF.
    RequestParser p;
    FEED(p, "*1\r\n$2\r\nabXX");
    CHECK(p.has_error());
  }
  // Encoders.
  {
    std::string s;
    AppendSimpleReply(s, "OK");
    CHECK(s == "+OK\r\n");
    s.clear();
    AppendErrorReply(s, "ERR nope");
    CHECK(s == "-ERR nope\r\n");
    s.clear();
    AppendIntReply(s, -7);
    CHECK(s == ":-7\r\n");
    s.clear();
    AppendBulkReply(s, std::string_view("hi", 2));
    CHECK(s == "$2\r\nhi\r\n");
    s.clear();
    AppendBulkReply(s, std::string_view(""));
    CHECK(s == "$0\r\n\r\n");
    s.clear();
    AppendNullBulkReply(s);
    CHECK(s == "$-1\r\n");
    s.clear();
    AppendArrayReply(s, 3);
    CHECK(s == "*3\r\n");
    s.clear();
    AppendNullArrayReply(s);
    CHECK(s == "*-1\r\n");
    s.clear();
    AppendDoubleReply(s, 3.0);
    CHECK(s == "3\r\n");
    s.clear();
    AppendDoubleReply(s, 1.5);
    CHECK(s == "1.5\r\n");
    s.clear();
    AppendDoubleReply(s, 10.0);
    CHECK(s == "10\r\n");
    s.clear();
    AppendDoubleReply(s, -0.5);
    CHECK(s == "-0.5\r\n");
    s.clear();
    AppendBulkArray(s, {"a", "bb"});
    CHECK(s == "*2\r\n$1\r\na\r\n$2\r\nbb\r\n");
  }
  if (failures == 0) {
    std::printf("test_resp: OK\n");
    return 0;
  }
  std::printf("test_resp: %d FAILURES\n", failures);
  return 1;
}
