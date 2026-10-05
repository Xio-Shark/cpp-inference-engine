// RESP2 wire protocol: incremental request parser + reply encoders.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace tinyredis::resp {

constexpr size_t kMaxInlineBytes = 64 * 1024;
constexpr size_t kMaxBulkBytes = 512ull * 1024 * 1024;
constexpr size_t kMaxMultiBulkElements = 1024;

// Incremental parser for a stream of client requests.
// Feed() appends raw bytes and extracts every fully-formed command.
// Supports both multibulk ("*2\r\n$3\r\nGET\r\n...") and inline
// ("SET a b\r\n") requests, mirroring the real Redis parser.
class RequestParser {
 public:
  // Returns all complete commands found in [data, data+n).
  // After a protocol error is reported the parser must be reset or the
  // connection dropped.
  std::vector<std::vector<std::string>> Feed(const char* data, size_t n);
  // Parse whatever is buffered (used by tests).
  std::vector<std::vector<std::string>> ParseBuffered();

  bool has_error() const { return !error_msg_.empty(); }
  const std::string& error_msg() const { return error_msg_; }
  void reset();

 private:
  // Attempts to consume exactly one command from the head of buf_.
  // Returns 1 on success, 0 if incomplete, and sets the error on failure.
  int TryParseOne(std::vector<std::string>* out);
  int TryParseInline(std::vector<std::string>* out);
  int TryParseMultiBulk(std::vector<std::string>* out);
  // Returns pointer to the payload of buf_ if a CRLF-terminated line is
  // available, else nullptr. line_len receives the length without CRLF.
  const char* PeekLine(size_t* line_len);
  bool HasBytes(size_t n) const { return buf_.size() >= n; }

  std::string buf_;
  std::string error_msg_;
};

// ---- Reply encoders (append into a shared output buffer for batching) ----
inline void AppendSimpleReply(std::string& out, std::string_view s) {
  out.push_back('+');
  out.append(s);
  out.append("\r\n", 2);
}
inline void AppendErrorReply(std::string& out, std::string_view s) {
  out.push_back('-');
  out.append(s);
  out.append("\r\n", 2);
}
inline void AppendIntReply(std::string& out, long long v) {
  char tmp[32];
  int n = std::snprintf(tmp, sizeof(tmp), "%lld", v);
  out.push_back(':');
  out.append(tmp, n);
  out.append("\r\n", 2);
}
void AppendBulkReply(std::string& out, std::string_view s);
void AppendNullBulkReply(std::string& out);
void AppendArrayReply(std::string& out, long long n);
void AppendNullArrayReply(std::string& out);
void AppendEmptyArrayReply(std::string& out);
// Encodes a list of bulk strings as an array.
void AppendBulkArray(std::string& out, const std::vector<std::string>& items);
void AppendDoubleReply(std::string& out, double d);  // %.17g, like Redis

}  // namespace tinyredis::resp
