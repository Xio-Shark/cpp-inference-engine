#include "resp.h"

#include <cstdio>

namespace tinyredis::resp {

namespace {
constexpr char kCR = '\r';
constexpr char kLF = '\n';

inline bool IsCrLfAt(const std::string& b, size_t i) {
  return i + 1 < b.size() && b[i] == kCR && b[i + 1] == kLF;
}
}  // namespace

const char* RequestParser::PeekLine(size_t* line_len) {
  const size_t n = buf_.size();
  size_t i = 0;
  bool saw_cr = false;
  for (; i < n; ++i) {
    char c = buf_[i];
    if (c == kCR) {
      saw_cr = true;
    } else if (c == kLF) {
      // Line ends either at "\n" or "\r\n".
      size_t end = saw_cr ? i - 1 : i;
      *line_len = end;
      return buf_.data();
    } else {
      saw_cr = false;
    }
  }
  return nullptr;
}

void RequestParser::reset() {
  buf_.clear();
  error_msg_.clear();
}

std::vector<std::vector<std::string>> RequestParser::Feed(const char* data,
                                                         size_t n) {
  if (!has_error()) {
    buf_.append(data, n);
    if (buf_.size() > kMaxBulkBytes) {
      error_msg_ = "ERR Protocol error: too big request buffer";
    }
  }
  return ParseBuffered();
}

std::vector<std::vector<std::string>> RequestParser::ParseBuffered() {
  std::vector<std::vector<std::string>> cmds;
  if (has_error()) return cmds;
  while (true) {
    std::vector<std::string> argv;
    int rc = TryParseOne(&argv);
    if (rc != 1) break;
    cmds.push_back(std::move(argv));
  }
  return cmds;
}

int RequestParser::TryParseOne(std::vector<std::string>* out) {
  if (buf_.empty()) return 0;
  if (buf_[0] == '*') {
    return TryParseMultiBulk(out);
  }
  return TryParseInline(out);
}

int RequestParser::TryParseInline(std::vector<std::string>* out) {
  size_t line_len = 0;
  if (PeekLine(&line_len) == nullptr) {
    if (buf_.size() > kMaxInlineBytes) {
      error_msg_ = "ERR Protocol error: too big inline request";
      return -1;
    }
    return 0;
  }
  // Split on whitespace, skip empty tokens.
  size_t i = 0;
  while (i < line_len) {
    while (i < line_len && (buf_[i] == ' ' || buf_[i] == '\t')) ++i;
    size_t start = i;
    while (i < line_len && buf_[i] != ' ' && buf_[i] != '\t') ++i;
    if (i > start) {
      out->emplace_back(buf_.substr(start, i - start));
    }
  }
  size_t consumed = line_len + (IsCrLfAt(buf_, line_len) ? 2 : 1);
  buf_.erase(0, consumed);
  return 1;
}

int RequestParser::TryParseMultiBulk(std::vector<std::string>* out) {
  size_t line_len = 0;
  if (PeekLine(&line_len) == nullptr) {
    if (buf_.size() > 32) {
      error_msg_ = "ERR Protocol error: invalid multibulk length";
      return -1;
    }
    return 0;
  }
  // buf_[0] == '*' guaranteed by caller.
  std::string num_str = buf_.substr(1, line_len - 1);
  if (num_str.empty() ||
      num_str.find_first_not_of("0123456789") != std::string::npos) {
    error_msg_ = "ERR Protocol error: invalid multibulk length";
    return -1;
  }
  long long count = 0;
  try {
    count = std::stoll(num_str);
  } catch (...) {
    error_msg_ = "ERR Protocol error: invalid multibulk length";
    return -1;
  }
  if (count > (long long)kMaxMultiBulkElements) {
    error_msg_ = "ERR Protocol error: invalid multibulk length";
    return -1;
  }
  size_t pos = line_len + (IsCrLfAt(buf_, line_len) ? 2 : 1);
  if (count <= 0) {
    // "*0\r\n" or "*-1\r\n": empty command -> drop it.
    buf_.erase(0, pos);
    *out = {};
    return 1;
  }
  if (!HasBytes(pos)) return 0;
  for (long long k = 0; k < count; ++k) {
    if (pos >= buf_.size()) return 0;
    if (buf_[pos] != '$') {
      error_msg_ = "ERR Protocol error: expected '$', got '" +
                   std::string(1, buf_[pos]) + "'";
      return -1;
    }
    // Find the CRLF ending the bulk-length line.
    size_t len_line = 0;
    {
      // Local scan from `pos` for the first LF.
      size_t j = pos;
      bool saw_cr = false;
      for (; j < buf_.size(); ++j) {
        if (buf_[j] == kCR) {
          saw_cr = true;
        } else if (buf_[j] == kLF) {
          len_line = saw_cr ? j - 1 : j;
          break;
        } else {
          saw_cr = false;
        }
      }
      if (j == buf_.size()) {
        if (buf_.size() - pos > 32) {
          error_msg_ = "ERR Protocol error: invalid bulk length";
          return -1;
        }
        return 0;
      }
    }
    std::string bulk_len_str = buf_.substr(pos + 1, len_line - (pos + 1));
    if (bulk_len_str.empty() ||
        bulk_len_str.find_first_not_of("0123456789") !=
            std::string::npos) {
      error_msg_ = "ERR Protocol error: invalid bulk length";
      return -1;
    }
    long long bulk_len = 0;
    try {
      bulk_len = std::stoll(bulk_len_str);
    } catch (...) {
      error_msg_ = "ERR Protocol error: invalid bulk length";
      return -1;
    }
    if (bulk_len < 0 || bulk_len > (long long)kMaxBulkBytes) {
      error_msg_ = "ERR Protocol error: invalid bulk length";
      return -1;
    }
    size_t after_len = len_line + (IsCrLfAt(buf_, len_line) ? 2 : 1);
    if (buf_.size() < after_len + (size_t)bulk_len + 2) return 0;
    // Payload must be followed by CRLF.
    size_t tail = after_len + (size_t)bulk_len;
    if (buf_[tail] != kCR || buf_[tail + 1] != kLF) {
      error_msg_ = "ERR Protocol error: expected CRLF after bulk data";
      return -1;
    }
    out->emplace_back(buf_.substr(after_len, bulk_len));
    pos = tail + 2;
  }
  buf_.erase(0, pos);
  return 1;
}

void AppendBulkReply(std::string& out, std::string_view s) {
  char tmp[32];
  int n = std::snprintf(tmp, sizeof(tmp), "%zu", s.size());
  out.push_back('$');
  out.append(tmp, n);
  out.append("\r\n", 2);
  out.append(s);
  out.append("\r\n", 2);
}

void AppendNullBulkReply(std::string& out) {
  out.append("$-1\r\n", 5);
}

void AppendArrayReply(std::string& out, long long n) {
  char tmp[32];
  int n2 = std::snprintf(tmp, sizeof(tmp), "%lld", n);
  out.push_back('*');
  out.append(tmp, n2);
  out.append("\r\n", 2);
}

void AppendNullArrayReply(std::string& out) {
  out.append("*-1\r\n", 5);
}

void AppendEmptyArrayReply(std::string& out) {
  out.append("*0\r\n", 4);
}

void AppendBulkArray(std::string& out, const std::vector<std::string>& items) {
  AppendArrayReply(out, (long long)items.size());
  for (const auto& s : items) AppendBulkReply(out, s);
}

void AppendDoubleReply(std::string& out, double d) {
  char tmp[64];
  int n = 0;
  if (d != d) {  // NaN
    n = std::snprintf(tmp, sizeof(tmp), "nan");
  } else if (d == 0.0) {
    // Redis prints "0" for both +0/-0.
    n = std::snprintf(tmp, sizeof(tmp), "0");
  } else {
    n = std::snprintf(tmp, sizeof(tmp), "%.17g", d);
  }
  out.append(tmp, n);
  out.append("\r\n", 2);
}

}  // namespace tinyredis::resp
