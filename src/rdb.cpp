#include "rdb.h"

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "value.h"

namespace tinyredis::rdb {

namespace {

const char kMagic[] = "TINYREDIS01";  // 11 bytes, no NUL
constexpr size_t kMagicLen = 11;

// ---- CRC32 (IEEE, reflected 0xEDB88320) ----
uint32_t Crc32(const char* data, size_t n) {
  static uint32_t table[256];
  static bool inited = false;
  if (!inited) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[i] = c;
    }
    inited = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    crc = table[(uint8_t)crc ^ (uint8_t)data[i]] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

void PutU8(std::string& out, uint8_t v) { out.push_back((char)v); }
void PutU32(std::string& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back((char)((v >> (8 * i)) & 0xff));
}
void PutU64(std::string& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back((char)((v >> (8 * i)) & 0xff));
}
void PutF64(std::string& out, double v) {
  static_assert(sizeof(double) == 8, "need 64-bit doubles");
  char buf[8];
  std::memcpy(buf, &v, 8);
  out.append(buf, 8);
}
void PutBytes(std::string& out, const std::string& s) {
  PutU32(out, (uint32_t)s.size());
  out.append(s);
}

struct Reader {
  const std::string& buf;
  size_t pos = 0;
  bool ok = true;
  explicit Reader(const std::string& b) : buf(b) {}
  bool Need(size_t n) {
    if (!ok || pos + n > buf.size()) {
      ok = false;
      return false;
    }
    return true;
  }
  uint8_t U8() {
    if (!Need(1)) return 0;
    return (uint8_t)buf[pos++];
  }
  uint32_t U32() {
    if (!Need(4)) return 0;
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)(uint8_t)buf[pos + i] << (8 * i);
    pos += 4;
    return v;
  }
  uint64_t U64() {
    if (!Need(8)) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)(uint8_t)buf[pos + i] << (8 * i);
    pos += 8;
    return v;
  }
  double F64() {
    if (!Need(8)) return 0.0;
    char tmp[8];
    std::memcpy(tmp, buf.data() + pos, 8);
    pos += 8;
    double d;
    std::memcpy(&d, tmp, 8);
    return d;
  }
  std::string Bytes(uint32_t len) {
    if (!Need(len)) return {};
    std::string s(buf, pos, len);
    pos += len;
    return s;
  }
};

std::string Serialize(const Database& db) {
  std::string out;
  out.reserve(1 << 20);
  out.append(kMagic, kMagicLen);
  uint8_t ndbs = 0;
  std::vector<int> used;
  for (int i = 0; i < Database::kNumDatabases; ++i) {
    if (!db.table(i).empty()) {
      used.push_back(i);
    }
  }
  ndbs = (uint8_t)used.size();
  PutU8(out, ndbs);
  for (int idx : used) {
    const Table& t = db.table(idx);
    PutU8(out, (uint8_t)idx);
    PutU32(out, (uint32_t)t.size());
    for (const auto& kv : t) {
      const Entry& e = kv.second;
      PutU8(out, (uint8_t)TypeOf(e.value));
      PutBytes(out, kv.first);
      PutU64(out, (uint64_t)e.expires_at_ms);
      switch (e.value.index()) {
        case 0: {  // String
          PutBytes(out, std::get<RawString>(e.value).s);
          break;
        }
        case 1: {  // Hash
          const auto& h = std::get<HashValue>(e.value).h;
          PutU32(out, (uint32_t)h.size());
          for (const auto& hk : h) {
            PutBytes(out, hk.first);
            PutBytes(out, hk.second);
          }
          break;
        }
        case 2: {  // List
          const auto& l = std::get<ListValue>(e.value).l;
          PutU32(out, (uint32_t)l.size());
          for (const auto& s : l) PutBytes(out, s);
          break;
        }
        case 3: {  // Set
          const auto& s = std::get<SetValue>(e.value).s;
          PutU32(out, (uint32_t)s.size());
          for (const auto& e2 : s) PutBytes(out, e2);
          break;
        }
        case 4: {  // ZSet
          const auto& z = std::get<ZSetValue>(e.value);
          PutU32(out, (uint32_t)z.dict.size());
          for (const auto& mk : z.dict) {
            PutBytes(out, mk.first);
            PutF64(out, mk.second);
          }
          break;
        }
        default:
          break;
      }
    }
  }
  uint32_t crc = Crc32(out.data(), out.size());
  PutU32(out, crc);
  return out;
}

std::string GetErrnoString() {
  std::ostringstream oss;
  oss << std::strerror(errno) << " (errno " << errno << ")";
  return oss.str();
}

time_t g_last_save = 0;

}  // namespace

time_t last_save_time() { return g_last_save; }
void set_last_save_time(time_t t) { g_last_save = t; }

bool SaveToFile(const std::string& path, Database& db, std::string* err) {
  std::string data = Serialize(db);
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      if (err) *err = "cannot open " + tmp + ": " + GetErrnoString();
      return false;
    }
    f.write(data.data(), (std::streamsize)data.size());
    f.flush();
    if (!f) {
      if (err) *err = "write failed on " + tmp;
      return false;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    if (err) *err = "rename failed: " + GetErrnoString();
    return false;
  }
  set_last_save_time(::time(nullptr));
  return true;
}

int BgSaveFork(const std::string& path, Database& db, std::string* err) {
  // Serialize in the parent (single snapshot, avoids copy-on-write churn on
  // small datasets; keeps the child trivially short-lived).
  std::string data = Serialize(db);
  std::string tmp = path + ".tmp";
  pid_t pid = ::fork();
  if (pid < 0) {
    if (err) *err = "fork failed: " + GetErrnoString();
    return -1;
  }
  if (pid == 0) {
    // Child: write temp file, fsync, rename, exit.
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) ::_exit(1);
    f.write(data.data(), (std::streamsize)data.size());
    f.flush();
    if (!f) ::_exit(1);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) ::_exit(1);
    ::_exit(0);
  }
  // Parent: do not wait here (the server loop reaps with waitpid WNOHANG).
  set_last_save_time(::time(nullptr));
  return (int)pid;
}

bool FileExists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

bool LoadFromFile(const std::string& path, Database& db, std::string* err) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    if (err) *err = "cannot open " + path;
    return false;
  }
  std::streamsize size = f.tellg();
  f.seekg(0);
  std::string data((size_t)size, '\0');
  if (size > 0 && !f.read(data.data(), size)) {
    if (err) *err = "short read on " + path;
    return false;
  }
  if (data.size() < kMagicLen + 5 || std::memcmp(data.data(), kMagic, kMagicLen) != 0) {
    if (err) *err = "bad magic (not a TINYREDIS dump)";
    return false;
  }
  // Verify CRC32 trailer.
  uint32_t stored = 0;
  for (int i = 0; i < 4; ++i) {
    stored |= (uint32_t)(uint8_t)data[data.size() - 4 + i] << (8 * i);
  }
  size_t body = data.size() - 4;
  if (Crc32(data.data(), body) != stored) {
    if (err) *err = "CRC mismatch (corrupt dump)";
    return false;
  }
  Reader r(data);
  r.pos = kMagicLen;
  uint8_t ndbs = r.U8();
  if (!r.ok) {
    if (err) *err = "truncated header";
    return false;
  }
  Database fresh;
  for (uint8_t d = 0; d < ndbs; ++d) {
    uint8_t idx = r.U8();
    uint32_t count = r.U32();
    if (!r.ok || idx >= Database::kNumDatabases) {
      if (err) *err = "bad db index";
      return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
      uint8_t type = r.U8();
      uint32_t klen = r.U32();
      std::string key = r.Bytes(klen);
      uint64_t exp = r.U64();
      if (!r.ok) {
        if (err) *err = "truncated entry";
        return false;
      }
      Value v;
      switch (type) {
        case 0: {
          RawString s;
          s.s = r.Bytes(r.U32());
          v = std::move(s);
          break;
        }
        case 1: {
          HashValue h;
          uint32_t n = r.U32();
          for (uint32_t j = 0; j < n && r.ok; ++j) {
            std::string fk = r.Bytes(r.U32());
            std::string fv = r.Bytes(r.U32());
            h.h.emplace(std::move(fk), std::move(fv));
          }
          v = std::move(h);
          break;
        }
        case 2: {
          ListValue l;
          uint32_t n = r.U32();
          for (uint32_t j = 0; j < n && r.ok; ++j) {
            l.l.push_back(r.Bytes(r.U32()));
          }
          v = std::move(l);
          break;
        }
        case 3: {
          SetValue s;
          uint32_t n = r.U32();
          for (uint32_t j = 0; j < n && r.ok; ++j) {
            s.s.insert(r.Bytes(r.U32()));
          }
          v = std::move(s);
          break;
        }
        case 4: {
          ZSetValue z;
          uint32_t n = r.U32();
          for (uint32_t j = 0; j < n && r.ok; ++j) {
            std::string member = r.Bytes(r.U32());
            double score = r.F64();
            z.dict[member] = score;
            z.sl.Insert(score, std::move(member));
          }
          v = std::move(z);
          break;
        }
        default: {
          if (err) *err = "unknown value type in dump";
          return false;
        }
      }
      if (!r.ok) {
        if (err) *err = "truncated value";
        return false;
      }
      fresh.Set(idx, std::move(key), std::move(v),
                exp ? (long long)exp : 0);
    }
  }
  size_t keep_max_memory = db.max_memory();
  db.FlushAll();
  db = std::move(fresh);  // moves the vector of tables + accounting
  db.set_max_memory(keep_max_memory);
  return true;
}

}  // namespace tinyredis::rdb
