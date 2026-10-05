// Custom on-disk snapshot format ("TINYREDIS RDB v1").
//
// Layout (little-endian):
//   magic "TINYREDIS01"            11 bytes
//   u8 ndbs                        number of db slots that follow
//   per db: u8 idx, u32 count
//     per entry:
//       u8 type (0..4)             value.h ValueType
//       u32 key_len, key
//       u64 expires_at_ms          0 = no TTL
//       payload:
//         String: u32 len, bytes
//         Hash:   u32 n, { u32 klen, k, u32 vlen, v } * n
//         List:   u32 n, { u32 len, bytes } * n
//         Set:    u32 n, { u32 len, bytes } * n
//         ZSet:   u32 n, { u32 mlen, member, f64 score } * n
//   u32 crc32                      CRC32 of all preceding bytes
//
// SAVE serializes synchronously; BGSAVE forks a child that writes to a
// temp file and renames over the target, mirroring the real Redis model.
#pragma once

#include <ctime>
#include <string>

#include "db.h"

namespace tinyredis::rdb {

// Serializes `db` to `path` (synchronous). Returns true on success.
bool SaveToFile(const std::string& path, Database& db, std::string* err);

// Fork-based background save: the child writes path + ".tmp" then renames.
// The parent returns immediately. Returns the child pid (> 0) on success.
int BgSaveFork(const std::string& path, Database& db, std::string* err);

// Replaces `db` contents with the file contents. Returns true on success.
bool LoadFromFile(const std::string& path, Database& db, std::string* err);

bool FileExists(const std::string& path);

time_t last_save_time();
void set_last_save_time(time_t t);

}  // namespace tinyredis::rdb
