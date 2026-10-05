// Command layer: arity table + dispatch. Shared by the server and tests.
#pragma once

#include <string>
#include <vector>

#include "db.h"

namespace tinyredis {

struct ClientState {
  int db_index = 0;
  std::string name;
  unsigned long long id = 0;
  bool in_multi = false;
  bool multi_error = false;
  std::vector<std::vector<std::string>> multi_queue;
  bool quit_requested = false;
};

struct ServerStats {
  unsigned long long total_commands = 0;
  unsigned long long total_connections = 0;
  unsigned long long rejected_connections = 0;
  unsigned long long expired_keys = 0;
  long long start_time_us = 0;
  size_t connected_clients = 0;
  bool shutdown_requested = false;
  bool shutdown_nosave = false;
};

struct ServerOptions {
  int port = 6379;
  std::string bind = "127.0.0.1";
  std::string dir = ".";
  std::string dbfilename = "dump.rdb";
  size_t maxmemory = 0;
  int num_databases = Database::kNumDatabases;
};

// Executes one command (argv[0] = name, case-insensitive) and appends the
// RESP2 reply to `out`. Handles MULTI/EXEC state internally.
void Dispatch(Database& db, ClientState& cs, ServerOptions& opts,
             ServerStats& stats, const std::vector<std::string>& argv,
             std::string& out);

// Glob pattern matcher used by KEYS / SCAN MATCH (* ? [..] with \ escape).
bool GlobMatch(const std::string& pattern, const std::string& str);

}  // namespace tinyredis
