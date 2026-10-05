// CLI entry point: option parsing, RDB restore, event loop bootstrap.
#include <cstdio>
#include <cstring>
#include <string>

#include "commands.h"
#include "rdb.h"
#include "server.h"
#include "utils.h"

namespace {

void PrintUsage() {
  std::printf(
      "Usage: tiny-redis [OPTIONS]\n"
      "\n"
      "A Redis-compatible key-value server in modern C++ (RESP2, kqueue).\n"
      "Use the official redis-cli / redis-benchmark to talk to it.\n"
      "\n"
      "Options:\n"
      "  -p, --port <port>         TCP port (default 6379)\n"
      "  -b, --bind <addr>         Bind address (default 127.0.0.1)\n"
      "  -d, --dir <path>          Working dir for the dump file (default .)\n"
      "  -f, --dbfilename <name>   Dump filename (default dump.rdb)\n"
      "  -m, --maxmemory <bytes>   Max memory in bytes, 0 = unlimited\n"
      "  -v, --version             Print version\n"
      "  -h, --help                Show this help\n");
}

}  // namespace

int main(int argc, char** argv) {
  tinyredis::ServerOptions opts;
  opts.port = 6379;
  opts.bind = "127.0.0.1";
  opts.dir = ".";
  opts.dbfilename = "dump.rdb";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "-p" || a == "--port") {
      opts.port = std::atoi(next());
    } else if (a == "-b" || a == "--bind") {
      opts.bind = next();
    } else if (a == "-d" || a == "--dir") {
      opts.dir = next();
    } else if (a == "-f" || a == "--dbfilename") {
      opts.dbfilename = next();
    } else if (a == "-m" || a == "--maxmemory") {
      opts.maxmemory = (size_t)std::strtoull(next(), nullptr, 10);
    } else if (a == "-v" || a == "--version") {
      std::printf("tiny-redis 1.0.0 (redis-compatible:7.4.0)\n");
      return 0;
    } else if (a == "-h" || a == "--help") {
      PrintUsage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      PrintUsage();
      return 1;
    }
  }

  tinyredis::Server server(opts);

  // Restore the dump, if any.
  std::string path = opts.dir + "/" + opts.dbfilename;
  if (tinyredis::rdb::FileExists(path)) {
    std::string err;
    if (tinyredis::rdb::LoadFromFile(path, server.db(), &err)) {
      std::fprintf(stderr, "[tiny-redis] restored %s (%lld keys)\n",
                   path.c_str(), (long long)server.db().TotalKeys());
    } else {
      std::fprintf(stderr, "[tiny-redis] restore failed: %s\n", err.c_str());
    }
  }

  std::string err;
  if (!server.Start(&err)) {
    std::fprintf(stderr, "[tiny-redis] %s\n", err.c_str());
    return 1;
  }
  std::fprintf(stderr,
               "[tiny-redis] listening on %s:%d (dbs: %d, dump: %s)\n",
               opts.bind.c_str(), opts.port, opts.num_databases,
               path.c_str());
  server.Run();
  std::fprintf(stderr, "[tiny-redis] bye\n");
  return 0;
}
