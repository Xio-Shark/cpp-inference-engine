// Single-threaded kqueue reactor (the pre-Redis-6 execution model):
// one event loop owns all connections and the database, so commands run
// lock-free and are trivially atomic per command.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "commands.h"
#include "resp.h"

namespace tinyredis {

struct Connection {
  int fd = -1;
  std::string in;
  std::string out;
  resp::RequestParser parser;
  ClientState state;
  bool want_write = false;
};

class Server {
 public:
  explicit Server(ServerOptions opts);
  ~Server();

  // Creates the listen socket and the kqueue. Returns false with *err set.
  bool Start(std::string* err);
  // Event loop; blocks until shutdown is requested.
  void Run();
  // Graceful: flag the loop to stop (optionally saving the RDB on exit).
  void RequestShutdown(bool save);

  Database& db() { return db_; }

 private:
  void OnAccept();
  void OnReadable(Connection& c);
  void OnWritable(Connection& c);
  void ProcessBuffered(Connection& c);
  void CloseConn(Connection& c);
  void CloseAll();
  void EnableWrite(Connection& c, bool enable);
  void PeriodicTasks();  // active expire + child reaping

  ServerOptions opts_;
  ServerStats stats_{};
  Database db_;
  int listen_fd_ = -1;
  int kq_ = -1;
  bool quit_ = false;
  unsigned long long next_client_id_ = 1;
  std::unordered_map<int, std::unique_ptr<Connection>> conns_;
};

}  // namespace tinyredis
