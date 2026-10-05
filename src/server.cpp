#include "server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rdb.h"
#include "utils.h"

namespace tinyredis {

static constexpr int kListenBacklog = 511;

static bool SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

Server::Server(ServerOptions opts) : opts_(std::move(opts)) {}

Server::~Server() {
  CloseAll();
  if (listen_fd_ >= 0) ::close(listen_fd_);
  if (kq_ >= 0) ::close(kq_);
}

bool Server::Start(std::string* err) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    *err = std::string("socket: ") + std::strerror(errno);
    return false;
  }
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)opts_.port);
  if (opts_.bind.empty() || opts_.bind == "0.0.0.0") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (::inet_pton(AF_INET, opts_.bind.c_str(), &addr.sin_addr) != 1) {
    *err = "invalid bind address: " + opts_.bind;
    return false;
  }
  if (::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
    *err = std::string("bind: ") + std::strerror(errno);
    return false;
  }
  if (::listen(listen_fd_, kListenBacklog) != 0) {
    *err = std::string("listen: ") + std::strerror(errno);
    return false;
  }
  SetNonBlocking(listen_fd_);
  kq_ = ::kqueue();
  if (kq_ < 0) {
    *err = std::string("kqueue: ") + std::strerror(errno);
    return false;
  }
  struct kevent ev;
  EV_SET(&ev, listen_fd_, EVFILT_READ, EV_ADD, 0, 0, nullptr);
  if (::kevent(kq_, &ev, 1, nullptr, 0, nullptr) < 0) {
    *err = std::string("kevent(listen): ") + std::strerror(errno);
    return false;
  }
  // Receive SIGINT/SIGTERM as kqueue events instead of dying.
  struct sigaction sa{};
  sa.sa_handler = [](int) {};
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);
  EV_SET(&ev, SIGINT, EVFILT_SIGNAL, EV_ADD, 0, 0, nullptr);
  ::kevent(kq_, &ev, 1, nullptr, 0, nullptr);
  EV_SET(&ev, SIGTERM, EVFILT_SIGNAL, EV_ADD, 0, 0, nullptr);
  ::kevent(kq_, &ev, 1, nullptr, 0, nullptr);
  ::signal(SIGPIPE, SIG_IGN);
  // Seed the skiplist RNG.
  ::srandom((unsigned)NowUs());
  stats_.start_time_us = NowUs();
  return true;
}

void Server::RequestShutdown(bool save) {
  stats_.shutdown_nosave = !save;
  stats_.shutdown_requested = true;
  quit_ = true;
}

void Server::CloseConn(Connection& c) {
  auto it = conns_.find(c.fd);
  if (it == conns_.end()) return;
  if (c.want_write) EnableWrite(c, false);
  ::close(c.fd);
  conns_.erase(it);
  if (stats_.connected_clients > 0) --stats_.connected_clients;
}

void Server::CloseAll() {
  while (!conns_.empty()) CloseConn(*conns_.begin()->second);
}

void Server::EnableWrite(Connection& c, bool enable) {
  struct kevent ev;
  if (enable && !c.want_write) {
    EV_SET(&ev, c.fd, EVFILT_WRITE, EV_ADD, 0, 0, &c);
    ::kevent(kq_, &ev, 1, nullptr, 0, nullptr);
    c.want_write = true;
  } else if (!enable && c.want_write) {
    EV_SET(&ev, c.fd, EVFILT_WRITE, EV_DELETE, 0, 0, &c);
    ::kevent(kq_, &ev, 1, nullptr, 0, nullptr);
    c.want_write = false;
  }
}

void Server::OnAccept() {
  while (true) {
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      if (errno == EINTR) continue;
      break;
    }
    ++stats_.total_connections;
    SetNonBlocking(fd);
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    auto conn = std::make_unique<Connection>();
    conn->fd = fd;
    conn->state.id = next_client_id_++;
    int cfd = fd;
    conns_[cfd] = std::move(conn);
    ++stats_.connected_clients;
    struct kevent ev;
    EV_SET(&ev, fd, EVFILT_READ, EV_ADD, 0, 0, conns_[cfd].get());
    ::kevent(kq_, &ev, 1, nullptr, 0, nullptr);
  }
}

void Server::OnWritable(Connection& c) {
  while (!c.out.empty()) {
    ssize_t n = ::write(c.fd, c.out.data(), c.out.size());
    if (n > 0) {
      c.out.erase(0, (size_t)n);
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      EnableWrite(c, true);
      return;
    }
    if (n < 0 && errno == EINTR) continue;
    CloseConn(c);
    return;
  }
  if (c.out.empty()) EnableWrite(c, false);
}

void Server::ProcessBuffered(Connection& c) {
  // Drain everything the parser already has (pipelining-friendly).
  bool shutdown_seen = false;
  std::vector<std::vector<std::string>> cmds = c.parser.ParseBuffered();
  for (auto& argv : cmds) {
    if (argv.empty()) continue;
    ++stats_.total_commands;
    Dispatch(db_, c.state, opts_, stats_, argv, c.out);
    if (c.state.quit_requested) break;
    if (stats_.shutdown_requested) {
      shutdown_seen = true;
      break;
    }
  }
  if (stats_.shutdown_requested || shutdown_seen) {
    stats_.shutdown_requested = true;
    quit_ = true;
    return;
  }
  if (c.state.quit_requested) {
    // Flush pending replies then close.
    OnWritable(c);
    if (!c.want_write) CloseConn(c);
    return;
  }
  if (c.parser.has_error()) {
    resp::AppendErrorReply(c.out, c.parser.error_msg());
    OnWritable(c);
    CloseConn(c);
    return;
  }
  if (!c.out.empty()) OnWritable(c);
}

void Server::OnReadable(Connection& c) {
  char buf[65536];
  std::vector<std::vector<std::string>> parsed_cmds;
  while (true) {
    ssize_t n = ::read(c.fd, buf, sizeof(buf));
    if (n > 0) {
      auto cmds = c.parser.Feed(buf, (size_t)n);
      for (auto& cmd : cmds) {
        parsed_cmds.push_back(std::move(cmd));
      }
      if (c.parser.has_error()) {
        resp::AppendErrorReply(c.out, c.parser.error_msg());
        OnWritable(c);
        CloseConn(c);
        return;
      }
      if ((size_t)n < sizeof(buf)) break;  // likely drained
      continue;
    }
    if (n == 0) {
      // Peer closed.
      CloseConn(c);
      return;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    if (errno == EINTR) continue;
    CloseConn(c);
    return;
  }

  bool shutdown_seen = false;
  for (auto& argv : parsed_cmds) {
    if (argv.empty()) continue;
    ++stats_.total_commands;
    Dispatch(db_, c.state, opts_, stats_, argv, c.out);
    if (c.state.quit_requested) break;
    if (stats_.shutdown_requested) {
      shutdown_seen = true;
      break;
    }
  }
  if (stats_.shutdown_requested || shutdown_seen) {
    stats_.shutdown_requested = true;
    quit_ = true;
    return;
  }
  if (c.state.quit_requested) {
    OnWritable(c);
    if (!c.want_write) CloseConn(c);
    return;
  }
  if (!c.out.empty()) OnWritable(c);
}

void Server::PeriodicTasks() {
  size_t expired = db_.ActiveExpireCycle();
  stats_.expired_keys += expired;
  // Reap BGSAVE children (non-blocking).
  while (::waitpid(-1, nullptr, WNOHANG) > 0) {
  }
}

void Server::Run() {
  struct kevent events[256];
  struct timespec ts{0, 100 * 1000 * 1000};  // 100ms cadence
  while (!quit_) {
    int n = ::kevent(kq_, nullptr, 0, events, 256, &ts);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    for (int i = 0; i < n; ++i) {
      struct kevent& ev = events[i];
      if (ev.filter == EVFILT_SIGNAL) {
        RequestShutdown(/*save=*/true);
        continue;
      }
      if (ev.ident == (uintptr_t)listen_fd_ && ev.filter == EVFILT_READ) {
        OnAccept();
        continue;
      }
      if (ev.udata == nullptr) continue;
      Connection* c = (Connection*)ev.udata;
      if (conns_.find(c->fd) == conns_.end()) continue;
      if (ev.filter == EVFILT_READ) {
        OnReadable(*c);
      } else if (ev.filter == EVFILT_WRITE) {
        OnWritable(*c);
      }
    }
    PeriodicTasks();
  }
  // Graceful shutdown.
  if (!stats_.shutdown_nosave) {
    std::string err;
    std::string path = opts_.dir + "/" + opts_.dbfilename;
    if (rdb::SaveToFile(path, db_, &err)) {
      std::fprintf(stderr, "[tiny-redis] saved %s\n", path.c_str());
    } else {
      std::fprintf(stderr, "[tiny-redis] save failed: %s\n", err.c_str());
    }
  }
  CloseAll();
}

}  // namespace tinyredis
