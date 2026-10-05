#!/bin/bash
# Smoke + parity test:
#   1. starts tiny-redis and a real redis-server side by side,
#   2. runs a fixed command matrix through both via the official
#      redis-cli and compares the (sorted) output,
#   3. runs redis-benchmark against both and prints the QPS table.
#
# Requires: redis-cli, redis-server, redis-benchmark (brew install redis).
set -u

PORT_TINY=${PORT_TINY:-6398}
PORT_REAL=${PORT_REAL:-6399}
BIN=${BIN:-build/tiny-redis}
WORK=$(mktemp -d /tmp/tiny-redis-smoke.XXXXXX)

cleanup() {
  [ -n "${TINY_PID:-}" ] && kill "$TINY_PID" 2>/dev/null
  [ -n "${REAL_PID:-}" ] && kill "$REAL_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== building =="
if command -v cmake >/dev/null 2>&1; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null || exit 1
  cmake --build build -j"$(sysctl -n hw.ncpu)" >/dev/null || exit 1
else
  make build || exit 1
fi

echo "== starting servers =="
"$BIN" -p "$PORT_TINY" -d "$WORK" &
TINY_PID=$!
redis-server --port "$PORT_REAL" --dir "$WORK" --save '' --appendonly no \
  --daemonize no --protected-mode no --logfile "$WORK/real.log" &
REAL_PID=$!
for i in $(seq 1 50); do
  redis-cli -p "$PORT_TINY" ping >/dev/null 2>&1 && \
  redis-cli -p "$PORT_REAL" ping >/dev/null 2>&1 && break
  sleep 0.1
done
redis-cli -p "$PORT_TINY" ping >/dev/null || { echo "tiny-redis failed to start"; exit 1; }
redis-cli -p "$PORT_REAL" ping >/dev/null || { echo "redis-server failed to start"; exit 1; }

echo "== parity matrix =="
PASS=0; FAIL=0
run_both() {
  local desc="$1"; shift
  local a b
  a=$(redis-cli -p "$PORT_TINY" "$@" 2>&1 | LC_ALL=C sort)
  b=$(redis-cli -p "$PORT_REAL" "$@" 2>&1 | LC_ALL=C sort)
  if [ "$a" = "$b" ]; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
    echo "DIFF [$desc] tiny:"
    echo "$a" | head -8
    echo "DIFF [$desc] real:"
    echo "$b" | head -8
  fi
}

# Both sides start from a clean slate.
redis-cli -p "$PORT_TINY" flushall >/dev/null
redis-cli -p "$PORT_REAL" flushall >/dev/null

# String commands.
run_both "set" set k hello
run_both "get" get k
run_both "get-missing" get missing
run_both "append" append k " world"
run_both "strlen" strlen k
run_both "setrange" setrange k 0 HELLO
run_both "getrange" getrange k 0 4
run_both "getrange-neg" getrange k -5 -1
run_both "incr-missing" incr ctr
run_both "incrby" incrby ctr 10
run_both "decr" decr ctr
run_both "incrbyfloat" incrbyfloat f 1.5
run_both "incrbyfloat2" incrbyfloat f 0.1
run_both "incrbyfloat3" incrbyfloat f 10
run_both "setnx-exists" setnx k v2
run_both "setnx-new" setnx k2 v2
run_both "setex" setex k3 100 v3
run_both "ttl" ttl k3
run_both "persist" persist k3
run_both "ttl-none" ttl k3
run_both "getset" getset k3 v4
run_both "mset" mset m1 1 m2 2 m3 3
run_both "mget" mget m1 m2 missing m3
run_both "set-bad-int" incr m1
run_both "type-str" type k
# Wrongtype interactions.
run_both "wrongtype-get" get myhash
# Keys / TTL.
run_both "exists" exists k m1 nope
run_both "del" del m1 m2 m3
run_both "rename" rename k2 k2b
run_both "rename-missing" rename nope x
run_both "keys" keys "user:*"
run_both "expire" expire k2b 50
run_both "ttl2" ttl k2b
run_both "expire-missing" expire nope 5
# Hash.
run_both "hset" hset myhash f1 v1 f2 v2
run_both "hset-update" hset myhash f2 v2b
run_both "hget" hget myhash f1
run_both "hget-missing" hget myhash zz
run_both "hlen" hlen myhash
run_both "hstrlen" hstrlen myhash f2
run_both "hincrby" hincrby myhash n 5
run_both "hincrby2" hincrby myhash n -2
run_both "hincrbyfloat" hincrbyfloat myhash fn 1.5
run_both "hdel" hdel myhash n f1
run_both "hexists" hexists myhash f1
run_both "hexists2" hexists myhash f2
run_both "hsetnx" hsetnx myhash f2 x
run_both "hmget" hmget myhash f2 missing
run_both "type-hash" type myhash
# List.
run_both "rpush" rpush mylist a b c
run_both "lpush" lpush mylist z
run_both "llen" llen mylist
run_both "lrange" lrange mylist 0 -1
run_both "lrange-part" lrange mylist 1 2
run_both "lrange-neg" lrange mylist -2 -1
run_both "lindex" lindex mylist 1
run_both "lindex-neg" lindex mylist -1
run_both "lset" lset mylist 1 A
run_both "linsert-before" linsert mylist before A aa
run_both "linsert-missing" linsert mylist after NOPE x
run_both "lrem" lrem mylist 1 aa
run_both "lpop" lpop mylist
run_both "rpop-count" rpop mylist 2
run_both "rpop-missing" rpop missing 1
run_both "lpop-empty" lpop emptylist
run_both "lpush-src" rpush src 1 2 3
run_both "rpoplpush" rpoplpush src dst
run_both "dst-range" lrange dst 0 -1
run_both "lmove" lmove src dst LEFT RIGHT
run_both "ltrim-dst" ltrim dst 0 0
run_both "dst-range2" lrange dst 0 -1
run_both "type-list" type mylist
# Set.
run_both "sadd" sadd myset b a c a
run_both "scard" scard myset
run_both "sismember" sismember myset a
run_both "sismember2" sismember myset nope
run_both "sadd2" sadd myset2 a d
run_both "sunion" sunion myset myset2
run_both "sinter" sinter myset myset2
run_both "sdiff" sdiff myset myset2
run_both "smove" smove myset myset2 c
run_both "sismember3" sismember myset2 c
run_both "srem" srem myset2 c nope
run_both "type-set" type myset
# Sorted set.
run_both "zadd" zadd myz 1 a 2 b 3 c 3.5 d
run_both "zscore" zscore myz b
run_both "zscore-missing" zscore myz zz
run_both "zcard" zcard myz
run_both "zrange" zrange myz 0 -1
run_both "zrange-ws" zrange myz 0 -1 withscores
run_both "zrange-part" zrange myz 1 2
run_both "zrange-neg" zrange myz -2 -1
run_both "zrange-rev" zrange myz 0 -1 rev
run_both "zrevrange" zrevrange myz 0 1
run_both "zrank" zrank myz c
run_both "zrank-missing" zrank myz zz
run_both "zrevrank" zrevrank myz c
run_both "zincrby" zincrby myz 10 a
run_both "zcount" zcount myz 2 11
run_both "zrangebyscore" zrangebyscore myz 2 +inf
run_both "zrangebyscore-excl" zrangebyscore myz "(2 +inf"
run_both "zrangebyscore-limit" zrangebyscore myz -inf +inf limit 1 2
run_both "zrevrangebyscore" zrevrangebyscore myz +inf "(2"
run_both "zadd-nx-exists" zadd myz nx 99 a
run_both "zadd-xx-missing" zadd myz xx 99 new
run_both "zadd-ch" zadd myz ch 5 a 4 e
run_both "zadd-incr" zadd myz incr 1 b
run_both "zpopmin" zpopmin myz
run_both "zpopmax" zpopmax myz 2
run_both "zremrangebyrank" zadd myz2 1 x 2 y 3 w
run_both "zremrangebyrank2" zremrangebyrank myz2 0 0
run_both "zremrangebyscore" zremrangebyscore myz2 -inf 2
run_both "zrem" zrem myz a nope
run_both "type-zset" type myz
# Errors / protocol behaviors.
run_both "unknown" NOSUCHCMD x
run_both "wrong-arity" get
run_both "wrong-arity2" set a
run_both "select-bad" select 99
run_both "ping-msg" ping hello
run_both "echo" echo yo
run_both "getdel" getdel k2b
run_both "dbsize" dbsize
# Multi.
run_both "multi" multi
run_both "queued-set" set tx1 1
run_both "queued-get" get tx1
run_both "exec" exec
run_both "get-tx1" get tx1

echo "== parity: $PASS pass, $FAIL fail =="

echo "== redis-benchmark (tiny-redis) =="
redis-benchmark -p "$PORT_TINY" -q -n 20000 -c 50 -d 100 \
  -t set,get,incr,lpush,rpop,sadd,hset,spop,zadd,zpopmin,ping,setrange,getrange \
  2>/dev/null | sed 's/^/  /'
echo "== redis-benchmark (redis-server) =="
redis-benchmark -p "$PORT_REAL" -q -n 20000 -c 50 -d 100 \
  -t set,get,incr,lpush,rpop,sadd,hset,spop,zadd,zpopmin,ping,setrange,getrange \
  2>/dev/null | sed 's/^/  /'

[ "$FAIL" -eq 0 ] && echo "SMOKE: PASS" || { echo "SMOKE: FAIL ($FAIL diffs)"; exit 1; }
