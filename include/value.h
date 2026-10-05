// Value model: the five Redis data types, stored per-key in a variant.
#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "zskiplist.h"

namespace tinyredis {

struct RawString {
  std::string s;
};

struct HashValue {
  std::unordered_map<std::string, std::string> h;
};

struct ListValue {
  std::list<std::string> l;
};

struct SetValue {
  std::unordered_set<std::string> s;
};

// Sorted set: skiplist ordered by (score, member) plus a member->score dict.
struct ZSetValue {
  zskiplist::SkipList sl;
  std::unordered_map<std::string, double> dict;
};

using Value = std::variant<RawString, HashValue, ListValue, SetValue, ZSetValue>;

enum class ValueType : uint8_t { kString = 0, kHash, kList, kSet, kZSet };

const char* ValueTypeName(ValueType t);
ValueType TypeOf(const Value& v);

}  // namespace tinyredis
