cmake_minimum_required(VERSION 3.16)
project(tiny-redis-server CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

find_package(Threads REQUIRED)

add_library(tinyredis
  src/resp.cpp
  src/zskiplist.cpp
  src/db.cpp
  src/rdb.cpp
  src/commands.cpp
  src/server.cpp
)
target_include_directories(tinyredis PUBLIC include)
target_link_libraries(tinyredis PUBLIC Threads::Threads)

add_executable(tiny-redis src/main.cpp)
target_link_libraries(tiny-redis PRIVATE tinyredis)

enable_testing()
add_executable(test_resp tests/test_resp.cpp)
target_link_libraries(test_resp PRIVATE tinyredis)
add_test(NAME resp COMMAND test_resp)

add_executable(test_zskiplist tests/test_zskiplist.cpp)
target_link_libraries(test_zskiplist PRIVATE tinyredis)
add_test(NAME zskiplist COMMAND test_zskiplist)

add_executable(test_commands tests/test_commands.cpp)
target_link_libraries(test_commands PRIVATE tinyredis)
add_test(NAME commands COMMAND test_commands)

add_executable(test_ttl_rdb tests/test_ttl_rdb.cpp)
target_link_libraries(test_ttl_rdb PRIVATE tinyredis)
add_test(NAME ttl_rdb COMMAND test_ttl_rdb)
