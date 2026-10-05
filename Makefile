CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -Iinclude
LDFLAGS ?= -pthread
BUILD_DIR := build

LIB_SRCS := src/resp.cpp src/zskiplist.cpp src/db.cpp src/rdb.cpp src/commands.cpp src/server.cpp
LIB_OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))

TARGET := $(BUILD_DIR)/tiny-redis
TESTS := $(BUILD_DIR)/test_resp $(BUILD_DIR)/test_zskiplist $(BUILD_DIR)/test_commands $(BUILD_DIR)/test_ttl_rdb

.PHONY: all build test clean run smoke

all: build

build: $(TARGET) $(TESTS)

$(BUILD_DIR)/.created:
	@mkdir -p $(BUILD_DIR)
	@touch $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)/.created
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tiny_main.o: src/main.cpp | $(BUILD_DIR)/.created
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(LIB_OBJS) $(BUILD_DIR)/tiny_main.o
	$(CXX) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/test_resp: tests/test_resp.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/test_zskiplist: tests/test_zskiplist.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/test_commands: tests/test_commands.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/test_ttl_rdb: tests/test_ttl_rdb.cpp $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

test: build
	@echo "Running test_resp..."
	@$(BUILD_DIR)/test_resp
	@echo "Running test_zskiplist..."
	@$(BUILD_DIR)/test_zskiplist
	@echo "Running test_ttl_rdb..."
	@$(BUILD_DIR)/test_ttl_rdb
	@echo "Running test_commands..."
	@$(BUILD_DIR)/test_commands
	@echo "All tests passed successfully!"

run: $(TARGET)
	$(TARGET)

smoke: $(TARGET)
	./scripts/smoke.sh

clean:
	rm -rf $(BUILD_DIR)
