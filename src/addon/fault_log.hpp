#pragma once
#include <string>

namespace fw {
// Records faults that end a program (where, and the call stack) to <logs_dir>\crash.txt - see fault_log.cpp.
void install_fault_log(const std::wstring& logs_dir);
}  // namespace fw
