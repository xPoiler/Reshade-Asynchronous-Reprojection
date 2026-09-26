#pragma once
// Minimal x64 inline hook: a 5-byte rel32 JMP at the target leads to a relay near the target,
// which jumps to the detour. The trampoline replays the displaced prologue instructions.
// Only position-independent prologue instructions (push/mov/sub/lea with stack or register
// operands) are relocated; anything else (RIP-relative, branches) makes install() fail safely.
#include <cstddef>
#include <cstdint>
#include <string>

namespace fw {

// Returns the number of bytes (>= 5) covered by whole relocatable instructions at code, or 0.
std::size_t relocatable_prologue_length(const std::uint8_t* code, std::size_t minimum = 5);

// Where a function's code really starts after any leading unconditional jumps (e.g. another tool's hook).
std::uint8_t* follow_jumps(std::uint8_t* code);

class InlineHook {
public:
    InlineHook() = default;
    InlineHook(const InlineHook&) = delete;
    InlineHook& operator=(const InlineHook&) = delete;
    ~InlineHook() { remove(); }

    // On success, original() returns a callable pointer to the unhooked function.
    bool install(void* target, void* detour);
    // Restores the original bytes. Never called while a detour may run (our module is pinned).
    void remove();
    bool installed() const { return target_ != nullptr; }
    void* original() const { return trampoline_; }
    const std::string& error() const { return error_; }

private:
    std::uint8_t* target_ = nullptr;
    std::uint8_t* stub_ = nullptr;  // relay + trampoline block
    void* trampoline_ = nullptr;
    std::uint8_t saved_[16]{};
    std::size_t saved_length_ = 0;
    std::string error_;
};

}  // namespace fw
