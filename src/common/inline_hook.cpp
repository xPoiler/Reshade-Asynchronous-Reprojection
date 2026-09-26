#include "common/inline_hook.hpp"
#include <windows.h>
#include <cstdio>
#include <cstring>

namespace fw {
namespace {

// Length of a ModRM (+SIB +displacement) operand, or 0 when unsupported. RIP-relative operands
// are accepted and reported through `rip` so the trampoline can re-target their displacement.
std::size_t modrm_length(const std::uint8_t* p, bool& rip) {
    const std::uint8_t modrm = p[0];
    const std::uint8_t mod = modrm >> 6, rm = modrm & 7;
    if (mod == 3) return 1;
    std::size_t len = 1;
    if (rm == 4) {  // SIB
        const std::uint8_t base = p[1] & 7;
        ++len;
        if (mod == 0 && base == 5) return 0;  // disp32 without base: unusual, refuse
    } else if (mod == 0 && rm == 5) {
        rip = true;
        return 5;  // ModRM + disp32 (RIP-relative)
    }
    if (mod == 1) len += 1;
    if (mod == 2) len += 4;
    return len;
}

struct Decoded {
    std::size_t length = 0;
    std::size_t disp_offset = 0;  // offset of a RIP-relative disp32 inside the instruction (0 = none)
};

// One instruction from the small accepted set; length 0 when unsupported.
Decoded decode(const std::uint8_t* p) {
    Decoded out;
    std::size_t prefix = 0;
    if (p[0] == 0x66) prefix = 1;  // operand-size prefix (e.g. mov [rsp+x], r16)
    const std::uint8_t* q = p + prefix;
    bool rex_w = false;
    if ((q[0] & 0xF0) == 0x40) { rex_w = (q[0] & 8) != 0; ++q; }
    const std::size_t head = static_cast<std::size_t>(q - p);
    const std::uint8_t op = q[0];
    if (op >= 0x50 && op <= 0x57) { out.length = head + 1; return out; }  // push r64
    bool rip = false;
    std::size_t opcode = 1, imm = 0, m = 0;
    switch (op) {
        case 0x89: case 0x8B: case 0x8D:  // mov r/m,r ; mov r,r/m ; lea
        case 0x31: case 0x33: case 0x29: case 0x2B: case 0x01: case 0x03:  // xor/sub/add
        case 0x85: case 0x3B: case 0x39:  // test/cmp
            m = modrm_length(q + 1, rip); break;
        case 0x83: m = modrm_length(q + 1, rip); imm = 1; break;  // grp1 r/m, imm8
        case 0x81: m = modrm_length(q + 1, rip); imm = 4; break;  // grp1 r/m, imm32
        case 0x80: m = modrm_length(q + 1, rip); imm = 1; break;  // grp1 r/m8, imm8 (cmp byte [rip+x], 0)
        case 0xB8: case 0xB9: case 0xBA: case 0xBB: case 0xBC: case 0xBD: case 0xBE: case 0xBF:
            out.length = head + 1 + (rex_w ? 8 : 4); return out;  // mov r, imm
        case 0x0F:
            if (q[1] != 0xB6 && q[1] != 0xB7) return out;  // movzx only
            opcode = 2; m = modrm_length(q + 2, rip); break;
        default:
            return out;
    }
    if (!m) return out;
    out.length = head + opcode + m + imm;
    if (rip) out.disp_offset = head + opcode + 1;
    return out;
}

void write_abs_jmp(std::uint8_t* at, const void* destination) {
    at[0] = 0xFF; at[1] = 0x25;  // jmp [rip+0]
    std::memset(at + 2, 0, 4);
    const auto value = reinterpret_cast<std::uint64_t>(destination);
    std::memcpy(at + 6, &value, 8);
}

// Allocates one page within +-1.5 GB of target so a rel32 jump can reach it.
std::uint8_t* allocate_near(const std::uint8_t* target) {
    SYSTEM_INFO info; GetSystemInfo(&info);
    const std::uint64_t granularity = info.dwAllocationGranularity;
    const std::uint64_t origin = reinterpret_cast<std::uint64_t>(target) & ~(granularity - 1);
    const std::uint64_t range = 0x60000000ull;
    for (std::uint64_t delta = granularity; delta < range; delta += granularity) {
        for (int direction = 0; direction < 2; ++direction) {
            const std::uint64_t address = direction ? origin + delta : origin - delta;
            if (!direction && origin < delta) continue;
            void* p = VirtualAlloc(reinterpret_cast<void*>(address), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
            if (p) return static_cast<std::uint8_t*>(p);
        }
    }
    return nullptr;
}

}  // namespace

std::size_t relocatable_prologue_length(const std::uint8_t* code, std::size_t minimum) {
    std::size_t total = 0;
    while (total < minimum) {
        const std::size_t n = decode(code + total).length;
        if (!n) return 0;
        total += n;
    }
    return total <= 16 ? total : 0;
}

// Follows unconditional jumps at a function's entry: jmp rel32 (E9), jmp rel8 (EB), jmp [rip+disp32]
// (FF 25, optionally REX.W-prefixed). Returns where the code really starts.
std::uint8_t* follow_jumps(std::uint8_t* code) {
    for (int hops = 0; hops < 8; ++hops) {
        if (code[0] == 0xE9) {
            std::int32_t rel; std::memcpy(&rel, code + 1, 4);
            code = code + 5 + rel;
        } else if (code[0] == 0xEB) {
            code = code + 2 + static_cast<std::int8_t>(code[1]);
        } else if ((code[0] == 0xFF && code[1] == 0x25) || (code[0] == 0x48 && code[1] == 0xFF && code[2] == 0x25)) {
            const int prefix = code[0] == 0x48 ? 1 : 0;
            std::int32_t disp; std::memcpy(&disp, code + prefix + 2, 4);
            std::uint8_t* slot = code + prefix + 6 + disp;
            std::uint8_t* next = nullptr;
            std::memcpy(&next, slot, sizeof(next));
            if (!next) break;
            code = next;
        } else {
            break;
        }
    }
    return code;
}

bool InlineHook::install(void* target, void* detour) {
    if (target_) { error_ = "already installed"; return false; }
    // Another tool may already have hooked this function (its first instruction is a jump to that
    // tool's detour or thunk). Follow the jumps and hook where they lead: the other hook keeps working
    // and ours runs after it.
    target = follow_jumps(static_cast<std::uint8_t*>(target));
    auto* code = static_cast<std::uint8_t*>(target);
    const std::size_t length = relocatable_prologue_length(code);
    if (!length) {
        char bytes[64]; std::snprintf(bytes, sizeof(bytes), "unsupported prologue %02X %02X %02X %02X %02X %02X",
                                      code[0], code[1], code[2], code[3], code[4], code[5]);
        error_ = bytes; return false;
    }
    std::uint8_t* stub = allocate_near(code);
    if (!stub) { error_ = "no memory near target"; return false; }
    // [0..13] relay to detour, [16..] trampoline: displaced bytes + jmp back.
    write_abs_jmp(stub, detour);
    std::uint8_t* trampoline = stub + 16;
    std::memcpy(trampoline, code, length);
    // Re-target RIP-relative displacements: the stub is within +-1.5 GB, so they still fit in 32 bits.
    for (std::size_t at = 0; at < length;) {
        const Decoded insn = decode(code + at);
        if (insn.disp_offset) {
            std::int32_t disp;
            std::memcpy(&disp, code + at + insn.disp_offset, 4);
            const std::int64_t absolute = reinterpret_cast<std::int64_t>(code + at + insn.length) + disp;
            const std::int64_t moved = absolute - reinterpret_cast<std::int64_t>(trampoline + at + insn.length);
            if (moved != static_cast<std::int32_t>(moved)) { VirtualFree(stub, 0, MEM_RELEASE); error_ = "RIP-relative out of range"; return false; }
            const auto moved32 = static_cast<std::int32_t>(moved);
            std::memcpy(trampoline + at + insn.disp_offset, &moved32, 4);
        }
        at += insn.length;
    }
    write_abs_jmp(trampoline + length, code + length);
    FlushInstructionCache(GetCurrentProcess(), stub, 64);

    const auto rel = reinterpret_cast<std::int64_t>(stub) - reinterpret_cast<std::int64_t>(code + 5);
    DWORD old = 0;
    if (!VirtualProtect(code, 16, PAGE_EXECUTE_READWRITE, &old)) {
        VirtualFree(stub, 0, MEM_RELEASE); error_ = "VirtualProtect failed"; return false;
    }
    std::memcpy(saved_, code, 16);
    saved_length_ = length;
    std::uint8_t patch[8];
    std::memcpy(patch, code, 8);
    patch[0] = 0xE9;
    const auto rel32 = static_cast<std::int32_t>(rel);
    std::memcpy(patch + 1, &rel32, 4);
    if ((reinterpret_cast<std::uintptr_t>(code) & 7) == 0) {
        // Single atomic store so no thread observes a half-written jump.
        InterlockedExchange64(reinterpret_cast<volatile LONG64*>(code), *reinterpret_cast<LONG64*>(patch));
    } else {
        std::memcpy(code, patch, 5);
    }
    VirtualProtect(code, 16, old, &old);
    FlushInstructionCache(GetCurrentProcess(), code, 16);
    target_ = code; stub_ = stub; trampoline_ = trampoline;
    return true;
}

void InlineHook::remove() {
    if (!target_) return;
    DWORD old = 0;
    if (VirtualProtect(target_, 16, PAGE_EXECUTE_READWRITE, &old)) {
        if ((reinterpret_cast<std::uintptr_t>(target_) & 7) == 0)
            InterlockedExchange64(reinterpret_cast<volatile LONG64*>(target_), *reinterpret_cast<LONG64*>(saved_));
        else
            std::memcpy(target_, saved_, 5);
        VirtualProtect(target_, 16, old, &old);
        FlushInstructionCache(GetCurrentProcess(), target_, 16);
    }
    // The stub is intentionally leaked: a thread may still be returning through the trampoline.
    target_ = nullptr; stub_ = nullptr; trampoline_ = nullptr;
}

}  // namespace fw
