#include "common/inline_hook.hpp"
#include <windows.h>
#include <cstdio>
#include <cstring>

using namespace fw;
static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); } } while (0)

using AddFn = int (*)(int, int);
static InlineHook g_hook;
static int detour(int a, int b) { return reinterpret_cast<AddFn>(g_hook.original())(a, b) * 10; }

static std::uint8_t* make_function(const std::uint8_t* code, std::size_t size) {
    auto* p = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    std::memcpy(p, code, size);
    FlushInstructionCache(GetCurrentProcess(), p, size);
    return p;
}

int main() {
    // int add(int a, int b) with a typical MSVC prologue (same shape as slSetConstants).
    const std::uint8_t add_code[] = {
        0x48, 0x89, 0x5C, 0x24, 0x08,  // mov [rsp+8], rbx
        0x57,                          // push rdi
        0x48, 0x83, 0xEC, 0x20,        // sub rsp, 20h
        0x8D, 0x04, 0x11,              // lea eax, [rcx+rdx]
        0x48, 0x83, 0xC4, 0x20,        // add rsp, 20h
        0x5F,                          // pop rdi
        0x48, 0x8B, 0x5C, 0x24, 0x08,  // mov rbx, [rsp+8]
        0xC3};
    auto* fn = make_function(add_code, sizeof(add_code));
    auto add = reinterpret_cast<AddFn>(fn);
    EXPECT(add(2, 3) == 5, "unhooked");
    EXPECT(g_hook.install(fn, reinterpret_cast<void*>(&detour)), g_hook.error().c_str());
    EXPECT(add(2, 3) == 50, "hooked");
    g_hook.remove();
    EXPECT(add(2, 3) == 5, "restored");
    EXPECT(std::memcmp(fn, add_code, sizeof(add_code)) == 0, "bytes restored");

    // Prologue shapes seen in sl.interposer.dll.
    const std::uint8_t set_tag[] = {0x40, 0x55, 0x56, 0x57, 0x41, 0x56, 0x48, 0x81, 0xEC, 0xB8, 0, 0, 0, 0x48, 0x8B, 0x05};
    EXPECT(relocatable_prologue_length(set_tag) == 6, "slSetTag prologue");
    const std::uint8_t get_req[] = {0x48, 0x83, 0xEC, 0x28, 0x4C, 0x8B, 0xC2, 0x8B, 0xD1};
    EXPECT(relocatable_prologue_length(get_req) == 7, "sub rsp + mov r8,rdx");
    const std::uint8_t rip_relative[] = {0x48, 0x8B, 0x05, 0x10, 0x20, 0x30, 0x40, 0x90};
    EXPECT(relocatable_prologue_length(rip_relative) == 7, "RIP-relative mov is relocatable");

    // int read_global(): sub rsp,28h ; mov eax,[rip+global] ; add rsp,28h ; ret  (same shape as
    // UE's ForceTagStreamlineBuffers prologue). The global lives 64 bytes after the code.
    std::uint8_t reader[80] = {0x48, 0x83, 0xEC, 0x28, 0x8B, 0x05, 0, 0, 0, 0, 0x48, 0x83, 0xC4, 0x28, 0xC3};
    const std::int32_t to_global = 64 - 10;  // disp from the end of the mov (offset 10) to offset 64
    std::memcpy(reader + 6, &to_global, 4);
    const std::int32_t value = 1234;
    std::memcpy(reader + 64, &value, 4);
    auto* rfn = make_function(reader, sizeof(reader));
    using ReadFn = int (*)();
    auto read = reinterpret_cast<ReadFn>(rfn);
    static InlineHook rip_hook;
    struct Detour { static int call() { return reinterpret_cast<ReadFn>(rip_hook.original())() + 1; } };
    EXPECT(read() == 1234, "unhooked global read");
    EXPECT(rip_hook.install(rfn, reinterpret_cast<void*>(&Detour::call)), rip_hook.error().c_str());
    EXPECT(read() == 1235, "trampoline reads the relocated RIP-relative global");
    rip_hook.remove();
    EXPECT(read() == 1234, "restored");
    // Already hooked by another tool: the entry is a jump (jmp rel32, or jmp [rip]) to the real code.
    // Our hook must land on the real code and the other tool's jump must keep working.
    {
        auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
        std::memcpy(block + 256, add_code, sizeof(add_code));
        const std::int32_t rel = 256 - 5;
        block[0] = 0xE9; std::memcpy(block + 1, &rel, 4);                       // jmp rel32 -> block+256
        block[16] = 0xFF; block[17] = 0x25; const std::int32_t zero = 0;       // jmp [rip+0] -> address below
        std::memcpy(block + 18, &zero, 4);
        std::uint8_t* real = block + 256; std::memcpy(block + 22, &real, 8);
        FlushInstructionCache(GetCurrentProcess(), block, 4096);
        EXPECT(follow_jumps(block) == block + 256, "follows jmp rel32");
        EXPECT(follow_jumps(block + 16) == block + 256, "follows jmp [rip]");
        auto via_jump = reinterpret_cast<AddFn>(block);
        EXPECT(via_jump(2, 3) == 5, "jump stub unhooked");
        EXPECT(g_hook.install(block, reinterpret_cast<void*>(&detour)), g_hook.error().c_str());
        EXPECT(via_jump(2, 3) == 50, "hooked through the other tool's jump");
        EXPECT(block[0] == 0xE9, "the other tool's jump is left alone");
        g_hook.remove();
        EXPECT(via_jump(2, 3) == 5, "restored behind the jump");
    }

    // A short conditional jump in the displaced bytes (FSR 4's ffxDispatch starts like this):
    // int f(void*, void* p) { if (!p) return 2; return 1; }  =  test rdx,rdx ; je +6 ; mov eax,1 ; ret ; mov eax,2 ; ret
    {
        const std::uint8_t cond[] = {0x48, 0x85, 0xD2, 0x74, 0x06, 0xB8, 1, 0, 0, 0, 0xC3, 0xB8, 2, 0, 0, 0, 0xC3};
        using CondFn = int (*)(void*, void*);
        auto* cfn = make_function(cond, sizeof(cond));
        auto f = reinterpret_cast<CondFn>(cfn);
        static InlineHook cond_hook;
        struct CondDetour { static int call(void* a, void* b) { return reinterpret_cast<CondFn>(cond_hook.original())(a, b) * 10; } };
        int dummy = 0;
        EXPECT(f(nullptr, &dummy) == 1 && f(nullptr, nullptr) == 2, "conditional function unhooked");
        EXPECT(cond_hook.install(cfn, reinterpret_cast<void*>(&CondDetour::call)), cond_hook.error().c_str());
        EXPECT(f(nullptr, &dummy) == 10, "hooked, jump not taken");
        EXPECT(f(nullptr, nullptr) == 20, "hooked, displaced short jump taken to its real target");
        cond_hook.remove();
        EXPECT(f(nullptr, nullptr) == 2, "restored");
    }

    const std::uint8_t call_first[] = {0xE8, 0, 0, 0, 0, 0x90};
    EXPECT(relocatable_prologue_length(call_first) == 0, "branches must be refused");

    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("hook tests passed\n");
    return 0;
}
