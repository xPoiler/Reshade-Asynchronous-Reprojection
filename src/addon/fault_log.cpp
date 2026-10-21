// Fault recorder: a game that installs its own crash handler (id Tech's DOOM Eternal) closes without a word when something
// goes wrong - no Windows error report, no dump, nothing in ReShade's log. What ends the process goes to
// FrameWarp\logs\crash.txt, with the call stack (module + offset per frame - the add-on's resolve with
// dev/tools/resolve_crash.py against the build folder's FrameWarp.pdb):
//  - faults (access violations, illegal instructions, stack overruns...), seen first by a vectored handler, then passed on
//    to whoever handles them, as before - at most 8 per run;
//  - every other exception raised (C++ exceptions, the driver's own), one short line each - at most 32: what was going on;
//  - abort, std::terminate, the C runtime's invalid-parameter and pure-call errors;
//  - the process exiting on its own (DllMain's process detach while the process ends): who called the exit, from where.
#include "addon/fault_log.hpp"

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace fw {
namespace {
wchar_t g_path[MAX_PATH] = {};
std::atomic<int> g_faults{0}, g_others{0};

bool fatal(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION: case EXCEPTION_ILLEGAL_INSTRUCTION: case EXCEPTION_PRIV_INSTRUCTION:
        case EXCEPTION_STACK_OVERFLOW: case EXCEPTION_INT_DIVIDE_BY_ZERO: case EXCEPTION_IN_PAGE_ERROR:
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: case 0xC0000409u /* stack buffer overrun / fail fast */: case 0xC0000374u /* heap corruption */:
            return true;
        default:
            return false;
    }
}

// "module+0xoffset" for an address (the module's file name only)
void where(DWORD64 address, char* out, size_t size) {
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &m) || !m) {
        std::snprintf(out, size, "0x%016llx", static_cast<unsigned long long>(address));
        return;
    }
    wchar_t name[MAX_PATH] = {};
    GetModuleFileNameW(m, name, MAX_PATH);
    const wchar_t* base = name;
    for (const wchar_t* p = name; *p; ++p) if (*p == L'\\' || *p == L'/') base = p + 1;
    std::snprintf(out, size, "%ls+0x%llx", base, static_cast<unsigned long long>(address - reinterpret_cast<DWORD64>(m)));
}

FILE* open_log() {
    FILE* f = nullptr;
    if (!g_path[0] || _wfopen_s(&f, g_path, L"a") != 0) return nullptr;
    return f;
}

void stamp(FILE* f) {
    SYSTEMTIME t{};
    GetLocalTime(&t);
    std::fprintf(f, "%04u-%02u-%02u %02u:%02u:%02u.%03u ", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
}

// The stack, unwound from a context (x64 unwind data; a frame without any: a leaf, its return address on top)
void stack(FILE* f, CONTEXT c) {
    char at[MAX_PATH + 32];
    for (int i = 0; i < 48 && c.Rip; ++i) {
        where(c.Rip, at, sizeof(at));
        std::fprintf(f, "  %2d %s\n", i, at);
        DWORD64 image = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(c.Rip, &image, nullptr);
        if (fn) {
            void* handler = nullptr;
            DWORD64 frame = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, c.Rip, fn, &c, &handler, &frame, nullptr);
        } else {
            if (!c.Rsp || IsBadReadPtr(reinterpret_cast<void*>(c.Rsp), 8)) break;
            c.Rip = *reinterpret_cast<DWORD64*>(c.Rsp);
            c.Rsp += 8;
        }
    }
}

// The current thread's own stack, from here
void here(FILE* f) {
    CONTEXT c{};
    RtlCaptureContext(&c);
    stack(f, c);
}

void note(const char* what) {
    if (FILE* f = open_log()) {
        stamp(f);
        std::fprintf(f, "%s, thread %lu\n", what, GetCurrentThreadId());
        here(f);
        std::fclose(f);
    }
}

void modules(FILE* f);
// Readable text near the fault: what the registers point to, and the thread's stack (64 KB up from its top) - a game's own
// fatal-error routine (id Tech: a breakpoint on the main thread) holds its message there though it never shows it
bool readable(const void* p, size_t n) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(p, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return false;
    return reinterpret_cast<const char*>(p) + n <= reinterpret_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize;
}
void strings_in(FILE* f, const char* label, const unsigned char* p, size_t n, int& budget) {
    size_t start = 0, run = 0;
    for (size_t i = 0; i <= n && budget > 0; ++i) {
        const bool text = i < n && p[i] >= 32 && p[i] < 127;
        if (text) { if (!run) start = i; ++run; continue; }
        if (run >= 10) {
            std::fprintf(f, "  text %s+0x%zx: %.*s\n", label, start, static_cast<int>(std::min<size_t>(run, 300)), reinterpret_cast<const char*>(p + start));
            --budget;
        }
        run = 0;
    }
}
void texts(FILE* f, const CONTEXT& c) {
    // (the registers, and the stack's first 256 bytes: a failed call's result and its arguments - DOOM Eternal breaks on a
    // Vulkan call's error code, left in rax)
    std::fprintf(f, "  rax %016llx rbx %016llx rcx %016llx rdx %016llx rsi %016llx rdi %016llx\n  r8 %016llx r9 %016llx r12 %016llx r13 %016llx r14 %016llx r15 %016llx rbp %016llx\n",
                 c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.R8, c.R9, c.R12, c.R13, c.R14, c.R15, c.Rbp);
    for (size_t off = 0; off < 256; off += 32) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(c.Rsp) + off;
        if (!readable(p, 32)) break;
        const auto* q = reinterpret_cast<const unsigned long long*>(p);
        std::fprintf(f, "  rsp+%02zx: %016llx %016llx %016llx %016llx\n", off, q[0], q[1], q[2], q[3]);
    }
    int budget = 60;
    const DWORD64 regs[] = {c.Rcx, c.Rdx, c.R8, c.R9, c.Rax, c.Rbx, c.Rsi, c.Rdi};
    static const char* const kNames[] = {"rcx", "rdx", "r8", "r9", "rax", "rbx", "rsi", "rdi"};
    for (int i = 0; i < 8; ++i)
        if (regs[i] > 0x10000 && readable(reinterpret_cast<void*>(regs[i]), 512)) strings_in(f, kNames[i], reinterpret_cast<const unsigned char*>(regs[i]), 512, budget);
    // (the stack, page by page while readable)
    const unsigned char* sp = reinterpret_cast<const unsigned char*>(c.Rsp);
    for (size_t off = 0; off < 65536 && budget > 0; off += 4096)
        if (readable(sp + off, 4096)) strings_in(f, "stack", sp + off, 4096, budget);
        else break;
    // (what the stack's first 8 KB point to: a message formatted into a heap buffer)
    int pointed = 60;
    for (size_t off = 0; off + 8 <= 8192 && pointed > 0; off += 8) {
        if (!readable(sp + off, 8)) break;
        const DWORD64 v = *reinterpret_cast<const DWORD64*>(sp + off);
        if (v <= 0x10000 || v >= 0x7fffffffffffull || !readable(reinterpret_cast<void*>(v), 256)) continue;
        char label[32];
        std::snprintf(label, sizeof(label), "[stack+0x%zx]", off);
        strings_in(f, label, reinterpret_cast<const unsigned char*>(v), 256, pointed);
    }
}
// Every other thread's stack (each suspended while it is read): a game's watchdog stops it with a breakpoint when its render
// or present thread hangs - the hung thread, and what it waits on, are among these
void all_threads(FILE* f) {
    using SnapFn = HANDLE(WINAPI*)(DWORD, DWORD);
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    int listed = 0;
    for (BOOL ok = Thread32First(snap, &te); ok && listed < 96; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
        HANDLE t = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
        if (!t) continue;
        if (SuspendThread(t) != static_cast<DWORD>(-1)) {
            CONTEXT c{};
            c.ContextFlags = CONTEXT_FULL;
            const bool got = GetThreadContext(t, &c) != FALSE;
            ResumeThread(t);  // (before writing: a suspended thread may hold the loader's or the heap's lock the writing needs)
            if (got) {
                std::fprintf(f, " thread %lu:\n", te.th32ThreadID);
                stack(f, c);
                ++listed;
            }
        }
        CloseHandle(t);
    }
    CloseHandle(snap);
}
LONG CALLBACK on_exception(EXCEPTION_POINTERS* e) {
    if (!e || !e->ExceptionRecord || !e->ContextRecord) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD& r = *e->ExceptionRecord;
    // (debugger messages and thread names: not worth a line)
    if (r.ExceptionCode == 0x40010006u || r.ExceptionCode == 0x4001000Au || r.ExceptionCode == 0x406D1388u) return EXCEPTION_CONTINUE_SEARCH;
    const bool is_fatal = fatal(r.ExceptionCode);
    if ((is_fatal ? g_faults : g_others).fetch_add(1) >= (is_fatal ? 8 : 32)) return EXCEPTION_CONTINUE_SEARCH;
    FILE* f = open_log();
    if (!f) return EXCEPTION_CONTINUE_SEARCH;
    char at[MAX_PATH + 32];
    where(reinterpret_cast<DWORD64>(r.ExceptionAddress), at, sizeof(at));
    stamp(f);
    std::fprintf(f, "%s 0x%08lx at %s, thread %lu", is_fatal ? "fault" : "exception", r.ExceptionCode, at, GetCurrentThreadId());
    if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2)
        std::fprintf(f, " (%s 0x%016llx)", r.ExceptionInformation[0] == 0 ? "reading" : r.ExceptionInformation[0] == 1 ? "writing" : "executing",
                     static_cast<unsigned long long>(r.ExceptionInformation[1]));
    std::fprintf(f, "\n");
    // (a breakpoint too: id Tech's engine stops itself with one when a check fails - DOOM Eternal - its stack shows which call)
    if (is_fatal || r.ExceptionCode == EXCEPTION_BREAKPOINT) stack(f, *e->ContextRecord);
    if (r.ExceptionCode == EXCEPTION_BREAKPOINT) { texts(f, *e->ContextRecord); modules(f); all_threads(f); }
    std::fclose(f);
    return EXCEPTION_CONTINUE_SEARCH;
}

// What else is in the process - overlays, capture tools, other Vulkan layers - and the Vulkan loader's variables (once at
// load, again when the program stops itself)
void modules(FILE* f) {
    static const char* const kVars[] = {"DISABLE_VK_LAYER_reshade_1", "DISABLE_VK_LAYER_reshade_2", "VK_INSTANCE_LAYERS", "VK_LOADER_LAYERS_ENABLE",
                                        "VK_LOADER_LAYERS_DISABLE", "VK_LAYER_PATH", "VK_ADD_LAYER_PATH", "VK_ICD_FILENAMES"};
    for (const char* v : kVars) {
        char value[512] = {};
        if (GetEnvironmentVariableA(v, value, sizeof(value))) std::fprintf(f, "  env %s=%s\n", v, value);
    }
    HMODULE list[1024];
    DWORD bytes = 0;
    using EnumFn = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
    const auto enum_modules = reinterpret_cast<EnumFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32EnumProcessModules"));
    if (!enum_modules || !enum_modules(GetCurrentProcess(), list, sizeof(list), &bytes)) return;
    for (DWORD i = 0; i < bytes / sizeof(HMODULE) && i < 1024; ++i) {
        wchar_t name[MAX_PATH] = {};
        GetModuleFileNameW(list[i], name, MAX_PATH);
        // (Windows' own: not listed)
        if (_wcsnicmp(name, L"C:\\WINDOWS\\System32\\", 20) == 0 || _wcsnicmp(name, L"C:\\WINDOWS\\WinSxS", 16) == 0) continue;
        std::fprintf(f, "  module %ls\n", name);
    }
}

void on_abort(int) { note("abort"); }
void on_terminate() { note("std::terminate"); std::abort(); }
void on_invalid_parameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) { note("invalid parameter (C runtime)"); }
void on_purecall() { note("pure virtual call"); }
}  // namespace

void install_fault_log(const std::wstring& logs_dir) {
    CreateDirectoryW(logs_dir.c_str(), nullptr);
    const std::wstring path = logs_dir + L"\\crash.txt";
    wcsncpy_s(g_path, path.c_str(), _TRUNCATE);
    if (FILE* f = open_log()) {
        stamp(f);
        wchar_t exe[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::fprintf(f, "add-on loaded in %ls\n", exe);
        modules(f);
        std::fclose(f);
    }
    AddVectoredExceptionHandler(1, on_exception);
    // (the add-on's own C runtime - static: these catch what goes wrong in it; the game's own runtime has its own)
    std::signal(SIGABRT, on_abort);
    std::set_terminate(on_terminate);
    _set_invalid_parameter_handler(on_invalid_parameter);
    _set_purecall_handler(on_purecall);
}

// The process exiting while the add-on is loaded (lpReserved non-null: the process ends, not an unload): who called it
void fault_log_process_exit() { note("process exiting (ExitProcess or the end of main)"); }
}  // namespace fw

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_DETACH && reserved) fw::fault_log_process_exit();
    return TRUE;
}
