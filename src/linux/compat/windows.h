// v7 - narrow source compatibility for the Linux build of release tools
// License: GPL v3, see LICENSE
#pragma once

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define __int64 long long
#define __cdecl
#define WINAPI
#define MAX_PATH 4096
#define _snprintf snprintf
#define _stricmp strcasecmp
#define _strnicmp strncasecmp
#define _fseeki64 fseeko
#define _ftelli64 ftello
#define _fileno fileno
#define _unlink unlink
#define ExitProcess exit
#define sscanf_s sscanf

typedef void *HANDLE;
typedef void *HMODULE;
typedef void *FARPROC;
typedef uint32_t DWORD;
typedef int BOOL;
typedef unsigned long long ULONGLONG;
typedef struct { long long QuadPart; } LARGE_INTEGER;

#define TRUE 1
#define FALSE 0
#define INVALID_HANDLE_VALUE ((HANDLE)-1)
#define INVALID_FILE_ATTRIBUTES UINT32_MAX
#define GENERIC_READ UINT32_C(0x80000000)
#define FILE_SHARE_READ 1
#define OPEN_EXISTING 3
#define FILE_ATTRIBUTE_NORMAL 0x80

/* The engine reads standard input through file descriptor 0, so a host that
   redirects stdin has to install the new descriptor on 0 physically. The
   descriptor 0 pointed at before the first redirect is kept in a private copy
   so the original stream can be restored afterwards. */
#define STD_INPUT_HANDLE ((DWORD)-10)
#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define STD_ERROR_HANDLE ((DWORD)-12)

struct std_handle_state
{
	int current[3];
	int original[3];
};

static inline struct std_handle_state *std_handle_slots(void)
{
	static struct std_handle_state state = { { 0, 1, 2 }, { -1, -1, -1 } };
	return &state;
}

static inline void Sleep(unsigned milliseconds)
{
    struct timespec delay = { (time_t)(milliseconds / 1000), (long)(milliseconds % 1000) * 1000000 };
    while (nanosleep(&delay, &delay) && errno == EINTR)
        ;
}

static inline BOOL QueryPerformanceFrequency(LARGE_INTEGER *frequency)
{
    frequency->QuadPart = 1000000000;
    return TRUE;
}

static inline BOOL QueryPerformanceCounter(LARGE_INTEGER *counter)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now))
        return FALSE;
    counter->QuadPart = now.tv_sec * 1000000000LL + now.tv_nsec;
    return TRUE;
}

static inline BOOL SetEnvironmentVariableA(const char *name, const char *value)
{
    return (value ? setenv(name, value, 1) : unsetenv(name)) == 0;
}

static inline unsigned long GetLastError(void)
{
    return errno;
}

static inline DWORD GetFileAttributesA(const char *path)
{
    struct stat info;
    return stat(path, &info) ? INVALID_FILE_ATTRIBUTES : FILE_ATTRIBUTE_NORMAL;
}

static inline BOOL DeleteFileA(const char *path)
{
    return unlink(path) == 0;
}

static inline DWORD GetTempPathA(DWORD capacity, char *buffer)
{
    const char *temp = getenv("TMPDIR");
    if (!temp || !*temp)
        temp = "/tmp";
    int length = snprintf(buffer, capacity, "%s%s", temp, temp[strlen(temp) - 1] == '/' ? "" : "/");
    return length < 0 ? 0 : (DWORD)length + ((DWORD)length >= capacity);
}

static inline DWORD GetModuleFileNameA(HMODULE module, char *buffer, DWORD capacity)
{
    (void)module;
    if (!capacity)
        return 0;
    ssize_t length = readlink("/proc/self/exe", buffer, capacity - 1);
    if (length < 0)
        return 0;
    buffer[length] = 0;
    return length;
}

static inline HANDLE CreateFileA(const char *path, DWORD access, DWORD share, void *security,
                                 DWORD disposition, DWORD attributes, HANDLE template_file)
{
    (void)access; (void)share; (void)security; (void)disposition;
    (void)attributes; (void)template_file;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    return fd < 0 ? INVALID_HANDLE_VALUE : (HANDLE)((uintptr_t)fd + 1);
}

static inline BOOL CloseHandle(HANDLE handle)
{
    return close((int)((uintptr_t)handle - 1)) == 0;
}

static inline FARPROC GetProcAddress(HMODULE module, const char *name)
{
    return dlsym(module, name);
}

static inline HMODULE GetModuleHandleA(const char *name)
{
    return dlopen(name, RTLD_NOW);
}

static inline void *_aligned_malloc(size_t bytes, size_t alignment)
{
    void *result;
    return posix_memalign(&result, alignment, bytes) ? NULL : result;
}

#define _aligned_free free

static inline BOOL CreatePipe(HANDLE *read_handle, HANDLE *write_handle, void *attributes, DWORD size)
{
    int descriptors[2];
    (void)attributes;
    (void)size;
    if (pipe(descriptors))
        return FALSE;
    *read_handle = (HANDLE)(uintptr_t)(descriptors[0] + 1);
    *write_handle = (HANDLE)(uintptr_t)(descriptors[1] + 1);
    return TRUE;
}

static inline BOOL WriteFile(HANDLE handle, const void *data, DWORD size, DWORD *written, void *overlapped)
{
    (void)overlapped;
    int descriptor = (int)((uintptr_t)handle - 1);
    if (descriptor < 0)
        return FALSE;
    const char *cursor = (const char *)data;
    DWORD total = 0;
    while (total < size)
    {
        ssize_t chunk = write(descriptor, cursor + total, size - total);
        if (chunk < 0)
        {
            if (errno == EINTR)
                continue;
            break;
        }
        total += (DWORD)chunk;
    }
    if (written)
        *written = total;
    return total == size;
}

static inline HANDLE GetStdHandle(DWORD which)
{
    int slot = -((int)which) - 10;
    if (slot < 0 || slot > 2)
        return INVALID_HANDLE_VALUE;
    return (HANDLE)(uintptr_t)(std_handle_slots()->current[slot] + 1);
}

static inline BOOL SetStdHandle(DWORD which, HANDLE handle)
{
    int slot = -((int)which) - 10;
    if (slot < 0 || slot > 2)
        return FALSE;
    int target = (int)((uintptr_t)handle - 1);
    if (target < 0)
        return FALSE;
    struct std_handle_state *state = std_handle_slots();
    if (state->original[slot] < 0)
    {
        int copy = fcntl(slot, F_DUPFD_CLOEXEC, 10);
        if (copy < 0)
            return FALSE;
        state->original[slot] = copy;
    }
    if (target == slot)
        target = state->original[slot];
    if (dup2(target, slot) < 0)
        return FALSE;
    state->current[slot] = state->original[slot];
    return TRUE;
}

/* x87 control word read: the Windows build compares the word before and after
   the workload to catch a leaked precision-control setting. Linux keeps the x87
   unit, so the same word is read straight out of the FPU state. */
static inline unsigned int _control87(unsigned int mask, unsigned int new_value)
{
    (void)mask;
    (void)new_value;
    unsigned short control_word = 0;
    __asm__ __volatile__("fnstcw %0" : "=m"(control_word));
    return control_word;
}
