// v7 - Linux system boundary for the assembly engine
// License: GPL v3, see LICENSE

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

// Some assembly import sites use 8-byte alignment. Realign before entering libc.
#define ABI __attribute__((ms_abi, force_align_arg_pointer))
#define IMPORT(name) __attribute__((visibility("hidden"))) \
    __typeof__(&name) const __imp_##name = name
#define INVALID_HANDLE ((void *)-1)
#define WAIT_FAILED UINT32_MAX
#define WAIT_TIMEOUT 258u


enum handle_kind { WORKER, CALLER, PROCESS };
struct handle
{
    enum handle_kind kind;
    pthread_t thread;
    pid_t id;
    int priority;
    cpu_set_t saved_affinity;
    uint64_t saved_mask;
    unsigned (ABI *start)(void *);
    void *argument;
};

static void *fd_handle(int fd)
{
    return fd < 0 ? INVALID_HANDLE : (void *)((uintptr_t)fd + 1);
}

static int handle_fd(void *handle)
{
    uintptr_t value = (uintptr_t)handle;
    return value && value <= INT_MAX ? (int)value - 1 : -1;
}

static int ABI CloseHandle(void *opaque)
{
    int fd = handle_fd(opaque);
    if (fd >= 0)
        return close(fd) == 0;
    if (!opaque || opaque == INVALID_HANDLE)
        return 0;
    struct handle *handle = opaque;
    // The worker's final assembly epilogue must finish before dlclose unmaps it.
    if (handle->kind == WORKER && pthread_join(handle->thread, NULL))
        abort();
    // Reap a child that has already exited so closing an unwaited process
    // handle never leaves a zombie. A still-running child stays the caller's
    // responsibility through WaitForSingleObject, matching the Windows object.
    if (handle->kind == PROCESS)
    {
        int status;
        pid_t result;
        do
            result = waitpid(handle->id, &status, WNOHANG);
        while (result < 0 && errno == EINTR);
    }
    free(handle);
    return 1;
}

static unsigned ABI WaitForSingleObject(void *opaque, unsigned timeout)
{
    struct handle *handle = opaque;
    if (!handle || opaque == INVALID_HANDLE || handle_fd(opaque) >= 0)
        return WAIT_FAILED;
    if (handle->kind == PROCESS)
    {
        int status;
        pid_t result;
        do
            result = waitpid(handle->id, &status, timeout == 0 ? WNOHANG : 0);
        while (result < 0 && errno == EINTR);
        return result > 0 ? 0 : result == 0 ? WAIT_TIMEOUT : WAIT_FAILED;
    }
    if (handle->kind == CALLER && timeout == 0)
    {
        if (syscall(SYS_tgkill, getpid(), handle->id, 0) == 0)
            return WAIT_TIMEOUT;
        return errno == ESRCH ? 0 : WAIT_FAILED;
    }
    return WAIT_FAILED;
}

static void *worker_start(void *opaque)
{
    struct handle *handle = opaque;
    handle->start(handle->argument);
    return NULL;
}

static void *ABI CreateThread(void *security, size_t stack, unsigned (ABI *start)(void *),
                              void *argument, unsigned flags, unsigned *id)
{
    struct handle *handle = calloc(1, sizeof(*handle));
    if (!handle)
        return NULL;
    handle->kind = WORKER;
    handle->start = start;
    handle->argument = argument;
    if (pthread_create(&handle->thread, NULL, worker_start, handle))
    {
        free(handle);
        return NULL;
    }
    return handle;
}

static int ABI DuplicateHandle(void *src, void *pseudo, void *dst, void **output,
                               unsigned access, int inherit, unsigned options)
{
    if (pseudo != (void *)-2)
        return 0;
    struct handle *handle = calloc(1, sizeof(*handle));
    if (!handle)
        return 0;
    handle->kind = CALLER;
    handle->id = gettid();
    *output = handle;
    return 1;
}

// Use dense scheduler bits for the CPUs allowed by the host's affinity mask.
static int cpu_ids[64];
static unsigned cpu_count;
static uint64_t core_masks[64];
static unsigned core_count;

static int topology_id(int cpu, const char *field, int *value)
{
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, field);
    FILE *file = fopen(path, "r");
    if (!file)
        return 0;
    int ok = fscanf(file, "%d", value) == 1;
    fclose(file);
    return ok;
}

// Containers can expose x86 CPU IDs through procfs while omitting sysfs.
static int topology_from_cpuinfo(int *packages, int *cores)
{
    FILE *file = fopen("/proc/cpuinfo", "r");
    if (!file)
        return 0;

    for (unsigned i = 0; i < cpu_count; ++i)
        packages[i] = cores[i] = -1;

    char *line = NULL;
    size_t capacity = 0;
    unsigned slot = cpu_count;
    while (getline(&line, &capacity, file) >= 0)
    {
        int cpu;
        if (sscanf(line, "processor : %d", &cpu) == 1)
        {
            slot = 0;
            while (slot < cpu_count && cpu_ids[slot] != cpu)
                ++slot;
        }
        else if (slot < cpu_count)
        {
            if (sscanf(line, "physical id : %d", &packages[slot]) != 1)
                sscanf(line, "core id : %d", &cores[slot]);
        }
    }
    free(line);
    fclose(file);
    for (unsigned i = 0; i < cpu_count; ++i)
        if (packages[i] < 0 || cores[i] < 0)
            return 0;
    return 1;
}

static int load_topology(void)
{
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
        return 0;

    int packages[64], cores[64];
    cpu_count = core_count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && cpu_count < 64; ++cpu)
        if (CPU_ISSET(cpu, &allowed))
            cpu_ids[cpu_count++] = cpu;

    for (unsigned i = 0; i < cpu_count; ++i)
    {
        if (!topology_id(cpu_ids[i], "physical_package_id", &packages[i]) ||
            !topology_id(cpu_ids[i], "core_id", &cores[i]))
        {
            if (!topology_from_cpuinfo(packages, cores))
            {
                errno = ENOTSUP;
                return 0;
            }
            break;
        }
    }

    int core_packages[64], core_ids[64];
    for (unsigned cpu = 0; cpu < cpu_count; ++cpu)
    {
        unsigned index = 0;
        while (index < core_count &&
               (core_packages[index] != packages[cpu] || core_ids[index] != cores[cpu]))
            ++index;
        if (index == core_count)
        {
            core_packages[index] = packages[cpu];
            core_ids[index] = cores[cpu];
            core_masks[index] = 0;
            ++core_count;
        }
        core_masks[index] |= UINT64_C(1) << cpu;
    }
    return cpu_count != 0;
}

struct processor_core_record
{
    uint32_t relationship, size;
    unsigned char reserved[22];
    uint16_t group_count;
    uint64_t mask;
    uint16_t group;
    unsigned char padding[6];
};
_Static_assert(sizeof(struct processor_core_record) == 48, "assembly topology layout");

static int ABI GetLogicalProcessorInformationEx(unsigned relation,
                                                 struct processor_core_record *buffer,
                                                 unsigned *length)
{
    if (!buffer && !load_topology())
        return 0;
    unsigned required = core_count * sizeof(*buffer);
    if (!buffer || *length < required)
    {
        *length = required;
        errno = ENOBUFS;
        return 0;
    }
    for (unsigned i = 0; i < core_count; ++i)
        buffer[i] = (struct processor_core_record){
            .size = sizeof(*buffer), .group_count = 1, .mask = core_masks[i]
        };
    *length = required;
    return 1;
}

static int ABI GetProcessAffinityMask(void *process, uint64_t *allowed, uint64_t *system)
{
    if (!cpu_count)
        return 0;
    *allowed = *system = UINT64_MAX >> (64 - cpu_count);
    return 1;
}

static unsigned ABI GetCurrentProcessorNumber(void)
{
    int cpu = sched_getcpu();
    for (unsigned i = 0; i < cpu_count; ++i)
        if (cpu_ids[i] == cpu)
            return i;
    return cpu < 0 ? UINT32_MAX : (unsigned)cpu;
}

static uint64_t ABI SetThreadAffinityMask(struct handle *handle, uint64_t mask)
{
    if (!handle || !mask)
        return 0;
    cpu_set_t current, requested;
    int caller = handle->kind == CALLER;
    int error = caller ? sched_getaffinity(handle->id, sizeof(current), &current) :
                         pthread_getaffinity_np(handle->thread, sizeof(current), &current);
    if (error)
        return 0;
    uint64_t previous = 0;
    CPU_ZERO(&requested);
    for (unsigned i = 0; i < cpu_count; ++i)
    {
        if (CPU_ISSET(cpu_ids[i], &current))
            previous |= UINT64_C(1) << i;
        if (mask & (UINT64_C(1) << i))
            CPU_SET(cpu_ids[i], &requested);
    }
    if (caller)
    {
        if (handle->saved_mask && mask == handle->saved_mask)
            requested = handle->saved_affinity;
        else
        {
            handle->saved_mask = previous ? previous : 1;
            handle->saved_affinity = current;
        }
        error = sched_setaffinity(handle->id, sizeof(requested), &requested);
    }
    else
        error = pthread_setaffinity_np(handle->thread, sizeof(requested), &requested);
    return error ? 0 : previous ? previous : 1;
}

// Preserve the engine's priority setting without requesting Linux real-time scheduling.
static int ABI SetThreadPriority(struct handle *handle, int priority)
{
    if (!handle)
        return 0;
    handle->priority = priority;
    return 1;
}

static int ABI GetThreadPriority(struct handle *handle)
{
    return handle ? handle->priority : INT_MAX;
}

static void ABI Sleep(unsigned milliseconds)
{
    struct timespec delay = { milliseconds / 1000, (milliseconds % 1000) * 1000000L };
    while (nanosleep(&delay, &delay) && errno == EINTR)
        ;
}

static int ABI WaitOnAddress(uint32_t *address, const uint32_t *expected,
                             size_t size, unsigned milliseconds)
{
    if (size != sizeof(*address))
        return 0;
    struct timespec delay = { milliseconds / 1000, (milliseconds % 1000) * 1000000L };
    long result = syscall(SYS_futex, address, FUTEX_WAIT_PRIVATE, *expected,
                          milliseconds == UINT32_MAX ? NULL : &delay, NULL, 0);
    return result == 0 || errno == EAGAIN || errno == EINTR;
}

static void ABI WakeByAddressAll(uint32_t *address)
{
    syscall(SYS_futex, address, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
}

static unsigned ABI GetLastError(void)
{
    return errno == ENOBUFS ? 122 : (unsigned)errno;
}

static unsigned ABI GetEnvironmentVariableA(const char *name, char *buffer, unsigned size)
{
    const char *value = getenv(name);
    if (!value)
        return 0;
    size_t length = strlen(value);
    if (length >= size)
        return length + 1;
    memcpy(buffer, value, length + 1);
    return length;
}

static void *ABI CreateFileA(const char *path, unsigned access, unsigned share, void *security,
                             unsigned disposition, unsigned attributes, void *template)
{
    int flags = (access & 0x40000000) ?
                ((access & 0x80000000) ? O_RDWR : O_WRONLY) : O_RDONLY;
    if (disposition == 2)
        flags |= O_CREAT | O_TRUNC;
    else if (disposition == 4)
        flags |= O_CREAT;
    else if (disposition != 3)
    {
        errno = EINVAL;
        return INVALID_HANDLE;
    }
    return fd_handle(open(path, flags | O_CLOEXEC, 0666));
}

static int ABI ReadFile(void *handle, void *output, unsigned size, unsigned *received, void *overlapped)
{
    ssize_t length;
    do
        length = read(handle_fd(handle), output, size);
    while (length < 0 && errno == EINTR);
    if (received)
        *received = length < 0 ? 0 : (unsigned)length;
    return length >= 0;
}

static int ABI WriteFile(void *handle, const void *input, unsigned size,
                         unsigned *written, void *overlapped)
{
    unsigned completed = 0;
    while (completed < size)
    {
        ssize_t length = write(handle_fd(handle), (const char *)input + completed, size - completed);
        if (length < 0 && errno == EINTR)
            continue;
        if (length <= 0)
            break;
        completed += length;
    }
    if (written)
        *written = completed;
    return completed == size;
}

static unsigned ABI GetFileSize(void *handle, unsigned *high)
{
    struct stat info;
    if (fstat(handle_fd(handle), &info))
        return UINT32_MAX;
    if (high)
        *high = (uint64_t)info.st_size >> 32;
    return info.st_size;
}

static int ABI SetFilePointerEx(void *handle, int64_t offset, int64_t *result, unsigned origin)
{
    off_t position = lseek(handle_fd(handle), offset, origin);
    if (position < 0)
        return 0;
    if (result)
        *result = position;
    return 1;
}

static int ABI SetEndOfFile(void *handle)
{
    int fd = handle_fd(handle);
    off_t position = lseek(fd, 0, SEEK_CUR);
    return position >= 0 && ftruncate(fd, position) == 0;
}

static int ABI DeleteFileA(const char *path)
{
    return unlink(path) == 0;
}

static int ABI MoveFileExA(const char *source, const char *target, unsigned flags)
{
    if (flags & 8)
    {
        int fd = open(source, O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return 0;
        int result = fsync(fd);
        int saved = errno;
        close(fd);
        errno = saved;
        if (result)
            return 0;
    }
    return rename(source, target) == 0;
}

static size_t page_size(void)
{
    return (size_t)sysconf(_SC_PAGESIZE);
}

static int protection(unsigned flags)
{
    switch (flags)
    {
        case 1: return PROT_NONE;
        case 2: return PROT_READ;
        case 4: return PROT_READ | PROT_WRITE;
        case 0x20: return PROT_READ | PROT_EXEC;
        case 0x40: return PROT_READ | PROT_WRITE | PROT_EXEC;
        default: return -1;
    }
}

static void *ABI VirtualAlloc(void *address, size_t size, unsigned type, unsigned flags)
{
    size_t page = page_size();
    int mode = protection(flags);
    if (address || !size || size > SIZE_MAX - page || mode != (PROT_READ | PROT_WRITE))
        return NULL;
    void *base = mmap(NULL, page + size, mode, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        return NULL;
    *(size_t *)base = page + size;
    return (char *)base + page;
}

static int ABI VirtualFree(void *address, size_t size, unsigned type)
{
    if (!address || size || type != 0x8000)
        return 0;
    void *base = (char *)address - page_size();
    return munmap(base, *(size_t *)base) == 0;
}

static int ABI VirtualProtect(void *address, size_t length, unsigned flags, unsigned *old)
{
    size_t page = page_size();
    size_t offset = (uintptr_t)address % page;
    int mode = protection(flags);
    if (mode < 0 || length > SIZE_MAX - offset ||
        mprotect((char *)address - offset, length + offset, mode))
        return 0;
    // These call sites transition executable text or an initially RW guard page.
    if (old)
        *old = flags == 0x40 ? 0x20 : 4;
    return 1;
}

static int narrow_path(const uint16_t *input, char *output, size_t size)
{
    size_t i = 0;
    while (input[i] && i + 1 < size && input[i] <= UCHAR_MAX)
    {
        output[i] = (char)input[i];
        ++i;
    }
    output[i] = 0;
    return input[i] == 0;
}

static void widen_path(uint16_t *output, const char *input)
{
    do
        *output++ = (unsigned char)*input;
    while (*input++);
}

static unsigned ABI GetTempPathW(unsigned size, uint16_t *output)
{
    const char *directory = getenv("TMPDIR");
    if (!directory || !*directory)
        directory = "/tmp";
    size_t length = strlen(directory);
    int slash = directory[length - 1] != '/';
    if (length + slash >= size)
        return length + slash + 1;
    widen_path(output, directory);
    if (slash)
    {
        output[length++] = '/';
        output[length] = 0;
    }
    return length;
}

// The allocator requests the temporary file name and immediately opens it.
static __thread char pending_temp_path[260];
static __thread int pending_temp_fd = -1;

static unsigned ABI GetTempFileNameW(const uint16_t *directory, const uint16_t *prefix,
                                     unsigned unique, uint16_t *output)
{
    char path[260];
    if (!narrow_path(directory, path, sizeof(path)))
        return 0;
    size_t length = strlen(path);
    if (sizeof(path) - length < sizeof("v7-XXXXXX"))
        return 0;
    memcpy(path + length, "v7-XXXXXX", sizeof("v7-XXXXXX"));
    int fd = mkstemp(path);
    if (fd < 0)
        return 0;
    if (unlink(path))
    {
        close(fd);
        return 0;
    }
    if (pending_temp_fd >= 0)
        close(pending_temp_fd);
    pending_temp_fd = fd;
    strcpy(pending_temp_path, path);
    widen_path(output, path);
    return 1;
}

static void *ABI CreateFileW(const uint16_t *wide, unsigned access, unsigned share, void *security,
                             unsigned disposition, unsigned attributes, void *template)
{
    char path[260];
    if (!narrow_path(wide, path, sizeof(path)))
        return INVALID_HANDLE;
    if (attributes & 0x04000000)
    {
        if (pending_temp_fd < 0 || strcmp(path, pending_temp_path))
        {
            errno = ENOENT;
            return INVALID_HANDLE;
        }
        int fd = pending_temp_fd;
        pending_temp_fd = -1;
        return fd_handle(fd);
    }
    if (strcmp(path, "CONOUT$") == 0)
        strcpy(path, "/dev/tty");
    return CreateFileA(path, access, share, security, disposition, attributes, template);
}

static void *ABI CreateFileMappingW(void *file, void *security, unsigned flags,
                                    unsigned high, unsigned low, const uint16_t *name)
{
    int fd = fcntl(handle_fd(file), F_DUPFD_CLOEXEC, 3);
    return fd < 0 ? NULL : fd_handle(fd);
}

struct mapping
{
    void *address;
    size_t size;
    struct mapping *next;
};
static pthread_mutex_t mappings_lock = PTHREAD_MUTEX_INITIALIZER;
static struct mapping *mappings;

static void *ABI MapViewOfFile(void *map, unsigned access, unsigned high,
                               unsigned low, size_t size)
{
    int fd = handle_fd(map);
    off_t offset = (uint64_t)high << 32 | low;
    struct stat info;
    if (!size)
    {
        if (fstat(fd, &info) || offset < 0 || info.st_size <= offset)
            return NULL;
        size = info.st_size - offset;
    }
    struct mapping *entry = malloc(sizeof(*entry));
    if (!entry)
        return NULL;
    int mode = PROT_READ | ((access & 2) ? PROT_WRITE : 0);
    void *address = mmap(NULL, size, mode, MAP_SHARED, fd, offset);
    if (address == MAP_FAILED)
    {
        free(entry);
        return NULL;
    }
    pthread_mutex_lock(&mappings_lock);
    *entry = (struct mapping){ address, size, mappings };
    mappings = entry;
    pthread_mutex_unlock(&mappings_lock);
    return address;
}

static int ABI UnmapViewOfFile(void *address)
{
    pthread_mutex_lock(&mappings_lock);
    struct mapping **link = &mappings;
    while (*link && (*link)->address != address)
        link = &(*link)->next;
    struct mapping *entry = *link;
    if (entry)
        *link = entry->next;
    pthread_mutex_unlock(&mappings_lock);
    if (!entry)
        return 0;
    int success = munmap(address, entry->size) == 0;
    free(entry);
    return success;
}

static int ABI DeviceIoControl(void *handle, unsigned code, void *input, unsigned input_size,
                               void *output, unsigned output_size, unsigned *returned, void *overlapped)
{
    // Linux ftruncate already creates holes in the temporary backing file.
    if (returned)
        *returned = 0;
    return code == 0x900c4;
}

static void *ABI GetStdHandle(int which)
{
    int fd = -which - 10;
    return fd >= 0 && fd <= 2 ? fd_handle(fd) : INVALID_HANDLE;
}

static unsigned ABI GetFileType(void *handle)
{
    struct stat info;
    if (fstat(handle_fd(handle), &info))
        return 0;
    if (S_ISREG(info.st_mode))
        return 1;
    if (S_ISCHR(info.st_mode))
        return 2;
    if (S_ISFIFO(info.st_mode) || S_ISSOCK(info.st_mode))
        return 3;
    errno = ENOTSUP;
    return 0;
}

static int ABI PeekNamedPipe(void *handle, void *buffer, unsigned length,
                             unsigned *received, unsigned *available,
                             unsigned *remaining, void *overlapped)
{
    int count;
    if (ioctl(handle_fd(handle), FIONREAD, &count))
        return 0;
    *available = count;
    return 1;
}

static int ABI GetConsoleScreenBufferInfo(void *handle, void *buffer)
{
    struct winsize window;
    if (ioctl(handle_fd(handle), TIOCGWINSZ, &window) || !window.ws_col)
        return 0;
    memset(buffer, 0, 22);
    uint16_t right = window.ws_col - 1;
    memcpy((char *)buffer + 14, &right, sizeof(right));
    return 1;
}

static int ABI GetConsoleMode(void *handle, unsigned *mode)
{
    struct termios terminal;
    if (tcgetattr(handle_fd(handle), &terminal))
        return 0;
    *mode = ((terminal.c_lflag & ICANON) ? 2 : 0) |
            ((terminal.c_lflag & ECHO) ? 4 : 0);
    return 1;
}

// Output is already ANSI-capable. ReadKey applies raw mode only for each poll.
static int ABI SetConsoleMode(void *handle, unsigned mode)
{
    return isatty(handle_fd(handle));
}

static int pending_key = -1;
static int ABI PeekConsoleInputA(void *handle, void *record, unsigned length, unsigned *count)
{
    int fd = handle_fd(handle);
    *count = 0;
    if (pending_key < 0)
    {
        struct termios saved, raw;
        if (tcgetattr(fd, &saved))
            return 0;
        raw = saved;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = raw.c_cc[VTIME] = 0;
        if (tcsetattr(fd, TCSANOW, &raw))
            return 0;
        unsigned char byte;
        ssize_t got = read(fd, &byte, 1);
        int error = tcsetattr(fd, TCSANOW, &saved);
        if (got == 1)
            pending_key = byte;
        if (error || (got < 0 && errno != EAGAIN && errno != EINTR))
            return 0;
    }
    if (pending_key >= 0 && length)
    {
        unsigned char *bytes = record;
        memset(bytes, 0, 20);
        bytes[0] = bytes[4] = 1; // KEY_EVENT, key down.
        bytes[14] = pending_key; // AsciiChar in the INPUT_RECORD layout.
        *count = 1;
    }
    return 1;
}

static int ABI ReadConsoleInputA(void *handle, void *record, unsigned length, unsigned *count)
{
    int result = PeekConsoleInputA(handle, record, length, count);
    if (*count)
        pending_key = -1;
    return result;
}

static int ABI CreatePipe(void **reader, void **writer, void *security, unsigned size)
{
    int fds[2];
    if (pipe2(fds, O_CLOEXEC))
        return 0;
    *reader = fd_handle(fds[0]);
    *writer = fd_handle(fds[1]);
    return 1;
}

static int ABI SetHandleInformation(void *handle, unsigned mask, unsigned flags)
{
    // Child inheritance is explicit through posix_spawn file actions.
    return handle_fd(handle) >= 0 && !(flags & mask);
}

static int arg_space(char c)
{
    return c == ' ' || c == '\t';
}

// Decode the quoted argument format emitted by execute.inc without invoking a shell.
static size_t split_command(char *command, char **argv)
{
    const char *source = command;
    char *target = command;
    size_t argc = 0;
    while (*source)
    {
        while (arg_space(*source))
            ++source;
        if (!*source)
            break;
        argv[argc++] = target;
        int quoted = 0;
        while (*source && (quoted || !arg_space(*source)))
        {
            size_t slashes = 0;
            while (*source == '\\')
            {
                ++slashes;
                ++source;
            }
            if (*source == '"')
            {
                memset(target, '\\', slashes / 2);
                target += slashes / 2;
                ++source;
                if (slashes % 2)
                    *target++ = '"';
                else if (quoted && *source == '"')
                    *target++ = *source++;
                else
                    quoted = !quoted;
            }
            else
            {
                memset(target, '\\', slashes);
                target += slashes;
                if (*source && (quoted || !arg_space(*source)))
                    *target++ = *source++;
            }
        }
        while (arg_space(*source))
            ++source;
        *target++ = 0;
    }
    argv[argc] = NULL;
    return argc;
}

// Windows executables are spelled with a trailing .exe in every embedded
// tool path (autoexec.logo DefineCommand wrappers, sidecar scripts). The
// same tools build extensionless on POSIX, so the suffix is stripped from
// argv[0] here -- the one place every engine spawn passes through.
static void strip_exe_suffix(char *name)
{
    size_t len = strlen(name);
    if (len > 4 && !strcasecmp(name + len - 4, ".exe"))
        name[len - 4] = 0;
}

static int ABI CreateProcessA(const char *application, char *command, void *process_attr,
                               void *thread_attr, int inherit, unsigned flags, void *environment,
                               const char *directory, const char *startup, void **result)
{
    void *output, *error_output;
    memcpy(&output, startup + 88, sizeof(output));
    memcpy(&error_output, startup + 96, sizeof(error_output));
    char *copy = strdup(command);
    char **argv = calloc(strlen(command) + 1, sizeof(*argv));
    struct handle *child = calloc(1, sizeof(*child));
    int error = ENOMEM;
    if (copy && argv && child && split_command(copy, argv))
    {
        strip_exe_suffix(argv[0]);
        posix_spawn_file_actions_t actions;
        error = posix_spawn_file_actions_init(&actions);
        if (!error)
        {
            error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
            if (!error)
                error = posix_spawn_file_actions_adddup2(&actions, handle_fd(output), STDOUT_FILENO);
            if (!error)
                error = posix_spawn_file_actions_adddup2(&actions, handle_fd(error_output), STDERR_FILENO);
            if (!error)
                error = posix_spawn_file_actions_addclosefrom_np(&actions, 3);
            if (!error)
                error = posix_spawnp(&child->id, application ? application : argv[0], &actions,
                                     NULL, argv, environ);
            posix_spawn_file_actions_destroy(&actions);
        }
    }
    free(argv);
    free(copy);
    if (error)
    {
        free(child);
        errno = error;
        return 0;
    }
    child->kind = PROCESS;
    result[0] = child;
    result[1] = NULL;
    return 1;
}

static void *ABI LoadLibraryA(const char *path)
{
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

extern int ABI DllMain(void *, unsigned, void *);
extern int ABI v7s_get_num_workers(void);

static const char self_marker;

/* GetProcAddress hands raw ELF symbols to engine callers. Those callers
 * (CallLibrary marshalling, v7_lib's resolve macros) invoke what they get
 * back with the Windows register convention, so a symbol handed out here
 * must be compiled with the ms_abi attribute on the host side -- the
 * engine's own exports are assembly and already follow it. */
static void *ABI GetProcAddress(void *module, const char *name)
{
    if (module != &self_marker)
        return module ? dlsym(module, name) : NULL;
    Dl_info info;
    if (!dladdr(&self_marker, &info))
        return NULL;
    void *self = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW);
    if (!self)
        return NULL;
    void *symbol = dlsym(self, name);
    dlclose(self);
    return symbol;
}

static void ABI TerminateProcess(void *process, unsigned exit_code)
{
    _exit(exit_code);
}

__attribute__((constructor))
static void attach_engine(void)
{
    DllMain((void *)&self_marker, 1, NULL);
}

__attribute__((destructor))
static void detach_engine(void)
{
    if (v7s_get_num_workers())
        DllMain(NULL, 0, NULL);
}

IMPORT(CloseHandle);
IMPORT(CreateFileA);
IMPORT(CreateFileMappingW);
IMPORT(CreateFileW);
IMPORT(CreatePipe);
IMPORT(CreateProcessA);
IMPORT(CreateThread);
IMPORT(DeleteFileA);
IMPORT(DeviceIoControl);
IMPORT(DuplicateHandle);
IMPORT(GetConsoleMode);
IMPORT(GetConsoleScreenBufferInfo);
IMPORT(GetCurrentProcessorNumber);
IMPORT(GetEnvironmentVariableA);
IMPORT(GetFileSize);
IMPORT(GetFileType);
IMPORT(GetLastError);
IMPORT(GetLogicalProcessorInformationEx);
IMPORT(GetProcAddress);
IMPORT(GetProcessAffinityMask);
IMPORT(GetStdHandle);
IMPORT(GetTempFileNameW);
IMPORT(GetTempPathW);
IMPORT(GetThreadPriority);
IMPORT(LoadLibraryA);
IMPORT(MapViewOfFile);
IMPORT(MoveFileExA);
IMPORT(PeekConsoleInputA);
IMPORT(PeekNamedPipe);
IMPORT(ReadConsoleInputA);
IMPORT(ReadFile);
IMPORT(SetConsoleMode);
IMPORT(SetEndOfFile);
IMPORT(SetFilePointerEx);
IMPORT(SetHandleInformation);
IMPORT(SetThreadAffinityMask);
IMPORT(SetThreadPriority);
IMPORT(Sleep);
IMPORT(TerminateProcess);
IMPORT(UnmapViewOfFile);
IMPORT(VirtualAlloc);
IMPORT(VirtualFree);
IMPORT(VirtualProtect);
IMPORT(WaitForSingleObject);
IMPORT(WaitOnAddress);
IMPORT(WakeByAddressAll);
IMPORT(WriteFile);
