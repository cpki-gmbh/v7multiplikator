// v7 - POSIX primitives for the Linux HTTP server
// License: GPL v3, see LICENSE
#pragma once
#include <windows.h> // Local CLI adapters.
#include <arpa/inet.h>
#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/random.h>
#include <sys/socket.h>

using SOCKET = int;
using CRITICAL_SECTION = pthread_mutex_t;
constexpr SOCKET INVALID_SOCKET = -1;
constexpr int SOCKET_ERROR = -1;
constexpr int SD_SEND = SHUT_WR;
constexpr int WSAETIMEDOUT = ETIMEDOUT;
constexpr int WSAEADDRINUSE = EADDRINUSE;
constexpr int WSAEACCES = EACCES;

static inline int WSAGetLastError()
{
    return errno == EAGAIN || errno == EWOULDBLOCK ? ETIMEDOUT : errno;
}

static inline void closesocket(SOCKET fd)
{
    close(fd);
}

static inline void InitializeCriticalSection(CRITICAL_SECTION *lock)
{
    if (pthread_mutex_init(lock, nullptr))
        abort();
}

static inline void DeleteCriticalSection(CRITICAL_SECTION *lock)
{
    pthread_mutex_destroy(lock);
}

static inline void EnterCriticalSection(CRITICAL_SECTION *lock)
{
    pthread_mutex_lock(lock);
}

static inline void LeaveCriticalSection(CRITICAL_SECTION *lock)
{
    pthread_mutex_unlock(lock);
}

// Descriptor plus one leaves zero available as the cross-platform null handle.
using server_event = uintptr_t;

static inline server_event server_create_event()
{
    return (uintptr_t)(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK) + 1);
}

static inline void server_signal_event(server_event event)
{
    while (eventfd_write((int)event - 1, 1) && errno == EINTR)
        ;
}

static inline void server_wait_event(server_event event, unsigned timeout)
{
    struct pollfd descriptor = { (int)event - 1, POLLIN, 0 };
    if (poll(&descriptor, 1, timeout) > 0)
    {
        eventfd_t value;
        while (eventfd_read(descriptor.fd, &value) && errno == EINTR)
            ;
    }
}

static inline void server_close_event(server_event event)
{
    close((int)event - 1);
}

static inline unsigned long long GetTickCount64()
{
    LARGE_INTEGER counter = {};
    QueryPerformanceCounter(&counter);
    return counter.QuadPart / 1000000;
}

static inline int server_random_bytes(unsigned char *buffer, size_t size)
{
    while (size)
    {
        ssize_t length = getrandom(buffer, size, 0);
        if (length < 0 && errno == EINTR)
            continue;
        if (length <= 0)
            return 0;
        buffer += length;
        size -= length;
    }
    return 1;
}

static inline unsigned GetEnvironmentVariableA(const char *name, char *output, unsigned capacity)
{
    const char *value = getenv(name);
    if (!value)
        return 0;
    size_t length = strlen(value);
    if (length >= capacity)
        return length + 1;
    memcpy(output, value, length + 1);
    return length;
}
