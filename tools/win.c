// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "win.h"
#include <errno.h>
#include <io.h>

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE 0x0020
#endif /* FILE_MAP_EXECUTE */

int __map_mman_error(unsigned int err, int deferr)
{
	if (err == 0)
		return 0;
	// no message table is linked: the raw Win32 code is returned as the errno value
	return err;
}

unsigned int __map_mmap_prot_page(int prot)
{
	unsigned int protect = 0;

	if (prot == PROT_NONE)
		return protect;

	if ((prot & PROT_EXEC) != 0)
	{
		protect = ((prot & PROT_WRITE) != 0) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
	}
	else
	{
		protect = ((prot & PROT_WRITE) != 0) ? PAGE_READWRITE : PAGE_READONLY;
	}

	return protect;
}

unsigned int __map_mmap_prot_file(int prot)
{
	unsigned int desiredAccess = 0;

	if (prot == PROT_NONE)
		return desiredAccess;

	if ((prot & PROT_READ) != 0)
		desiredAccess |= FILE_MAP_READ;
	if ((prot & PROT_WRITE) != 0)
		desiredAccess |= FILE_MAP_WRITE;
	if ((prot & PROT_EXEC) != 0)
		desiredAccess |= FILE_MAP_EXECUTE;

	return desiredAccess;
}

void *mmap(void *addr, int len, int prot, int flags, int fildes, __int64 off)
{
	HANDLE fm, h;
	void *map = MAP_FAILED;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4293)
#endif

	unsigned int dwFileOffsetLow = (unsigned int)(off & 0xFFFFFFFFL);
	unsigned int dwFileOffsetHigh = (unsigned int)((off >> 32) & 0xFFFFFFFFL);
	unsigned int protect = __map_mmap_prot_page(prot);
	unsigned int desiredAccess = __map_mmap_prot_file(prot);

	__int64 maxSize = off + (__int64)len;

	unsigned int dwMaxSizeLow = (unsigned int)(maxSize & 0xFFFFFFFFL);
	unsigned int dwMaxSizeHigh = (unsigned int)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

	errno = 0;

	if (len == 0
	    /* unsupported flag combinations */
	    || (flags & MAP_FIXED) != 0
	    /* unsupported protection combinations */
	    || prot == PROT_EXEC)
	{
		errno = EINVAL;
		return MAP_FAILED;
	}

	h = ((flags & MAP_ANONYMOUS) == 0) ? (HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

	if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
	{
		errno = EBADF;
		return MAP_FAILED;
	}

	fm = CreateFileMappingA(h, 0, protect, dwMaxSizeHigh, dwMaxSizeLow, 0);

	if (fm == 0)
	{
		errno = __map_mman_error(GetLastError(), EPERM);
		return MAP_FAILED;
	}

	map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

	CloseHandle(fm);

	if (map == 0)
	{
		errno = __map_mman_error(GetLastError(), EPERM);
		return MAP_FAILED;
	}

	return map;
}

int munmap(void *addr, int len)
{
	if (UnmapViewOfFile(addr))
		return 0;

	errno = __map_mman_error(GetLastError(), EPERM);

	return -1;
}

int mprotect(void *addr, int len, int prot)
{
	unsigned int newProtect = __map_mmap_prot_page(prot);
	DWORD oldProtect = 0;

	if (VirtualProtect(addr, len, newProtect, &oldProtect))
		return 0;

	errno = __map_mman_error(GetLastError(), EPERM);

	return -1;
}

int msync(void *addr, int len, int flags)
{
	if (FlushViewOfFile(addr, len))
		return 0;

	errno = __map_mman_error(GetLastError(), EPERM);

	return -1;
}

int mlock(void *addr, int len)
{
	if (VirtualLock((LPVOID)addr, len))
		return 0;

	errno = __map_mman_error(GetLastError(), EPERM);

	return -1;
}

int munlock(void *addr, int len)
{
	if (VirtualUnlock((LPVOID)addr, len))
		return 0;

	errno = __map_mman_error(GetLastError(), EPERM);

	return -1;
}

// portable clock_gettime function for windows
int clock_gettime(int clk_id, struct timespec *tp)
{
	unsigned int ticks = GetTickCount();
	tp->tv_sec = ticks / 1000;
	tp->tv_nsec = (ticks % 1000) * 1000000;
	return 0;
}
