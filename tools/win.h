// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#define WIN32_LEAN_AND_MEAN // exclude rarely-used stuff from windows headers
#include <windows.h>
#include <time.h>

#define ftell _ftelli64

// below code is originally from mman-win32 (third-party, BSD-licensed).
// THIRD-PARTY EXCEPTION: style rules from AGENTS.md section 6a are not applied
// to the vendored mman-win32 block below; only real compile bugs are fixed.
//
/*
 * sys/mman.h
 * mman-win32
 */

#ifndef _WIN32_WINNT        // allow use of features specific to windows XP or later.
#define _WIN32_WINNT 0x0501 // change this to the appropriate value to target other versions of windows.
#endif

/* all the headers include this file. */
#ifndef _MSC_VER
#include <_mingw.h>
#endif

#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4

#define MAP_FILE 0
#define MAP_SHARED 1
#define MAP_PRIVATE 2
#define MAP_TYPE 0xf
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS

#define MAP_FAILED ((void *)-1)

/* flags for msync. */
#define MS_ASYNC 1
#define MS_SYNC 2
#define MS_INVALIDATE 4

/* flags for portable clock_gettime call. */
#define CLOCK_REALTIME 0

	void *mmap(void *addr, int len, int prot, int flags, int fildes, __int64 off);
	int munmap(void *addr, int len);
	int mprotect(void *addr, int len, int prot);
	int msync(void *addr, int len, int flags);
	int mlock(void *addr, int len);
	int munlock(void *addr, int len);
	int clock_gettime(int clk_id, struct timespec *tp);

#ifdef __cplusplus
};
#endif
