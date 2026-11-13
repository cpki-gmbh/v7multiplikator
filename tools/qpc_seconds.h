// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * qpc_seconds.h -- shared QPC wall clock for the single-file benchmark tools.
 *
 * Include after windows.h. get_seconds() initializes the QPC frequency once
 * and returns fractional seconds; a failed QPC call is named and aborts --
 * benchmark numbers from a broken timer are worse than no numbers.
 */
#pragma once
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

static inline double get_seconds(void)
{
	static LARGE_INTEGER freq = { 0 };
	static int have_freq = 0;
	if (!have_freq)
	{
		if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0)
		{
			printf("[FATAL] get_seconds: QueryPerformanceFrequency failed or returned 0 -- no usable timer.\n");
			exit(EXIT_FAILURE);
		}
		have_freq = 1;
	}
	LARGE_INTEGER now;
	if (!QueryPerformanceCounter(&now))
	{
		printf("[FATAL] get_seconds: QueryPerformanceCounter failed.\n");
		exit(EXIT_FAILURE);
	}
	return (double)now.QuadPart / (double)freq.QuadPart;
}
