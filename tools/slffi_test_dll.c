// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* slffi_test_dll.c -- Mini C DLL for SectorLogo FFI (C-CALL) testing
 *
 * Provides deterministic endpoints that the SectorLogo validator calls
 * via C-CALL descriptors like:
 *   ["slffi_test.dll" "slffi_add" "INT" ["INT" "INT"]]
 *
 * Build: cl /nologo /O2 /LD slffi_test_dll.c /Fe:..\bin\slffi_test.dll
 *        (Linux) make bin-linux/slffi_test.so
 *
 * Functions cover: 0-5 args, INT/PTR/STR/VOID returns, stack args (>4).
 *
 * The engine calls every CallLibrary target with the Windows register
 * convention on every platform, so the Linux build must carry the ms_abi
 * attribute or every argument-taking endpoint would read its parameters
 * from the wrong registers. */
#include <string.h>
#include <stdlib.h>

#ifdef __linux__
#define SLFFI_EXPORT __attribute__((ms_abi))
#else
#define SLFFI_EXPORT __declspec(dllexport)
#endif

/* Deterministic integer arithmetic: 2 args */
SLFFI_EXPORT int slffi_add(int first, int second)
{
	return first + second;
}
SLFFI_EXPORT int slffi_mul(int first, int second)
{
	return first * second;
}
SLFFI_EXPORT int slffi_sub(int first, int second)
{
	return first - second;
}

/* Single-arg INT: doubling */
SLFFI_EXPORT int slffi_double(int input_value)
{
	return input_value * 2;
}

/* Zero args: returns a ant */
SLFFI_EXPORT int slffi_noargs(void)
{
	return 42;
}

/* Zero args: an INT return that does NOT fit a 30-bit NUMBER payload.
   INT is the only integer return type the FFI offers, so this value has no
   correct spelling in a script -- the marshaller must diagnose it rather than
   let the payload mask turn 0x40000000 into 0. */
SLFFI_EXPORT int slffi_too_big(void)
{
	return 0x40000000;
}

/* PTR arg -> INT return: string length */
SLFFI_EXPORT int slffi_strlen(char *string_ptr)
{
	if (!string_ptr)
		return 0;
	return (int)strlen(string_ptr);
}

/* PTR arg -> STR return: returns the same pointer (identity) */
SLFFI_EXPORT char *slffi_identity(char *string_ptr)
{
	return string_ptr;
}

/* VOID return + INT arg: sets a global, returns nothing */
int test_global_value = 0;
SLFFI_EXPORT void slffi_set_global(int new_value)
{
	test_global_value = new_value;
}
SLFFI_EXPORT int slffi_get_global(void)
{
	return test_global_value;
}

/* Three args: tests 3-register passing (rcx, rdx, r8) */
SLFFI_EXPORT int slffi_add3(int first, int second, int third)
{
	return first + second + third;
}

/* Five args: tests stack arg passing (args 5+ on stack) */
SLFFI_EXPORT int slffi_add5(int first, int second, int third, int fourth, int fifth)
{
	return first + second + third + fourth + fifth;
}

/* PTR arg + INT arg: strncmp-style comparison, returns 0/1 */
SLFFI_EXPORT int slffi_starts_with(char *string_ptr, int prefix_len)
{
	if (!string_ptr)
		return 0;
	if ((int)strlen(string_ptr) >= (int)prefix_len)
		return 1;
	else
		return 0;
}
