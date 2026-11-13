// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* test_engine_death.c -- pseudo-exception catch test (docs/plans/fatal_to_pseudo_exception_plan.md, T1/T2)
 *
 * Forces a FATAL through the public v7_chat wrapper (the init-order contract
 * check fires because no inference context was created), then verifies the
 * host process SURVIVES the unwind, can query the death reason, sees further
 * gated calls early-return without side effects, and can cleanly FreeLibrary
 * the dead engine. This is the lean variant: no DB load, no RunState
 * allocation -- safe to run under memory pressure during verify.bat.
 * (The full-context NULL-output variant exercises the identical unwind path
 * deeper inside the DLL; it needs the ~10 GB RunState and stays a manual test.)
 *
 * usage:  test_engine_death.exe <dll_path>
 * exit 0 = all subtests passed; nonzero = failure.
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <string.h>
#include <windows.h>

#define SENTINEL_BUF_SIZE 64 /* small buffer with an 'X' sentinel for the dead-gate check */

/* Sec.170 coverage for the TEB_THREAD_ID offset (engine_landing BUG-1).
 * Every thread-identity tripwire in the engine -- the SectorLogo private-stack
 * owner check and the landing zone's own ownership test -- reads this one slot,
 * and 0 is the "unowned" sentinel in both, so the read MUST be a real nonzero
 * thread id. A wrong offset fails this immediately and loudly.
 *
 * The slot is read as a direct gs-relative load, never through a TEB
 * self-pointer: gs:[0] (NT_TIB.ExceptionList) is not a usable TEB base here
 * and reads 0, so dereferencing it faults on the NULL page.
 *
 * TEB_THREAD_ID = 0x48 = ClientId.UniqueThread, low dword. The two offsets a
 * reader is most likely to reach for are both wrong, and the second one fails
 * SILENTLY rather than loudly, which is what makes it the dangerous one:
 *   0x40 = ClientId.UniqueProcess, low dword = the PROCESS id. Nonzero, so
 *          every tripwire looks alive, but it is identical on the host thread
 *          and on every worker -- so a worker-thread FATAL would pass the
 *          "do I own the landing zone?" test and unwind onto the host's stack.
 *   0x44 = ClientId.UniqueProcess, high dword = 0 on every real thread, which
 *          reads as the "unowned" sentinel and silently disables every
 *          tripwire that reads it.
 * Comparing against GetCurrentThreadId() catches the 0x40 case: the PID never
 * equals a thread id. */
#define TEB_THREAD_ID_SLOT 0x48

static int teb_thread_id_is_sane(void)
{
	unsigned teb_tid = (unsigned)__readgsqword(TEB_THREAD_ID_SLOT);

	if (teb_tid == 0)
	{
		printf("FAIL: TEB ClientId.UniqueThread low dword read 0 -- the id is the\n");
		printf("      'unowned' sentinel, so every thread-identity tripwire is dead\n");
		return 0;
	}
	if (teb_tid != (unsigned)GetCurrentThreadId())
	{
		printf("FAIL: TEB+0x%02X read %u but GetCurrentThreadId() is %u -- wrong offset\n",
		       TEB_THREAD_ID_SLOT, teb_tid, (unsigned)GetCurrentThreadId());
		return 0;
	}
	return 1;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 2)
	{
		printf("Usage: %s <dll_path>\n", argv[0]);
		printf("  dll_path is REQUIRED -- path to v7_threading.dll (no default)\n");
		return 1;
	}

	/* step 0: the thread-identity tripwire the whole catch machinery rests on.
	 * A dead offset here means handle_fatal would unwind a WORKER thread onto
	 * the host's stack instead of taking the terminal path. */
	if (teb_thread_id_is_sane() == 0)
		return 1;
	printf("[t0] TEB thread id is a real nonzero identity -- ok\n");

	/* step 1: load the engine (workers are NOT started until v7_init runs) */
	struct v7_lib dll;
	if (v7_load(argv[1], &dll) != 0)
	{
		printf("FAIL: cannot load '%s'\n", argv[1]);
		return 1;
	}
	printf("[t0] engine loaded\n");

	if (dll.v7_engine_is_dead() != 0)
	{
		printf("FAIL: freshly loaded engine must be alive (is_dead == 0)\n");
		v7_unload(&dll);
		return 1;
	}
	if (dll.v7_get_death_reason()[0] != 0)
	{
		printf("FAIL: death reason must be empty while alive\n");
		v7_unload(&dll);
		return 1;
	}
	printf("[t0] alive, reason empty -- ok\n");

	/* step 1b (t0b): the CPUID gate. Force the flag the DllMain probe writes, then
	 * call a public entry -- it must FATAL, unwind HERE, and leave the machine state
	 * untouched (cpu_gate_fatal is reached by jmp from a gated site, so handle_fatal
	 * restores the full callee-saved file and jumps to engine_saved_rsp). Then
	 * restore the flag and prove the death is recoverable: v7_free_inference resets
	 * the death state even with no context, so the rest of this suite can still run. */
	printf("[t0b] forcing cpu_features_ok = 0 and calling v7_initialize_inference...\n");
	*dll.cpu_features_ok = 0;
	dll.v7_initialize_inference();
	if (dll.v7_engine_is_dead() != 1)
	{
		printf("FAIL: unsupported-CPU gate must park the engine\n");
		v7_unload(&dll);
		return 1;
	}
	const char *cpu_reason = dll.v7_get_death_reason();
	if (strstr(cpu_reason, "cpu_gate") == 0)
	{
		printf("FAIL: death reason must name the cpu gate, got: %s\n", cpu_reason);
		v7_unload(&dll);
		return 1;
	}
	printf("[t0b] host SURVIVED -- reason: %s\n", cpu_reason);
	*dll.cpu_features_ok = 1;
	dll.v7_free_inference();
	if (dll.v7_engine_is_dead() != 0)
	{
		printf("FAIL: v7_free_inference must reset the death state\n");
		v7_unload(&dll);
		return 1;
	}
	printf("[t0b] engine alive again after teardown -- ok\n");

	/* step 2 (t1): forced FATAL via the init-order contract violation.
	 * v7_chat's gate installs the landing zone, the wrapper's context check
	 * FATALs, handle_fatal records + prints + parks the engine and unwinds
	 * RIGHT BACK HERE. The process must survive with every callee-saved
	 * register restored (the zone restores them -- that is the whole point). */
	printf("[t1] v7_chat before v7_initialize_inference (forces a FATAL)...\n");
	dll.v7_chat("trigger", 0, 0, 0, 1);

	if (dll.v7_engine_is_dead() != 1)
	{
		printf("FAIL: engine must be dead (is_dead == 1) after the caught FATAL\n");
		v7_unload(&dll);
		return 1;
	}
	const char *reason = dll.v7_get_death_reason();
	if (reason[0] == 0)
	{
		printf("FAIL: death reason must be non-empty after death\n");
		v7_unload(&dll);
		return 1;
	}
	printf("[t1] host SURVIVED the unwind -- reason: %s\n", reason);

	/* step 3 (t2): dead engine -- a further gated call must return instantly,
	 * touch nothing, and leave the engine dead. */
	char sentinel[SENTINEL_BUF_SIZE];
	sentinel[0] = 'X';
	sentinel[1] = 0;
	dll.v7_chat("second call", 0, sentinel, SENTINEL_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead() != 1)
	{
		printf("FAIL: engine must still be dead after the second call\n");
		v7_unload(&dll);
		return 1;
	}
	if (sentinel[0] != 'X' || sentinel[1] != 0)
	{
		printf("FAIL: dead-gated call must not touch the output buffer\n");
		v7_unload(&dll);
		return 1;
	}
	printf("[t2] dead-gated second call returned instantly, buffer untouched -- ok\n");

	/* step 4 (t3): clean teardown of a dead-but-loaded engine. No context was
	 * ever created (v7_free_inference no-ops), no workers exist, so plain
	 * FreeLibrary must succeed -- proving a host can recover from an engine
	 * FATAL without dying itself. */
	v7_unload(&dll);
	printf("[t3] FreeLibrary after death succeeded\n");

	printf("engine death test: all passed\n");
	return 0;
}
