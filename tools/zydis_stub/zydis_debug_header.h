// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* zydis_debug_header.h -- RELEASE STUB (no-op)
 *
 * This is the release-tree replacement for the dev root zydis_debug_header.h.
 * The release ships WITHOUT the Zydis library (Zydis.lib, zydis-master\,
 * zycore-c-master\), so the crash-disassembly VEH machinery is compiled out:
 * ZydisInit() and ZydisSetCrashLogFile() are silent no-ops and a crash in a
 * release-built tool prints nothing extra -- it just crashes.
 *
 * DEV MUST NEVER BUILD AGAINST THIS STUB. Dev keeps the real Zydis and the
 * real header at the repo root; build.bat picks the stub ONLY when
 * zydis-master\include\Zydis\Zydis.h and Zydis.lib are both absent. See
 * docs\zydis_debug_readme.txt for how to restore real Zydis in a tree that
 * only has the stub.
 *
 * The public interface is exactly the two entry points every tool calls:
 *   ZydisInit()                  -- call at the very start of main()
 *   ZydisSetCrashLogFile(path)   -- optional crash-log redirect (no-op here)
 *
 * The include guard matches the real header so a tree can never pull both.
 */

#ifndef ZYDIS_DEBUG_HEADER_H
#define ZYDIS_DEBUG_HEADER_H

static void ZydisInit(void)
{
	/* no-op: the release carries no Zydis decoder and no VEH crash handler */
}

static void ZydisSetCrashLogFile(const char* crash_log_path)
{
	/* no-op: no decoder means no crash dump, so there is nothing to redirect */
	(void)crash_log_path;
}

#endif /* ZYDIS_DEBUG_HEADER_H */
