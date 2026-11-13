The release tree ships a no-op stub in tools\zydis_stub\zydis_debug_header.h: it uses the same
header name the dev tree includes (tools\zydis_debug_header.h) but implements nothing, so every
C/C++ target that includes the debug header compiles cleanly without the real decoder. The real
Zydis VEH crash-disassembly machinery (the Zydis libraries plus the full header) exists only in
the dev tree and is deliberately not shipped - release builds stay small and self-contained.

The dev-to-release sync replaces the stub copy on every sync; build.bat
compiles tools against the stub in the release and against the real decoder
in dev. Regenerate the stub only when the include contract changes in dev.
