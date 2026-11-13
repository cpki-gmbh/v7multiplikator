# test_engine_death

Diagnostic error-path test for the pseudo-exception machinery: forces an
internal FATAL through the public API and checks the host survives.

## Run

```bat
bin\test_engine_death.exe bin\v7_threading.dll
```

Takes only the DLL path -- no database is needed.

## What it proves (per Sec.172)

1. A public call that hits an internal FATAL returns control to the C caller
   (the gated landing zone unwinds instead of killing the process).
2. `v7_engine_is_dead()` reports the death and `v7_get_death_reason()`
   returns the reason text.
3. A dead-gated second call returns instantly with the buffer untouched.
4. `v7_free_inference()` + unload revive the engine for the next context
   (the CPUID forced-flag sub-test t0b also lives here).

verify.bat runs this as one of its smokes.
