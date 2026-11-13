# About AGENTS_dev.md

`AGENTS_dev.md` is the development tree's internal agent handbook — the operating rules for AI agents
contributing to v7. This reference records development policies: the review/report loop, the
verification gates, the coding and commenting conventions, and the engineering decisions behind them.

It is a development artifact, not user documentation:

- For building and running V7, start with [README.md](README.md) and the
  [installation guide](getting_started.md).
- For architecture and APIs, consult the [architecture guide](architecture.md),
  [native API reference](native_api.md), [model guide](models_and_vision.md),
  and [SectorLogo manual](sectorlogo.md).
- The rules in it bind the development process, not the software: nothing
  in it changes what the shipped binaries do.

The handbook was copied from the development tree's `AGENTS.md` and is included here as an edited reference copy. Its historical implementation
notes and proposed changes must not be treated as proof of the current binary behavior. The development repository's current `AGENTS.md`
remains authoritative for active agent instructions.

Some older sections of the handbook describe superseded mechanisms. In particular,
the worker spin contract has been updated in §4a; other historical process
notes should be verified against the release source before application.
