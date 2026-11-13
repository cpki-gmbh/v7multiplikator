# Research experiments (archived, read-only history)

Sidecar strategy experiments from the RAG research phase. Each directory holds
a knowledge base, a `sidecar_*.logo` script applying one injection strategy,
and a `run.bat` that runs the canonical prompt against the model with that
sidecar active. These are historical records — the findings live in the
research logbook.

## Experiments

| Directory | Strategy tested |
|-----------|-----------------|
| `herbivorous_cats/` | False-belief override: inject "cats are strict herbivores" (contradicts a fact the model is certain about). |
| `helion_pump/` | Fictional-fact injection: maintenance data for a pump that does not exist. |
| `ardmark/` | Fictional history: a capital move of a country that does not exist. |
| `bible/`, `herbivorous/`, `sidecar_*.logo` (22 variants) | The strategy matrix from the injection-strategy research. |

## Run one

```bat
examples\research\herbivorous_cats\run.bat     (from the project root)
examples\research\helion_pump\run.bat
```

The Ardmark storyteller demo graduated out of research: it now lives at
`examples\ardmark\` and is the project's prime RAG demo (see its own README).

## Further reading

- the master research log (dev-only `attic\docs_research_logbook\`) — the master experiment log
- the `sidecar_*.logo` scripts in this directory - each strategy's rewrite steps are documented in its header comments
- `examples/README.md` — the top-level examples guide
