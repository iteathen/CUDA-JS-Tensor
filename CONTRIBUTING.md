# Contributing

Issues and discussions are open to everyone. Please begin substantial API, backend, dtype, layout, synchronization, or dependency changes with an issue or discussion that identifies the owned invariant, use cases, exclusions, lifecycle, public CUDA-JS dependency, evidence, and cleanup.

Read [`AGENTS.md`](AGENTS.md), the governing ADR/specification, and the system registry. Work on a short-lived branch, keep changes ownership-coherent, add focused tests, run `npm run verify`, and open a pull request against `main`.

Do not submit native addons, direct CUDA bindings, `.cu`/`.ptx` production files, generated artifacts, machine-specific provider paths, model/training policy, or deep CUDA-JS imports.

Bug reports should include exact package/revision, Node/OS/ABI, CUDA-JS version, CUDA environment when relevant, minimal shape/dtype/layout, expected and actual behavior, and whether all resources closed terminally. Remove credentials and stable machine identifiers.

