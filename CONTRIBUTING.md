# Contributing

Issues and discussions are open to everyone. Please begin substantial API, backend, dtype, layout, synchronization, or dependency changes with an issue or discussion that identifies the owned invariant, use cases, exclusions, lifecycle, public CUDA-JS dependency, evidence, and cleanup.

## AI-assisted development

CUDA-JS-Tensor may use substantial AI-agent assistance. AI-generated code, prose, analysis, and model review are working material, not validation evidence. Contributors and maintainers remain responsible for understanding the change and for every correctness, provenance, security, compatibility, and qualification claim attached to it.

AI-assisted contributions are welcome under the same review and `npm run verify` bar as any other contribution. Routine AI use does not require a prompt log; disclose material assistance when it affects provenance, licensing, security review, reproducibility, or another repository requirement. Do not cite model agreement as proof of correctness.

Read [`AGENTS.md`](AGENTS.md), the governing ADR/specification, and the system registry. Work on a short-lived branch, keep changes ownership-coherent, add focused tests, run `npm run verify`, and open a pull request against `main`.

Do not submit native addons, direct CUDA bindings, `.cu`/`.ptx` production files, generated artifacts, machine-specific provider paths, model/training policy, or deep CUDA-JS imports.

Bug reports should include exact package/revision, Node/OS/ABI, CUDA-JS version, CUDA environment when relevant, minimal shape/dtype/layout, expected and actual behavior, and whether all resources closed terminally. Remove credentials and stable machine identifiers.
