# CUDA-JS-Tensor foundation plan

**Date:** 2026-08-26
**Parent objective:** Deliver a separate, universal, easy-and-complete tensor library over public CUDA-JS, with a complete SIMT path and qualified optional dense acceleration.

## Global invariants

- Public contracts are domain-neutral and backend-neutral.
- Convenience is normalization over complete contracts.
- One session/device in v1; no cross-device tensor identity.
- No direct native CUDA or CUDA-JS internals.
- No production `.cu`/`.ptx`.
- No training, NN, MCGS or first-consumer policy in core.
- `the_restaurant` work is deferred; plan retention is the only authorized action.

## Focus branches

| ID | Owner/output | Depends on | Acceptance / falsifier |
|---|---|---|---|
| `TENSOR-BOOT-000` | Public repo, authority, specs, validation, protections | Project-owner direction | Remote main and settings read back; no production claim. |
| `CJS-TENSOR-VIEW-001` | Upstream public typed device-view/dtype facade | Accepted CUDA-JS SPEC-0021 reassessment | Completed in CUDA-JS SPEC-0021; exact dependency revision recorded. |
| `CJS-DEVICE-LIB-002` | Upstream typed device-callable import/library composition | RDC/LTO and resource/participation contracts | Completed in CUDA-JS SPEC-0028; tensor provider profile remains separate. |
| `CJS-PREPARED-DAG-003` | Upstream prepared finite operation DAG baseline | Accepted operation/transfer lifecycle | Completed in CUDA-JS SPEC-0020 as semantic single-stream replay. |
| `CJS-LIB-ADAPTER-004` | Upstream context-bound library framework and cuBLASLt profile | Public typed views + operations | Completed in CUDA-JS SPEC-0023/SPEC-0029 with exact independent GEMM oracle. |
| `CJS-DEVICE-NUMERIC-005` | Upstream consumer-neutral dense numeric Device-JS profile ([CUDA-JS #139](https://github.com/iteathen/CUDA-JS/issues/139)) | Implemented program semantics expose exact executor needs | Completed in CUDA-JS SPEC-0030; alpha.14 also corrects native prepared-DAG identity and proves exact recorded-profile replay. |
| `TENSOR-VALUE-010` | Portable TensorSession/TensorSpec/Tensor value model | `CJS-TENSOR-VIEW-001` for native path | Implemented portable/package profile; zero/one/many shapes, views, limits, ownership, overload equivalence, deletion and cleanup pass. Native promotion remains unclaimed. |
| `TENSOR-PROGRAM-011` | Immutable TensorProgram and finite static TensorPlan | `TENSOR-VALUE-010` | Implemented portable/package profile: first dense inference, declared alias/lifetime/distinct-allocation identity and failure tests pass; execution facts remain explicitly unresolved. |
| `TENSOR-SIMT-012` | ResolvedTensorPlan plus generated Device-JS dense baseline | public views/compiler/operations, `TENSOR-PROGRAM-011`, and achieved `CJS-DEVICE-NUMERIC-005` profile | Integrated through PR #7 at `main@a2c9d4a0`; alpha.3 passes exact session/backend/program/workspace/cleanup binding, full portable/package validation and an independent installed-package Windows native mathematical/replay/lifecycle fixture. |
| `TENSOR-CUBLASLT-013` | Host accelerated matmul adapter ([closed issue #8](https://github.com/iteathen/CUDA-JS-Tensor/issues/8)); SPEC-0006 | CUDA-JS SPEC-0031 at alpha.15, `TENSOR-PROGRAM-011`, `TENSOR-SIMT-012` | Integrated through protected PR #10 at `main@8910309a`; exact native oracle, mixed prepared replay and terminal cleanup pass. SIMT stays the default pending separate performance evidence. |
| `TENSOR-FUSION-017` | Optional exact generated elementwise-chain fusion ([issue #14](https://github.com/iteathen/CUDA-JS-Tensor/issues/14)) | `TENSOR-SIMT-012`, accepted SPEC-0007 | Exact rounding/observability/access/failure identity; complete unfused deletion. Selected as the next child-specification lane. |
| `TENSOR-ARENA-018` | Optional result-owned material arena | accepted post-fusion material schedule, CUDA-JS SPEC-0021 | Alias-closed range/rollback/child cleanup; complete distinct-allocation deletion. Follows fusion. |
| `TENSOR-BATCHED-GEMM-015` | Optional rank-3 f32 cuBLASLt profile | future CUDA-JS strided-batched plan/node child ([CUDA-JS #75](https://github.com/iteathen/CUDA-JS/issues/75)) | One bounded provider plan, no rank-2 host loop; currently blocked upstream. |
| `TENSOR-PRECISION-GEMM-016` | Optional f16/bf16/f64 cuBLASLt profiles | future CUDA-JS typed plan/node children ([CUDA-JS #75](https://github.com/iteathen/CUDA-JS/issues/75)) | Exact input/compute/scale/output semantics; currently blocked upstream. |
| `TENSOR-DEVICE-DENSE-014` | Device-callable compact dense subgraph | `CJS-DEVICE-LIB-002`, selected CUDA-JS parallel helpers, `TENSOR-SIMT-012` | Exact useful collective participation/resource/publication gates; deferred rather than implemented as a serial device function. |
| `TENSOR-RELEASE-020` | Package consumers, examples, compatibility and release evidence | accepted preceding slices | Clean tarball, unrelated consumers, exact support claims, no private dependency. |

Only one shared CUDA-JS contract changes at a time. Downstream evidence is invalidated when its exact upstream package/revision or contract changes.

## Sequencing

1. Bootstrap and protect the repository.
2. Integrate the generic CUDA-JS prerequisites and record one exact compatibility pair. Current pair is CUDA-JS `af29b95e` / `0.1.0-alpha.15`, including SPEC-0030 dense numeric Device-JS, native prepared-DAG correction and SPEC-0031 mixed cuBLASLt nodes.
3. Implement portable value/program semantics over public CUDA-JS only. `TENSOR-VALUE-010`, `TENSOR-PROGRAM-011` and resolved generated SIMT execution under `TENSOR-SIMT-012` are integrated.
4. Implement the consumer-neutral CUDA-JS dense numeric Device-JS profile exposed by program-semantics assessment, then the generated SIMT path. Both are integrated; SPEC-0005 governs the Tensor resolution/lowering/result boundary without generated native-source workarounds.
5. Add the bounded SPEC-0006 host-planned cuBLASLt realization over public CUDA-JS alpha.15. Preserve the complete SIMT default until performance qualification; device-callable dense adapters remain later work.
6. Specify and implement exact elementwise fusion, then result-owned arena reuse over the post-fusion material schedule.
7. Extend accelerated batched/precision and device-callable profiles only after their bounded CUDA-JS dependencies are accepted.
8. Qualify and publish only the exact achieved surface.

## Backpressure and split rules

Split before a focus branch cannot retain implementation, focused tests, repository validation, exact-head review, cleanup and handoff in one coherent context. Defer extra dtypes, operations, layouts, backends, examples and polish before reducing required semantic/lifecycle evidence.

## Cleanup

Each branch disposes generated artifacts, caches, worktrees, remote branches, drafts and experiments after preserving required evidence. Protected main, user work, source authority and qualification evidence remain protected.
