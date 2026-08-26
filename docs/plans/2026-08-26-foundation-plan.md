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
| `CJS-DEVICE-NUMERIC-005` | Upstream consumer-neutral dense numeric Device-JS profile ([CUDA-JS #139](https://github.com/iteathen/CUDA-JS/issues/139)) | Implemented program semantics expose exact executor needs | Add f64/f16/bf16 pointer/local/cast/arithmetic plus exact abs/special-value math needed by unrelated numerical consumers; no tensor vocabulary upstream. |
| `TENSOR-VALUE-010` | Portable TensorSession/TensorSpec/Tensor value model | `CJS-TENSOR-VIEW-001` for native path | Implemented portable/package profile; zero/one/many shapes, views, limits, ownership, overload equivalence, deletion and cleanup pass. Native promotion remains unclaimed. |
| `TENSOR-PROGRAM-011` | Immutable TensorProgram and finite static TensorPlan | `TENSOR-VALUE-010` | Implemented portable/package profile: first dense inference, declared alias/lifetime/distinct-allocation identity and failure tests pass; execution facts remain explicitly unresolved. |
| `TENSOR-SIMT-012` | ResolvedTensorPlan plus generated Device-JS dense baseline | public views/compiler/operations, `TENSOR-PROGRAM-011`, and achieved `CJS-DEVICE-NUMERIC-005` profile | Exact session/backend/program/workspace/cleanup binding plus independent portable/native mathematical cases pass; no maintained native source. |
| `TENSOR-CUBLASLT-013` | Host accelerated matmul adapter | `CJS-LIB-ADAPTER-004`, `TENSOR-PROGRAM-011` | Exact/tolerance oracle and representative benefit including all overhead. |
| `TENSOR-DEVICE-DENSE-014` | Device-callable compact dense subgraph | `CJS-DEVICE-LIB-002`, `TENSOR-SIMT-012` | Exact eligibility/resource/participation gates and end-to-end evidence; otherwise fallback. |
| `TENSOR-RELEASE-020` | Package consumers, examples, compatibility and release evidence | accepted preceding slices | Clean tarball, unrelated consumers, exact support claims, no private dependency. |

Only one shared CUDA-JS contract changes at a time. Downstream evidence is invalidated when its exact upstream package/revision or contract changes.

## Sequencing

1. Bootstrap and protect the repository.
2. Integrate the four generic CUDA-JS prerequisites and record one exact compatibility pair. Completed at CUDA-JS `2da65ff2` / `0.1.0-alpha.12`.
3. Implement portable value/program semantics over public CUDA-JS only. `TENSOR-VALUE-010` and the pure `TENSOR-PROGRAM-011` static-plan slice are implemented; resolved execution advances under `TENSOR-SIMT-012`.
4. Implement the consumer-neutral CUDA-JS dense numeric Device-JS profile exposed by program-semantics assessment, then the generated SIMT path. The current Device-JS surface has f32 math but lacks f64/f16/bf16 pointer/local arithmetic and exact abs/NaN-profile helpers; CUDA-JS-Tensor may not work around those gaps with generated native source.
5. Add cuBLASLt, then device-callable dense adapters only after exact public CUDA-JS mechanisms and license/profile review.
6. Qualify and publish only the exact achieved surface.

## Backpressure and split rules

Split before a focus branch cannot retain implementation, focused tests, repository validation, exact-head review, cleanup and handoff in one coherent context. Defer extra dtypes, operations, layouts, backends, examples and polish before reducing required semantic/lifecycle evidence.

## Cleanup

Each branch disposes generated artifacts, caches, worktrees, remote branches, drafts and experiments after preserving required evidence. Protected main, user work, source authority and qualification evidence remain protected.
