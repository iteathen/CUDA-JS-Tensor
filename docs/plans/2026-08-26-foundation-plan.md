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
| `CJS-TENSOR-VIEW-001` | Upstream public typed device-view/dtype facade | Accepted CUDA-JS SPEC-0021 reassessment | Installed consumer can create/use/close a typed view without deep import. |
| `CJS-DEVICE-LIB-002` | Upstream typed device-callable import/library composition | RDC/LTO and resource/participation contracts | Two unrelated consumers compose typed device functions; lifecycle and divergence fail closed. |
| `CJS-PREPARED-DAG-003` | Upstream prepared finite operation DAG baseline | Accepted operation/transfer lifecycle | Immutable finite DAG preserves dependencies, errors and terminal cleanup. |
| `CJS-LIB-ADAPTER-004` | Upstream context-bound library framework and cuBLASLt profile | Public typed views + operations | Generic lifecycle first; exact independent GEMM oracle; no tensor semantics in CUDA-JS. |
| `TENSOR-VALUE-010` | Portable TensorSession/TensorSpec/Tensor value model | `CJS-TENSOR-VIEW-001` for native path | Zero/one/many shapes, views, limits, ownership, overload equivalence and deletion pass. |
| `TENSOR-PROGRAM-011` | TensorProgram/TensorPlan/ResolvedTensorPlan | `TENSOR-VALUE-010`; DAG port may remain injected mock | Dense inference, alias/lifetime/workspace, identity and failure tests pass. |
| `TENSOR-SIMT-012` | Generated Device-JS dense baseline | public views/compiler/operations | Independent portable/native mathematical cases and cleanup pass; no maintained native source. |
| `TENSOR-CUBLASLT-013` | Host accelerated matmul adapter | `CJS-LIB-ADAPTER-004`, `TENSOR-PROGRAM-011` | Exact/tolerance oracle and representative benefit including all overhead. |
| `TENSOR-DEVICE-DENSE-014` | Device-callable compact dense subgraph | `CJS-DEVICE-LIB-002`, `TENSOR-SIMT-012` | Exact eligibility/resource/participation gates and end-to-end evidence; otherwise fallback. |
| `TENSOR-RELEASE-020` | Package consumers, examples, compatibility and release evidence | accepted preceding slices | Clean tarball, unrelated consumers, exact support claims, no private dependency. |

Only one shared CUDA-JS contract changes at a time. Downstream evidence is invalidated when its exact upstream package/revision or contract changes.

## Sequencing

1. Bootstrap and protect the repository.
2. Open separate CUDA-JS issues/specification packets for the four generic prerequisites.
3. Implement portable value/program semantics while native dependencies remain injected or blocked.
4. Implement the generated SIMT path after public typed views are usable.
5. Add cuBLASLt, then device-callable dense adapters only after exact public CUDA-JS mechanisms and license/profile review.
6. Qualify and publish only the exact achieved surface.

## Backpressure and split rules

Split before a focus branch cannot retain implementation, focused tests, repository validation, exact-head review, cleanup and handoff in one coherent context. Defer extra dtypes, operations, layouts, backends, examples and polish before reducing required semantic/lifecycle evidence.

## Cleanup

Each branch disposes generated artifacts, caches, worktrees, remote branches, drafts and experiments after preserving required evidence. Protected main, user work, source authority and qualification evidence remain protected.
