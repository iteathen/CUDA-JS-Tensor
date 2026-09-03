# SPEC-0006 Addendum: CUDA-JS provider ownership correction

**Status:** Accepted implementation correction

**Date:** 2026-09-03

**Parent:** accepted `SPEC-0006-host-planned-cublaslt-matmul.md`

**Issue owner:** #44

**Exact lower dependency:** CUDA-JS protected `main@85d92d4a04385b0edbc7a19c2bce3d256642bf2f`, including accepted SPEC-0029 capability-projection and borrower-lifecycle addenda

## Outcome

Correct only the lower CUDA-JS ownership assumptions in SPEC-0006. Tensor mathematics, eligibility, `simt | prefer-cublaslt | cublaslt` policy, deterministic SIMT fallback, Tensor-owned workspace budgets/count limits, prepared-node placement, run semantics and result ownership remain unchanged.

This addendum supersedes SPEC-0006 only where it previously assigned native-provider sharing, concrete provider error-code interpretation or provider workspace alignment to CUDA-JS-Tensor.

## Provider lifecycle

CUDA-JS-Tensor no longer owns a session/runtime cuBLASLt singleton, `WeakMap`, reference count, last-user close rule or failed-close quarantine.

Each resolved Tensor backend that attempts cuBLASLt realization obtains its own public `CudaCublasLt` capability from `runtime.openCublasLt()`. Under the exact dependency, that public capability is a CUDA-JS-owned borrower over the one admitted runtime-owned cuBLASLt provider resource.

The Tensor backend owns only its borrower capability and its fixed Tensor-selected plans:

```text
Tensor resolved backend
  -> one public CudaCublasLt borrower
     -> zero or more fixed Tensor-selected CudaCublasLtMatmulPlan children
```

Tensor closes its plans before its borrower. CUDA-JS decides whether closing that borrower also closes the underlying native provider. Independent Tensor backends sharing one runtime do not coordinate each other's borrower lifetime.

No Tensor state may determine when the underlying provider handle is created, retained, destroyed, quarantined or reacquired.

## Lower capability facts

The accepted CUDA-JS provider record is authoritative for provider-owned workspace alignment:

```text
adapter.provider.workspaceAlignmentBytes
```

When a selected CUDA-JS plan requires workspace, Tensor records that exact positive safe-integer alignment in its resolved workspace allocation requirement. The lower value is therefore visible through the resolved provider/workspace projection but is not copied into Tensor source as a CUDA/provider constant.

The pre-provider Tensor backend request no longer contains `alignment.workspaceByteOffset: 256`. Its retained `alignment.operandByteOffset: 4` is independently Tensor-owned because the accepted accelerated profile admits only contiguous `f32` Tensor operands and Tensor owns their semantic byte-offset validity.

Removing the copied workspace-alignment field intentionally changes the backend-profile request compatibility identity. The resolved backend identity also changes because the lower public provider identity now includes `workspaceAlignmentBytes`. These are accepted prerelease ownership corrections, not mathematical Tensor changes.

## Preference fallback classification

`prefer-cublaslt` remains a Tensor semantic policy. The policy no longer branches on provider-family implementation codes such as `CUBLASLT_*`.

A lower CUDA-JS failure encountered while opening the optional cuBLASLt borrower or creating an otherwise Tensor-eligible plan is an ordinary preference fallback exactly when:

```text
error.category === "unsupported"
```

This includes lower-admitted provider/profile absence or incompatibility and lack of a supported candidate under the bounded request. Tensor records stable Tensor-owned reasons rather than raw lower error-code strings:

- `provider-unsupported` for borrower/provider admission failure;
- `plan-unsupported` for lower plan/candidate rejection.

Tensor-local accelerator-only pressure remains independently owned. The existing bounded binding-count and workspace-count gates remain preference fallbacks with stable Tensor reasons:

- `binding-limit`;
- `workspace-count-limit`.

A strict `cublaslt` request never converts these failures to SIMT.

Lower categories other than `unsupported` remain hard failures at this boundary, including validation, backpressure, provider-contract failure, stale/closed ownership, deferred/immediate driver failure, restart-required, permission and internal failure. Cleanup failure is never converted into a successful fallback.

## What does not change

This addendum does not move any of the following into CUDA-JS:

- TensorProgram/TensorPlan meaning;
- rank/dtype/layout/stride/active-extent eligibility;
- `simt | prefer-cublaslt | cublaslt` policy;
- Tensor workspace byte/count/session ceilings;
- Tensor prepared binding-count pressure;
- Tensor semantic fallback recording;
- plan/result/material ownership;
- mathematical/reference evidence.

It also does not authorize any new native code, provider API, performance policy, hidden workspace, operation family, scheduler, launch resolver or preparation transaction.

## Required evidence

Portable and package qualification must prove at least:

- production Tensor code contains no `SHARED_CUBLASLT` or equivalent provider-lifetime singleton/refcount;
- two resolved Tensor backends on one runtime obtain and close independent public borrowers without coordinating underlying-provider lifetime;
- a Tensor backend closes its plans before its own borrower;
- ordinary preference fallback depends on public `category === 'unsupported'`, not provider-family codes;
- non-`unsupported` lower failures remain hard;
- Tensor-local binding/workspace-count pressure still falls back only under `prefer-cublaslt`;
- selected workspace records derive `requiredByteOffsetAlignment` from the public provider capability;
- the backend-profile request identity no longer contains copied provider workspace alignment;
- the resolved provider identity contains the lower public alignment fact;
- strict acceleration, numerical/reference fixtures, cleanup and package/public-import tests remain green;
- the exact CUDA-JS dependency is the protected integrated borrower/capability revision named above.

Native numerical/provider evidence is not upgraded by this ownership refactor. Existing native claims remain bounded to their recorded compatible pair until rerun against the new exact dependency if a new native support claim is needed.

## Falsifiers

Rollback this correction if Tensor still needs to own underlying provider destruction, match provider-family codes for ordinary preference, hard-code a CUDA/provider workspace requirement, deep-import CUDA-JS, or add a second native/provider lifecycle.

If the public lower seam cannot preserve Tensor's mathematical and semantic policy without one of those behaviors, route the missing consumer-neutral capability back to CUDA-JS rather than restoring a Tensor-local workaround.

## Non-goals

No executable-preparation generalization (#45/#40 follow-up), no launch-topology refactor, no new accelerator family, no performance tuning, no NN/search semantics, no native source, and no CUDA-MCGS #122 work.
