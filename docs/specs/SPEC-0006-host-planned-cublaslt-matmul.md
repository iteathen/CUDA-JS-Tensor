# SPEC-0006: Host-planned cuBLASLt f32 matmul realization

**Status:** Accepted implementation profile
**Date:** 2026-08-26
**Parent:** SPEC-0003 through SPEC-0005
**Issue:** #8
**Exact dependency:** `cuda-js@0.1.0-alpha.15` from protected `main@af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d`; implementation entered protected main at `20f831cc51766aee726313f7f78819b576d56307`

## Outcome

Permit one immutable resolved tensor plan to replace eligible generated SIMT matmul kernels with public CUDA-JS SPEC-0031 `cublaslt-f32-matmul` nodes. Generated kernels and library nodes remain in one public prepared operation DAG, one CUDA-JS-owned private stream, one final event and one ordinary whole-DAG operation. CUDA-JS-Tensor adds no scheduler, native boundary, host wait between nodes or provider control.

The complete SPEC-0005 SIMT lowering remains available for every SPEC-0004 program. Accelerator removal therefore deletes only an optional realization and cannot remove mathematical coverage.

## Resolution policy and overloads

`ResolvedTensorPlan.create` and `resolveTensorPlan` retain their existing overloads and add one exact backend policy:

```text
backend = "simt" | "prefer-cublaslt" | "cublaslt"
```

`simt` remains the documented convenience default until representative total-cost evidence justifies changing it. `prefer-cublaslt` is an explicit preference: structurally ineligible matmuls and exact unavailable-profile/algorithm outcomes fall back to SIMT with recorded reasons. `cublaslt` is strict: every nonempty matmul must pass the profile and produce a public plan or resolution rejects. Other operation families continue to use SIMT in either accelerated policy.

Selection happens once during resolution. No run-time shape probe, heuristic selection, retry or host callback enters `run()` or the GPU hot path. The resolved descriptor records the requested policy, aggregate realization (`simt`, `cublaslt`, or `mixed`), every executable-node backend, every matmul eligibility/fallback reason, provider profile/identity, public plan facts and workspace choice.

## Exact v1 eligibility

A matmul may select the accelerator only when all of these facts are statically true:

- both inputs and the output are rank 2 with nonzero `m`, `n` and `k`;
- input, accumulator and output dtype are exactly `f32`;
- both logical operands and the output are row-major contiguous;
- neither input has an active extent, a broadcast stride or a derived byte displacement relative to its bound CUDA-JS view;
- `transposeA` and `transposeB` are the only layout transformations and enter the public plan unchanged;
- the exact public CUDA-JS adapter/provider profile admits the plan on the selected session/runtime/device;
- selected workspace plus SIMT reduction workspace stays within the resolved plan ceiling, each workspace fits the session tensor limit, the entire empty-session plan fits the session byte limit and the prepared binding ceiling remains satisfied.

Rank-3/batched matmul, f16/bf16/f64, mixed precision, arbitrary strides, active dimensions, epilogues, bias and hidden packing are explicit SIMT fallbacks or strict-request rejections. No cast, copy, padding or transpose materialization is inserted to manufacture eligibility.

## Composition and memory

The deterministic full SIMT lowering is constructed first as the complete fallback and identity anchor. For each selected matmul its one generated kernel is replaced at the same execution-node ID and dependency position by one fixed CUDA-JS SPEC-0029 plan. A and B bind the existing logical operand storage. C and D bind the already planned read-write output storage with `beta = 0`, so acceleration creates no extra result tensor and preserves SPEC-0004 distinct material ownership.

Each nonzero CUDA-JS-selected workspace is one explicit per-run read-write tensor allocation at byte offset zero. Workspace ceilings are consumed deterministically in program order; no hidden allocation, reuse arena or per-run heuristic exists. CUDA-JS derives library-node access ranges, leases plans/views/workspaces, checks hazards and owns enqueue/completion/failure semantics.

The tensor session/runtime has one shared public cuBLASLt adapter lease. Resolved plans own only their fixed matmul plans and prepared-DAG references; the last lease closes the adapter after all child plans close. Results continue to own all run-created material and workspace tensors and never own borrowed inputs.

## Failure and fallback

Only exact provider/profile absence, an already caller-occupied optional adapter, exact no-algorithm outcomes and bounded accelerator-only resource gates are eligible preference fallbacks. Provider identity, ABI, lifecycle, cleanup, stale-resource, cross-runtime, malformed-contract and unexpected execution failures remain hard failures. A strict request never silently falls back.

Preparation, plan creation and session-registration rollback close the prepared DAG, functions, module, plans and adapter lease in dependency order. Cleanup failure is reported as cleanup-unproved and never converted into a successful SIMT fallback.

## Evidence and promotion limits

Portable tests prove deterministic selection, overload equivalence, strict rejection, preference fallback, mixed prepared-node construction, shared-adapter ownership, workspace accounting and cleanup orchestration. Native evidence must use installed public packages and compare accelerator output independently with SPEC-0004 mathematics while proving replay and zero live/orphaned resources on the exact recorded provider/device profile.

Passing equivalence does not prove a speedup, tensor-core use, CUDA Graph execution, Linux support, broader CUDA/provider/device compatibility, batched GEMM, multi-GPU behavior or a recommended default. A default-policy promotion requires separate representative end-to-end measurements including planning, provider open, plan creation, workspace, warmup, submission, synchronization, small/tail cases, latency distribution, memory high-water and the best credible SIMT baseline.

The alpha.4 implementation integrated through protected PR #10 at `main@8910309a0aff9b8da4fc281949068d8d1fcaa6ea`. It passes the full portable/package gate and an unrelated installed-package native fixture on exact Node 26.7.0 / Windows CUDA 13.3 / compute_75 / GTX 1660 Ti / cuBLASLt 13.5.1. Native evidence replays a mixed kernel → cuBLASLt → kernel DAG twice, checks strict double-transpose output independently, retains the complete SIMT catalog fixture and reaches zero live/orphaned CUDA-JS resources. This is bounded correctness/lifecycle evidence only.

## Falsifiers and non-goals

Rollback this child if it needs private CUDA-JS imports, native CUDA/PTX source, a second scheduler, host advancement between prepared nodes, hidden data transformation, unbounded workspace, unleased resources or weakened SIMT semantics.

Device-callable cuBLASDx/CUTLASS, convolution, unrelated fusion, arena reuse, multi-device/P2P/collectives, NN/training/search policy, package publication and `the_restaurant` mutation remain outside this profile. The retained Restaurant plan remains documentation only.
