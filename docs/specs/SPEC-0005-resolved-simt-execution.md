# SPEC-0005: Resolved SIMT execution

**Status:** Accepted implementation profile
**Date:** 2026-08-26
**Parent:** SPEC-0000 through SPEC-0004
**Issue:** #6
**Exact dependency:** `cuda-js@0.1.0-alpha.14` from protected `main@fb27296cffd7191180b0e3cd609224ed2ded182e`

## Outcome

`ResolvedTensorPlan` binds one immutable `TensorPlan` to one live `TensorSession`, one complete generated Device-JS SIMT realization, exact compiler/module/prepared-DAG identities, finite workspace, and one cleanup graph. A `TensorProgram` convenience input first normalizes through `TensorPlan.create`; it does not create different execution semantics.

Canonical resolution and equivalent overloads are:

```text
resolveTensorPlan(session, tensorPlan, options?)
resolveTensorPlan(session, tensorProgram, options?)
ResolvedTensorPlan.create(session, tensorPlanOrProgram, options?)
```

The first options are exact: `backend: "simt"`, a block size from 32/64/128/256/512/1024 with default 256, and a positive workspace ceiling no greater than 64 MiB. Defaults appear in immutable resolved identity.

## Complete first SIMT realization

Every material operation admitted by SPEC-0004 lowers to deterministic restricted Device-JS submitted only through public CUDA-JS contracts. View nodes change indexing/alias facts and emit no kernel. Fill, copy, cast, contiguous materialization, unary/binary elementwise work and rank-2/rank-3 matmul use one logical work item per output element. Matmul accumulates in the declared accumulator dtype and rounds once to the output dtype.

Reduction uses the declared identity, row-major reduced-coordinate order, power-of-two identity padding and an explicit balanced adjacent-pair tree. Gather and tree stages use one finite per-run workspace; no sequential shortcut, host callback or hidden backend order substitutes for `fixed-tree-v1`. A `backend-defined` program may use the same generated tree, but retains its weaker program-level determinism claim and a distinct compatibility identity.

The generated source is private implementation. Public records expose semantic identities and bounded descriptors, never source, CUDA syntax, native handles, paths, pointers, streams, events, modules, functions or actor tokens. Maintained `.cu` and `.ptx` remain forbidden.

## Bounds and prepared execution

The first resolved profile permits at most 32 generated kernels, 64 named device bindings, `2^32-1` logical work items per kernel, and 64 MiB total reduction workspace selected no higher than both the resolve option and session limits. Resolution rejects a plan whose distinct materials and workspace cannot fit an otherwise empty session or whose individual allocation exceeds the per-tensor limit before compiler work. Each run then enforces the session's current live-tensor and remaining-byte capacity while allocating its distinct resources, with rollback on pressure failure.

Nonempty generated kernels compile as one public Device-JS program, load as one CUDA-JS module, resolve one function capability per kernel, and form one immutable public SPEC-0020 semantic prepared DAG. Canonical dependencies serialize generated stages on one private CUDA-JS stream and one submitted DAG returns one ordinary CUDA-JS operation. Empty logical work resolves explicitly without compiler, module or prepared resources.

## Runtime bindings and results

`run()` accepts an exact named record, positional input array, or the sole `Tensor` directly. A no-input program accepts omitted or empty bindings. Every input must be a live capability from the resolved session with the exact program specification. Declared alias-group equality/inequality is verified against runtime storage identity before allocation or submission.

One resolved plan admits one run at a time. Each run allocates distinct material values and reduction workspaces; inputs remain borrowed. On completion, `TensorExecutionResult` owns all run-created material tensors, workspaces and derived output views. `outputs` is the exact named record, `output` is non-null only for one output, and `get(name)` is the exact lookup. Closing a result closes its owned graph in dependency order but never closes inputs. A resolved plan owns its live result capabilities for cascade cleanup, and the session owns its resolved plans; therefore easy top-level session close reaches every Tensor-created resource before the borrowed/owned runtime boundary. Live user-created child views cause retryable backpressure; unproved cleanup becomes explicit and non-graceful.

`Tensor.write(bytes)` and `Tensor.read()` are the public seeding/observation port. They copy exactly the logical bytes of a live contiguous non-broadcast tensor, enforce access authority, snapshot caller/result storage, and perform no dtype conversion. Empty transfers are explicit zero-byte successes. Strided transfer requests reject until a separate gather/scatter contract is accepted.

## Lifecycle and failure

The resolved plan owns compiler output consumption, module, function and prepared-DAG capabilities. Each submitted CUDA-JS operation is awaited and closed before `run()` returns. A primary execution failure plus operation-cleanup failure is retained as one cleanup-unproved error. Resolution/run rollback closes every created resource in reverse dependency order; cleanup failure never masquerades as the primary error or a graceful result.

`ResolvedTensorPlan.close()` is backpressured by an active run, then closes prepared DAG, functions and module. Results must close before their session. Session terminal state remains the aggregate owner of any resource whose cleanup cannot be proved.

## Non-goals and promotion limits

This profile does not implement cuBLASLt selection, fusion, arena reuse, multi-run concurrency, multi-device tensors, P2P, collectives, device-callable subgraphs, dynamic shapes beyond the accepted active-axis contract, tensor cores, fast math or performance policy. `the_restaurant`, neural-network, training and MCGS semantics remain outside the component.

Portable mocks prove normalization and lifecycle only. Native promotion requires an installed public package, independent expected mathematics covering every operation family and dtype boundary relied upon by the claim, exact CUDA-JS/provider/device identity, replay, failure controls, and zero-live/orphaned terminal cleanup. A pass does not establish performance benefit or broader hardware/platform support.

The alpha.3 implementation integrated through PR #7 at `main@a2c9d4a022e016c5ec249b38cdc43dc398309a37` passes the full portable/package gate and an unrelated installed-package fixture on exact Node 26.7.0 / Windows CUDA 13.3 / compute_75 / GTX 1660 Ti. The fixture replays one 10-kernel prepared DAG twice and checks view/material operations, copied transfers, f64-to-i32 NaN/infinity saturation, f16/bf16 arithmetic, adversarial fixed-tree f32 reduction, rank-2 matmul, session cascade ownership and zero live/orphaned CUDA-JS resources. This qualifies only those semantics on that recorded profile; it is not performance, tensor-core, accelerator, Linux, broader-GPU or production-stability evidence.
