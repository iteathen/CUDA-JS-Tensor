# SPEC-0009 Addendum: Device-callable static gather and ordered concat

**Status:** Proposal

**Date:** 2026-09-02

**Owner:** `tensor.execution.device-item`

**Parent:** `SPEC-0009-item-parallel-device-callable-tensor-program.md`

**Issue:** #37

**Mathematical dependency:** accepted SPEC-0010 static `gather` / ordered `concat`; protected accepted-state wording is being reconciled separately by PR #36

## Outcome

Extend the accepted SPEC-0009 item-parallel device-callable profile with exactly two SPEC-0010 operation families when they preserve the existing caller-owned item axis:

1. static indexed `gather` over a non-item axis of one item-varying source; and
2. ordered `concat` over a non-item axis when every input is item-varying.

This addendum owns only operation-specific item classification, item-preserving callable lowering obligations, additive compatibility identity, and the associated no-cross-item-write/workspace/output-ABI evidence. It does not restate SPEC-0010 mathematics, add a scheduler, create a new workspace owner, change the callable ABI shape, or add CUDA/provider/product semantics.

`unary:erf` is deliberately outside this child. SPEC-0009 already owns the `unary` propagation rule; `unary:erf` may use that existing rule after CUDA-JS #157 provides the public Device-JS helper and Tensor explicitly implements/qualifies the generator path.

This document is proposal authority only. Current production device-callable compilation must continue rejecting `gather` and `concat` until this child is independently accepted and implemented.

## Assessment and owner boundary

Accepted SPEC-0009 classifies item propagation by operation family. Current `device-item-profile.mjs` has explicit branches for views, copy/cast/contiguous/unary, binary, reduce and matmul. It has no `gather` or `concat` branch, so a new materialized operation without new authority remains non-item and fails closed as shared material. That is the correct current behavior.

The missing facts therefore belong here rather than in SPEC-0010 or CUDA-JS:

- whether a new operation preserves the caller-owned item axis;
- whether shared and item-varying inputs are admissible;
- whether one invocation can read or write another item's logical region;
- whether existing item-major outputs and dtype-partitioned per-item workspace remain sufficient; and
- which compatibility identity proves those item semantics were selected.

SPEC-0010 remains the mathematical/shape/index/order owner. CUDA-JS remains the generic Device-JS compiler/runtime/library owner. Consumers remain the owner of request identity, batching, scatter, publication, model and search meaning.

## Item model remains unchanged

This addendum does not create a second item model.

An item-varying value still has logical axis 0 equal to `itemCapacity`, and invocation `itemIndex = i` owns only logical slice `i` plus the corresponding public-output and workspace regions. Shared inputs still have no Tensor item axis. V1 still rejects `activeAxis0` and leaves ready occupancy to the caller.

Every materialized node and public output must still be item-varying. No operation in this child may turn a shared value into item-varying merely by indexing it with `itemIndex`.

## Static indexed `gather`

The ordinary mathematical record, index validation, capacity shape, dtype, materialization and canonical metadata are owned by SPEC-0010.

The device-callable child admits a `gather` node only when all of the following hold:

- its source value is already item-varying;
- source and output rank are at least one;
- source and output capacity axis 0 equal the exact selected `itemCapacity`;
- `options.axis !== 0`; and
- SPEC-0010 normalization has already validated the finite canonical static index list.

The node remains item-varying. For invocation `i`, every output coordinate uses source item coordinate `i`; only the selected nonzero gather axis is replaced by the corresponding canonical static index. Duplicate, nonmonotonic and empty index lists retain their exact SPEC-0010 meaning independently inside each item.

Gathering axis 0 rejects before CUDA-JS compiler work because it would select, duplicate, reorder or omit caller-owned items and therefore redefine SPEC-0009 participation.

A shared source also rejects before compiler work. Materializing a shared gather result is outside the first SPEC-0009 profile, and treating a shared source coordinate as `itemIndex` would invent an item axis not present in the source contract.

No runtime index tensor, bounds status, item-generation lookup, global table or extra callable parameter is introduced. The static index list is immutable program metadata.

## Ordered `concat`

The ordinary mathematical record, input order, shape compatibility, dtype, materialization and canonical axis semantics are owned by SPEC-0010.

The device-callable child admits a `concat` node only when all of the following hold:

- every input value is already item-varying;
- every input and output rank are at least one;
- every input and output capacity axis 0 equal the exact selected `itemCapacity`; and
- `options.axis !== 0`.

The node remains item-varying. For invocation `i`, concatenation partitions only the selected nonzero axis in exact canonical input order while every source coordinate keeps item coordinate `i`.

Concat on axis 0 rejects before compiler work because it would merge/change the caller-owned item axis.

Any shared concat input rejects in this first child. Ordinary SPEC-0010 concat has no broadcast semantics. Allowing a shared input with an item-varying concat by interpreting one of its coordinates as `itemIndex` would silently manufacture a new item axis; accepting a rank/shape special case would create new concat mathematics in the wrong owner. A later profile may add shared-input behavior only if ordinary Tensor semantics and the SPEC-0009 item model support it without such reinterpretation.

No segment table, runtime shape state, global scratch or extra callable parameter is introduced. Input order, axis and segment boundaries are immutable plan facts.

## Callable ABI and storage

The accepted SPEC-0009 ABI remains structurally unchanged:

1. `itemIndex: u32`;
2. canonical TensorProgram input pointers;
3. canonical public output pointers; and
4. one workspace pointer for each used material dtype.

Axes, gather indices and concat input order do not become ABI parameters.

Public outputs remain distinct contiguous item-major bindings. Materialized gather/concat nodes use the existing dtype-partitioned workspace owner. Each dtype partition still contains `itemCapacity` equal per-item regions and invocation `i` may write only region `i`.

No hidden allocation, spill, packing allocation, cross-item reserve, provider workspace or post-ignition capacity growth is authorized.

## Generated Device-JS obligations

The private Tensor generator may realize the accepted mathematics with ordinary finite restricted Device-JS indexing, arithmetic, conditionals and loops already owned by the base profile.

For gather, generated source must derive the source offset from the same invocation item base as the output and replace only the selected nonzero axis with the compile-time canonical index.

For concat, generated source must choose the canonical input segment for the output coordinate and derive every selected input offset from the same invocation item base.

Generated code may not use thread/block identity, atomics, barriers, mailboxes, dynamic allocation, host callbacks, runtime index buffers or CUDA/private source escape paths merely to support these operations.

A semantically valid SPEC-0010 record may still exceed existing SPEC-0005 / CUDA-JS Device-JS source, AST, binding, parameter, logical-work or workspace limits. Such a record rejects with the owning resolved-pressure truth before execution; it is never truncated, approximated, split by a host progression loop or backed by hidden runtime state.

## Compatibility identity and legacy behavior

Base SPEC-0009 programs that contain no device-callable `gather` or `concat` retain exactly:

```text
SPEC-0009-item-parallel-device-tensor-program-v1
```

and retain their existing item classification, callable ABI, workspace/output layout, generator behavior and compatibility identity.

A TensorDeviceProgram whose accepted plan uses device-callable `gather` or `concat` selects exactly one additive item-semantics child:

```text
SPEC-0009-item-parallel-device-tensor-program-v1
+SPEC-0009-gather-concat-v1
```

The existing TensorDeviceProgram compatibility owner binds this child together with plan identity, item classification, ABI, workspace/output layout, generator version and copied CUDA-JS library identity. No second compatibility registry is introduced.

`unary:erf` does not select this child merely because it is a SPEC-0010 operation; its item propagation remains the existing SPEC-0009 unary rule. Its TensorProgram plan identity and CUDA-JS numeric/library contract provide their separately owned compatibility facts.

## Failure and cleanup

Unsupported item classification rejects before `compileDeviceLibrary()` and creates no CUDA-JS live resource.

At minimum, these conditions are typed unsupported failures:

- gather source is shared;
- gather axis is 0;
- gather output does not retain exact item axis 0;
- any concat input is shared;
- concat axis is 0; or
- concat output does not retain exact item axis 0.

Existing invalid SPEC-0010 mathematical records remain validation failures in their mathematical owner before this child is considered.

The child owns no live resource. Compilation failures, copied library lifecycle and runtime resource cleanup remain exactly with CUDA-JS and the base SPEC-0009/TensorSession owners.

## Required portable falsifiers

Before implementation may be called complete, permanent evidence must prove at least:

### Gather

- non-axis-0 gather of an item-varying source preserves item classification and exact `itemCapacity`;
- axis-0 gather rejects before compiler work;
- shared-source gather rejects before compiler work;
- duplicate and nonmonotonic static indices produce exact independent per-item results;
- empty static indices retain exact zero-sized-axis semantics without cross-item access;
- a mutation that uses a constant/neighbor item base is killed by no-cross-item-read/write evidence.

### Concat

- non-axis-0 concat of item-varying inputs preserves item classification and exact `itemCapacity`;
- axis-0 concat rejects before compiler work;
- any shared concat input rejects before compiler work;
- input order is preserved independently inside every item;
- a mutation that chooses a segment from one item and reads another item's base is killed.

### Cross-operation / ABI / lifecycle

- multiple item indices executed in materially different orders produce identical per-item mathematical outputs;
- two or more item invocations cannot write one another's public-output/workspace region under the declared owner contract;
- `itemIndex >= itemCapacity` retains the existing status `1` and zero-input-read/zero-output-write/zero-workspace-write behavior;
- partial occupancy leaves noninvoked item output/workspace regions unchanged;
- parameter order/count and workspace/output descriptor schemas remain base-SPEC-0009-compatible;
- base programs retain exact pre-child contract/identity/generated behavior;
- child programs select the exact additive contract and cannot be rehydrated/treated as base-only compatibility;
- removing this child leaves host-planned Tensor and base SPEC-0009 complete;
- first-consumer deletion leaves no model/chess/search vocabulary or special mapping in production owners.

## Native/package evidence

Package evidence must consume root exports and compose the returned TensorDeviceProgram only through the public CUDA-JS library mechanism. Native promotion later requires an exact installed-package CUDA-JS/Tensor pair and multiple/partial item execution showing independent mathematical parity and terminal resource cleanup on the recorded environment.

No portable/reference result is native support evidence, and no native result is CUDA-MCGS/Vector readiness by itself.

## Acceptance gate

This proposal may become implementation authority only after review confirms:

1. the item axis remains solely caller-owned SPEC-0009 axis 0;
2. gather cannot select/reorder across items;
3. concat cannot merge/change the item axis;
4. shared inputs are rejected wherever admitting them would invent item semantics absent from ordinary Tensor mathematics;
5. no new ABI parameter, workspace owner, hidden runtime table or host progression loop is required;
6. additive compatibility leaves base programs exact;
7. current implementation remains fail-closed until explicit child support is added; and
8. the profile remains useful without the first model/MCGS consumer.

## Non-goals

Runtime index tensors, gather on the item axis, dynamic gather failure protocols, shared-source materialization, concat on the item axis, shared-input concat/broadcast semantics, dynamic concat, scatter/update, cross-item reductions or communication, scheduler/batch/publication ownership, model/head/chess/search semantics, native Tensor code, CUDA-JS private imports, cooperative execution or performance claims.
