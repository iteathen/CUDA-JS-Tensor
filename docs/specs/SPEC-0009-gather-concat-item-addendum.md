# SPEC-0009 Addendum: Device-callable static gather and ordered concat

**Status:** Accepted

**Date:** 2026-09-05

**Owner:** `tensor.execution.device-item`

**Parent:** `SPEC-0009-item-parallel-device-callable-tensor-program.md`

**Issue:** #37

**Mathematical dependency:** accepted and protected SPEC-0010 ordinary static `gather` and ordered `concat` semantics.

**Current lower peer:** `cuda-js@0.1.0-alpha.18@30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60` through public CUDA-JS only.

## Outcome

Extend the accepted SPEC-0009 item-parallel device-callable profile with exactly two already-accepted SPEC-0010 operation families when they preserve the existing caller-owned item axis:

1. static indexed `gather` over a non-item axis of one item-varying source; and
2. ordered `concat` over a non-item axis when every input is item-varying.

This addendum owns operation-specific item classification, item-preserving callable lowering obligations, additive compatibility identity, and no-cross-item-write/workspace/output-ABI evidence. SPEC-0010 remains the sole owner of gather/concat mathematics, shape, static indices and input order. CUDA-JS remains the generic Device-JS/compiler/library/runtime owner.

`unary:erf` is outside this child. Protected #52 already implements it through SPEC-0009's existing unary propagation rule without a new item contract.

## Existing item model remains authoritative

An item-varying value has logical axis 0 equal to `itemCapacity`. Invocation `itemIndex = i` owns only logical slice `i` and the corresponding public-output and workspace regions. Shared inputs have no Tensor item axis. V1 continues to reject `activeAxis0`; occupancy remains caller-owned.

Every materialized node and every public output must remain item-varying. This child may not manufacture an item axis for shared data and may not let one invocation read or write another item's owned material/output region.

## Static indexed gather

The ordinary gather record, index validation, output shape, dtype and materialization are inherited unchanged from SPEC-0010.

Device-callable gather is admitted only when:

- the source is already item-varying;
- source and output rank are at least one;
- source and output capacity axis 0 both equal the selected `itemCapacity`;
- `options.axis !== 0`; and
- the finite static `indices` list has already passed SPEC-0010 normalization.

The result remains item-varying. For invocation `i`, every source coordinate retains item coordinate `i`; only the selected nonzero gather axis is replaced by the canonical static index for the output coordinate. Duplicate, nonmonotonic and empty index lists retain their exact ordinary SPEC-0010 meaning independently within each item.

Gather on axis 0 rejects before CUDA-JS compiler work because it would select, duplicate, reorder or omit caller-owned items. A shared gather source also rejects before compiler work because assigning `itemIndex` to a source with no item axis would invent item semantics.

No runtime index tensor, status buffer, table, dynamic bounds protocol or new callable parameter is added. Static indices remain immutable program metadata.

## Ordered concat

The ordinary concat record, dtype/rank/shape compatibility, input order, axis semantics and materialization are inherited unchanged from SPEC-0010.

Device-callable concat is admitted only when:

- every input is already item-varying;
- every input and the output have rank at least one;
- every input and output capacity axis 0 equals the selected `itemCapacity`; and
- `options.axis !== 0`.

The result remains item-varying. For invocation `i`, segment selection partitions only the selected nonzero axis in canonical input order; every selected input coordinate retains item coordinate `i`.

Concat on axis 0 rejects before compiler work because it would merge/change caller-owned items. Any shared concat input rejects in this first child. Ordinary SPEC-0010 concat has no broadcast semantics, so accepting a shared input by treating one coordinate as `itemIndex` would create mathematics/item meaning in the wrong owner.

No segment-table parameter, runtime shape state, global scratch or hidden allocation is introduced. Input order, axis and segment boundaries are immutable plan facts.

## Callable ABI and storage

The SPEC-0009 callable ABI is unchanged:

1. `itemIndex: u32`;
2. TensorProgram input pointers in canonical order;
3. public output pointers in canonical order; and
4. one workspace pointer for each used material dtype in canonical Tensor dtype order.

Gather axes/indices and concat axis/input order are not ABI parameters.

Public outputs remain distinct contiguous item-major bindings. Gather/concat material nodes use the existing dtype-partitioned per-item workspace owner. Invocation `i` may write only region `i`. No liveness-policy change, hidden allocation, spill, packing allocation, provider workspace, cross-item reserve or post-construction capacity growth is authorized.

## Generated Device-JS obligations

The private Tensor generator may use only the finite restricted Device-JS indexing, integer arithmetic, conditionals and loops already available through public CUDA-JS.

For gather, generated source derives source and destination storage from the same invocation item base and replaces only the selected nonzero coordinate with immutable static-index metadata.

For concat, generated source selects the canonical input segment for the output coordinate and derives every selected source offset from the same invocation item base.

Generated source may not add thread/block identity, atomics, barriers, mailboxes, dynamic allocation, host callbacks, runtime index buffers or private/native CUDA paths.

A mathematically valid SPEC-0010 program may still exceed existing Tensor device-item/CUDA-JS source, AST, parameter, workspace or compiler bounds. Such pressure rejects with the owning lower pressure truth; it is never truncated, approximated or split into a host progression loop.

## Compatibility identity and legacy behavior

A device-item profile with no device-callable gather/concat retains exactly:

```text
SPEC-0009-item-parallel-device-tensor-program-v1
```

and retains its existing classification, callable ABI, workspace/output layout, generated source rules and compatibility identity.

A device-item profile containing admitted gather and/or concat selects exactly:

```text
SPEC-0009-item-parallel-device-tensor-program-v1+SPEC-0009-gather-concat-v1
```

The existing device-item compatibility owner binds this contract together with the TensorPlan identity, item classification, ABI, workspace/output layout, output family and public CUDA-JS compatibility facts. No second registry or identity manager is introduced.

`unary:erf` does not select this child; its SPEC-0010 TensorProgram identity and public CUDA-JS dense+erf contract remain separately owned facts.

## Failure and lifecycle

Unsupported item classification rejects before `compileDeviceLibrary()` and creates no CUDA-JS resource. At minimum, these cases reject:

- gather source is shared;
- gather axis is 0;
- gather input/output lose exact item axis 0;
- any concat input is shared;
- concat axis is 0; or
- concat output loses exact item axis 0.

Invalid ordinary gather/concat records remain SPEC-0010 validation failures before this child is considered.

This child owns no live resource. CUDA-JS compilation/resource lifecycle and TensorSession lifecycle remain unchanged.

## Required portable/package falsifiers

Implementation is complete only when permanent evidence proves:

- non-axis-0 gather preserves item classification/capacity and exact duplicate, nonmonotonic and empty-index behavior per item;
- gather axis 0 and shared-source gather reject before compiler work;
- non-axis-0 concat preserves item classification/capacity and exact input order per item;
- concat axis 0 and any shared concat input reject before compiler work;
- generated gather/concat source derives every read/write from the same invocation item base;
- materially different invocation orders cannot change per-item mathematical results;
- `itemIndex >= itemCapacity` retains existing status `1` before any read/write;
- partial occupancy leaves noninvoked output/workspace regions untouched by construction;
- ABI parameter order/count and workspace/output descriptor schemas remain base-SPEC-0009-compatible;
- base programs retain the exact base contract and unchanged generated behavior when this child is unused;
- child programs select the exact additive contract;
- packed public-package compilation composes only through root CUDA-JS-Tensor exports and public CUDA-JS library composition;
- removing this child leaves ordinary Tensor and base SPEC-0009 complete.

Native/provider qualification remains separate and must use an exact installed CUDA-JS/Tensor pair. Portable evidence is not native support evidence.

## Non-goals

Runtime index tensors, gather on item axis 0, dynamic gather failure protocols, shared-source gather materialization, concat on item axis 0, shared-input concat/broadcast semantics, dynamic concat, scatter/update, cross-item reduction/communication, scheduler/batch/publication ownership, model/head/chess/search semantics, native Tensor code, CUDA-JS private imports, cooperative execution or performance claims.
