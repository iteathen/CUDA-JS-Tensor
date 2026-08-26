# SPEC-0009: Item-parallel device-callable Tensor program

**Status:** Accepted implementation profile

**Date:** 2026-08-26

**Parent:** SPEC-0000 through SPEC-0005

**Issue:** #19

**Exact dependency:** `cuda-js@0.1.0-alpha.16` from protected `main@4971302cfb48431c0843126a59d5884d84a81641`, including accepted Device-JS SPEC-0013, selected-runtime target resolution SPEC-0017, typed leaf-library composition SPEC-0028, and dense numeric SPEC-0030

## Outcome

Compile one accepted immutable TensorProgram into one copied typed Device-JS leaf library whose exported function executes exactly one statically independent item. The caller supplies an item index and owns which device participant invokes the function. Many caller participants may invoke different items concurrently without Tensor owning a scheduler, queue, kernel, launch, batch-formation rule, result publication protocol or host progress loop.

This profile supplies a complete device-callable correctness path for finite item-independent dense programs. It does not promise that one item uses a whole warp/block, that every shape is efficient, or that a particular provider/Tensor Core path is selected. A later cooperative profile may accelerate the same mathematics without changing this contract.

Deleting this child leaves ordinary host-planned `ResolvedTensorPlan` execution complete. Deleting any first consumer leaves the callable contract useful for another independent-item Device-JS pipeline.

## Public construction and overloads

The canonical asynchronous constructor is:

```text
compileTensorDeviceProgram(session, TensorPlan, {
  itemCapacity,
  itemInputs,
  output = "ptx",
  maxWorkspaceBytes = 64 MiB
}) -> TensorDeviceProgram
```

Passing a `TensorProgram` is the equivalent convenience overload and normalizes through `TensorPlan.create(program)`. `itemInputs` is an exact nonempty array of distinct TensorProgram input names and normalizes into canonical TensorProgram input order. Meaning-insensitive caller order therefore cannot create a second identity. Unknown fields, duplicate names, missing inputs, unsupported output formats, nonpositive capacity, or invalid workspace ceilings reject before compiler work.

`output` is `"ptx" | "lto-ir"` and selects the existing CUDA-JS SPEC-0028 library artifact family. PTX is the documented default. It is not a performance policy. The chosen family enters compatibility identity and the eventual caller program must use CUDA-JS-compatible homogeneous composition.

## Item model and independence

An **item-varying value** has logical axis 0 equal to `itemCapacity`; each item index names one disjoint logical slice. An **immutable shared input** has no item axis and may be read by every invocation. Only explicitly named program inputs begin item-varying. Every program input, value, node and output is classified during construction.

V1 requires static capacity shapes and rejects `activeAxis0`. Ready occupancy belongs to the caller: it invokes only admitted indices. This avoids converting one current active extent into hidden post-ignition Tensor state.

The accepted propagation rules are:

- `copy`, `cast`, `contiguous` and `unary` preserve an item-varying input;
- `reshape` preserves the item axis only when input and output axis 0 both equal `itemCapacity` and the remaining per-item element count is unchanged;
- `permute` preserves it only when axis 0 remains axis 0;
- `slice` preserves it only when axis 0 is the complete identity slice;
- `broadcast` preserves it only when rank and axis 0 remain unchanged;
- `binary` is item-varying when at least one operand is item-varying and every shared operand is invariant along the aligned output item axis;
- `reduce` preserves it only when axis 0 is not reduced;
- rank-2 `matmul` preserves it when non-transposed A is item-varying by rows and B is shared;
- rank-3 `matmul` preserves its batch axis when at least one rank-3 operand is item-varying and every other rank-3 operand is either item-varying or broadcasts batch dimension one.

Every materialized node and public output must be item-varying. A materialized shared node, cross-item reduction, transposed rank-2 item rows, batch-sensitive operation, ambiguous view transform or shared operand that varies along the aligned item dimension rejects. V1 therefore cannot create concurrent writes to shared material state.

All accepted operation mathematics, scalar conversion, integer width behavior, floating special values, reduction order and matmul accumulation remain SPEC-0004 semantics. Item decomposition changes physical participation only.

## Callable ABI

The library exports one logical Device-JS function named `tensorRunItem`. Its first parameter is `itemIndex: u32`. Remaining parameters are exact typed pointers in this canonical order:

1. TensorProgram inputs in canonical program order;
2. public outputs in canonical output order;
3. one workspace pointer for each used material dtype in canonical Tensor dtype order.

At most 64 total parameters are permitted by CUDA-JS SPEC-0028. Exceeding that bound rejects before compiler work. The returned public parameter records name the semantic role, program input/output/value when applicable, dtype, parameter index, item-varying/shared classification, element/byte bounds and access role. The caller passes matching same-program resident storage; Tensor does not expose or inspect raw pointers.

The function returns `u32`: `0` means the selected item mathematics completed, and `1` means `itemIndex >= itemCapacity`. An out-of-range call performs no input read, workspace write or output write. Other semantic validation is precompiled; runtime CUDA faults remain CUDA-JS failures. Return status is not a request incarnation, freshness, publication or cancellation result.

`TensorDeviceProgram.importAs(alias)` returns a copied CUDA-JS `DeviceJsImport` record for this exact library/export. Alias syntax and collisions remain CUDA-JS-owned. The program exposes the copied public CUDA-JS library record because that record is the required public composition capability; it exposes no generated CUDA, Tensor-generated Device-JS source, raw symbol choice, native handle or provider path.

## Storage and workspace

Inputs retain their accepted TensorSpec shape, dtype, offset, stride and access meaning. An item-varying input selects axis-0 coordinate `itemIndex`; shared inputs use their full declared logical coordinates. Inputs are never mutated.

Each public output is a distinct contiguous item-major pointer binding with `itemCapacity * perItemElementCount` elements. Output aliases and views in the mathematical TensorProgram are copied into these explicit consumer-facing bindings so the callable ABI never asks the consumer to infer TensorPlan internal storage aliasing.

Every internal material is assigned one distinct aligned range inside the workspace partition for its dtype. Each dtype partition contains `itemCapacity` equal per-item regions; invocation `i` touches only region `i`. Views retain exact relative offsets/strides over their owning input or material. The public workspace descriptor declares dtype, parameter, per-item elements/bytes, capacity elements/bytes, alignment, every material range and aggregate high-water bytes.

V1 performs no liveness reuse, hidden allocation, spill, packing allocation or backend workspace. `maxWorkspaceBytes` bounds the sum of all dtype partitions. A zero-workspace program omits all workspace parameters. Result-owned arena reuse and later callable workspace reuse are separate optional policies.

The caller owns allocation and lifetime of input, output and workspace bindings. They must remain valid for every invocation and consumer read. CUDA-JS owns resource capabilities and launch/import lifetimes; this copied library owns no live runtime resource after compilation.

## Generated Device-JS and CUDA-JS boundary

CUDA-JS-Tensor deterministically generates one restricted Device-JS device function and calls public `compileDeviceLibrary()` with one explicit export. Generated source is private implementation data and is not returned. Tensor does not compile CUDA C++, author PTX, deep-import CUDA-JS, choose external native symbols or link the consumer program.

The function uses ordinary finite Device-JS loops and exact scalar/pointer helpers. It does not use thread/block identity, barriers, shared memory, warp primitives, atomics, mailboxes or dynamic allocation. Participation is therefore caller-defined: a thread, a selected lane, or another safe caller participant may invoke it once for one item.

Compilation snapshots the exact plan, item profile, function metadata and output family. Tensor compatibility identity binds the plan identity, item classification, ABI, workspace/output layouts, numeric contract, generator version, CUDA-JS library semantic identity, format and architecture. CUDA-JS artifact/provider/cache identity remains authoritative and separately present in its copied record.

## Consumer obligations and non-ownership

The consumer must:

- bind exact compatible input/output/workspace pointers;
- invoke at most one non-overlapping writer per item/output/workspace region at a time;
- validate its own item/request generation before invocation and before publication;
- provide progress, cancellation, failure routing, output readiness and any scatter/cache semantics;
- keep bound storage resident and valid for the required interval; and
- compose/import/link only through public CUDA-JS contracts.

Tensor owns no evaluator capability, model artifact format, queue, dynamic batch, timeout, scatter table, publication word, cache, root epoch, search policy, graph node, neural head or multi-GPU coordinator. A consumer may layer those meanings over the callable ABI without changing Tensor mathematics.

One TensorSession is bound to one CUDA-JS runtime and therefore one selected device/target. A multi-GPU consumer creates independent sessions and compiles or reuses compatible programs per selected runtime. Tensor does not silently share workspace, artifacts, queues or state across devices; CUDA-JS rejects target-incompatible library composition, and the consumer owns partitioning, replication, coordination and result combination.

## Failure, lifecycle and cleanup

Construction validates and lowers completely before calling CUDA-JS. A classification, range, parameter, workspace or unsupported-operation failure creates no Tensor or CUDA-JS live resource. CUDA-JS compile/link failures retain CUDA-JS error ownership and cannot become a successful fallback.

`TensorDeviceProgram` is an immutable copied program/library descriptor, not a runtime resource and has no `close()`. The TensorSession/runtime used for compilation remains independently owned by its existing lifecycle. A copied library may be retained or discarded like the public CUDA-JS artifact record; compatibility is revalidated by CUDA-JS when a consumer imports it.

Removing this profile removes its public API, generator, contract records and tests. It leaves TensorSession, TensorProgram/TensorPlan, ResolvedTensorPlan, complete SIMT execution, exact fusion and cuBLASLt execution unchanged.

## Evidence and claim limits

Portable evidence must prove strict option normalization, item classification and rejection, deterministic source/library/compatibility identity, parameter ordering/bounds, workspace isolation/formulas, output packing, out-of-range no-write source semantics, independent multi-item mathematics, multiple outputs, shared weights, views, elementwise, non-item reduction and rank-2/rank-3 matmul, exact import records, copied inputs and no production dependency on the superseded experiment frame.

Package evidence must consume only root exports and compose the returned library into an unrelated Device-JS caller. Native evidence must compile/link/load/execute through the exact public installed CUDA-JS pair, independently compare multiple/partial-capacity outputs, and prove graceful zero-live/orphaned resource teardown on the recorded environment.

Passing evidence establishes a device-callable correctness/lifecycle profile only. It does not establish CUDA-MCGS integration, search quality, simultaneous kernel residency, speedup, preferred batch size, tensor-core use, cuBLASDx/CUTLASS, cooperative participation, Linux support, multi-GPU support or a performance-recommended default.

## Falsifiers and forward profiles

Rollback or narrow this child if item independence is guessed, one invocation can write another item's region, output aliasing becomes implicit, generated code needs a host-produced intermediate, a raw/native escape is required, a compile failure leaks ownership, or deletion damages the complete host-planned path.

Later independently removable profiles may add:

- liveness-based callable workspace reuse;
- caller-selected block/warp-cooperative execution using the minimum accepted generic CUDA-JS helpers;
- device-callable provider acceleration such as qualified cuBLASDx or CUTLASS;
- explicit batch-sensitive/cross-item tensor semantics under a different participation contract; and
- multi-session orchestration above one-device Tensor programs.

None is implied by this first profile.
