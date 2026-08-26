# SPEC-0001: Tensor session, specification and value model

**Status:** Accepted
**Date:** 2026-08-26

## TensorSession

All overloads normalize to one canonical open record:

```text
TensorSession.open()
TensorSession.open({ device, limits, defaults })
TensorSession.open(cudaRuntime)
TensorSession.open({ runtime: cudaRuntime, limits, defaults })
```

The no-argument form opens one compiler-enabled CUDA-JS runtime with conservative documented limits so the complete SIMT baseline can remain available in the same session epoch. A selector chooses one runtime/device before tensor allocation. An injected runtime is borrowed by default; `{ runtime, runtimeOwnership: "owned" }` explicitly transfers runtime-close authority. Ownership and compiler mode are inspectable in the canonical open record; later plan resolution rejects a borrowed compiler-disabled runtime when its selected backend requires compilation.

V1 contains exactly one runtime/device epoch. A tensor is usable only by its issuing live session. Closing a session closes plans, programs, views and allocations in dependency order, then closes an owned runtime. A borrowed runtime is not closed, but all tensor-owned resources must be terminal before session close succeeds.

## TensorSpec

A canonical specification contains:

- closed dtype identifier supplied by an accepted public CUDA-JS dtype registry;
- rank from 0 through 16;
- `capacityShape`, each dimension a safe nonnegative integer;
- optional `activeAxis0` with `0 <= extent <= capacityShape[0]` and a declared maximum equal to capacity axis 0;
- element strides, row-major contiguous by default;
- byte offset and required alignment;
- mutability/access class and alias group when applicable.

Rank zero is scalar. A zero dimension is valid and yields zero elements. The product and byte range must remain safe, bounded and within session limits. Strides are zero or positive in v1; zero-stride broadcast views are read-only unless a later exact write/conflict contract permits them. Negative strides reject.

The active extent changes only the logical axis-0 length inside the fixed capacity/storage/plan bound. It cannot reallocate, change rank, change other dimensions, or invalidate compiled layout identity.

## Tensor capability

`Tensor` is an opaque session-owned capability over an accepted public CUDA-JS typed view/allocation. Public products expose dtype, capacity/logical shape, strides, offset, byte length, access, state and compatibility identity—never a native pointer or CUDA-JS private token.

Views are child capabilities. A parent allocation cannot close while a live view or operation lease exists. Exact overlap is computed from finite byte regions and stride semantics; planners must reject alias patterns a selected backend cannot preserve.

The first implemented value profile admits changed child views from a contiguous parent when the child's exact byte envelope stays inside the parent. A strided parent admits only an identical child specification until a bounded exact reachable-element-subset contract is accepted. Envelope overlap alone is not treated as proof for two arbitrary strided sets.

`Tensor.write(bytes)` and `Tensor.read()` copy exactly the logical byte sequence of a contiguous non-broadcast tensor and enforce its access role. They snapshot bytes in both directions, perform no implicit dtype conversion, and treat empty logical tensors as explicit zero-byte transfers. Strided host gather/scatter is outside v1 and rejects rather than guessing an order.

## Defaults

Row-major contiguous storage, output allocation, and safe accumulator choices may be inferred. Every inferred fact appears in the resolved plan. Defaults never infer neural/model meaning, cross-device movement, unsafe in-place writes, reduced precision, or a performance claim.

## Implemented portable profile

`cuda-js-tensor@0.1.0-alpha.3` retains the `TensorSpec.create(...)`, `TensorSession.open(...)`, root allocation, bounded child views, exact session accounting, cross-session rejection, owned/borrowed runtime close, and package-internal planner inspection implemented in the earlier value slices, and adds the exact copied-byte transfer port needed by public execution consumers.

The documented session defaults are `f32`, `read-write`, row-major storage, 128 MiB per physical allocation, 256 MiB of session-owned allocation capacity, and 1,024 live tensor capabilities. V1 required alignment equals dtype width because the public CUDA-JS view contract provides no stronger portable identity. Empty tensors use a minimal dtype-width physical allocation while retaining a zero-element logical/view range; the physical allocation is inspectable and counted.

Portable mock and installed-package tests prove normalization, bounds, ownership and orchestration only. No CUDA-JS-Tensor native or performance claim follows.
