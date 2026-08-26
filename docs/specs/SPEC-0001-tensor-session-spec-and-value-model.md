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

The no-argument form opens one default CUDA-JS runtime with conservative documented limits. A selector chooses one runtime/device before tensor allocation. An injected runtime remains caller-created but transfers tensor-child lifecycle authority to the session only as explicitly declared by the canonical record; ownership mode is inspectable.

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

## Defaults

Row-major contiguous storage, output allocation, and safe accumulator choices may be inferred. Every inferred fact appears in the resolved plan. Defaults never infer neural/model meaning, cross-device movement, unsafe in-place writes, reduced precision, or a performance claim.
