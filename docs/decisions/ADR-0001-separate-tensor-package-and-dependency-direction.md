# ADR-0001: Separate tensor package and dependency direction

**Status:** Accepted
**Date:** 2026-08-26

## Context

Broad tensor operations have an independent public audience, release cadence, dependency surface, conformance matrix, accelerator licensing burden, and lifecycle from CUDA-JS core, CUDA-MCGS, or one training system. Putting them in CUDA-JS would turn a native runtime into a numerical framework; putting them in CUDA-MCGS or a neural project would encode the first consumer.

## Decision

Create the public `iteathen/CUDA-JS-Tensor` repository and future npm package `cuda-js-tensor`.

Dependency direction is:

```text
consumer -> cuda-js-tensor -> public cuda-js
```

CUDA-JS-Tensor never deep-imports CUDA-JS, adds direct native CUDA access, or requires CUDA-JS to depend on tensor semantics. Missing generic mechanisms are specified upstream as consumer-neutral CUDA-JS capabilities before tensor implementation uses them.

The package owns tensor semantics and planning. It does not own neural architectures, training, autodiff, MCGS, domain encoders, distributed policy, or CUDA provider internals.

## Consequences

- Users who need only CUDA-JS do not pay tensor dependencies or surface area.
- Tensor adapters can evolve and qualify independently.
- Multiple consumers can share one vocabulary without coupling their product policies.
- Cross-repository compatibility must be explicit and exact.
- Initial work is slower because public CUDA-JS gaps cannot be bypassed; this is intentional.

## Alternatives rejected

- Add tensors to CUDA-JS core: rejects runtime/library separation.
- Add tensors to CUDA-MCGS: makes search the permanent first consumer.
- Build in `the_restaurant`: makes one training system the owner and blocks other consumers.
- Direct native addon: violates CUDA-JS ownership and creates a second unsafe runtime.
