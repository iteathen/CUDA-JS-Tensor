# Tensor sessions and values

This component owns immutable tensor specifications, session limits, allocations, views, copied-byte transfer, and cleanup.

A session binds one CUDA-JS runtime/device. It can create its own runtime or borrow one; a borrowed runtime is not closed by the session. Tensor views share their parent's allocation and retain explicit access and lifetime rules.

Current limits include bounded allocations, restricted strided child views, and contiguous-only host read/write. Empty tensors retain a minimal backing allocation. Dtype conversion and operation/backend policy belong elsewhere.

- [Public interface](index.mjs).
- [Session/value specification](../../docs/specs/SPEC-0001-tensor-session-spec-and-value-model.md): constructors, defaults, views, limits, and lifecycle.
- [Programs](../tensor-program/README.md) and [execution](../tensor-execution/README.md).
- [Project requirements](../../README.md) and [native conformance](../../conformance/native/README.md).

The native smoke checks only its recorded allocation/view/cleanup invocation; it does not establish general numerical or platform support.
