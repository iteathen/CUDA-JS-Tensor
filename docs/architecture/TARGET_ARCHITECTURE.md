# Target Architecture

```text
Public convenience
  TensorSession.open(...)
  tensor factories and operation overloads
        |
        v
Canonical semantics
  TensorSpec -> Tensor capability
  TensorProgram -> TensorPlan -> ResolvedTensorPlan
        |
        v
Backend-neutral execution ports
  allocation/view | compile/link | prepared operations | library call | device-callable import
        |
        v
Versioned public CUDA-JS contracts
        |
        v
Private generated/device/provider realization
```

## LEGO ownership

- `tensor.session` owns one runtime/device epoch and all child tensor resources.
- `tensor.spec` owns dtype, capacity shape, optional active extent, strides, layout, offset and byte range.
- `tensor.program` owns immutable mathematical dependencies.
- `tensor.plan` owns validation, aliasing, lifetimes, workspace and backend resolution.
- operation owners define semantics independently of realization.
- backend adapters translate one resolved operation/subgraph to public CUDA-JS ports.
- conformance compares all backends against independent mathematical oracles.

## Ease and completeness

Convenience overloads may infer row-major contiguous strides, allocate outputs, select safe accumulation defaults, and recommend a qualified backend from static pre-execution facts. They cannot infer model meaning, mutate semantics after launch, hide an allocation, exceed declared bounds, silently copy across devices, or create a second runtime.

Every inference appears in `ResolvedTensorPlan` and its compatibility identity. Expert forms bypass no validation and convenience forms create no alternate engine.

## First profile

V1 is one device/session, ranks 0–16, static capacity shapes, optional bounded active axis 0, closed dtypes admitted by CUDA-JS, row-major default, positive/zero strides, no negative strides, dense elementwise/reduction/matmul operations, and finite workspace.

Multi-device use composes independent sessions and independently compiled programs. Cross-session tensor use rejects before device work. Peer access, collectives, sharding and migration require future contracts.

