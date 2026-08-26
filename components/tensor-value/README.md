# Tensor session and value foundation

This component implements accepted `TENSOR-VALUE-010` without owning operation or backend policy.

`TensorSpec` is the immutable canonical value for dtype, capacity/logical shape, bounded active axis 0, element strides, byte offset, alignment, access, layout, and byte-range identity. `TensorSession` binds exactly one CUDA-JS runtime/device epoch, applies explicit limits and documented defaults, and creates opaque `Tensor` capabilities. Root tensors own one CUDA-JS allocation and typed view; child tensor views share that allocation without exposing it.

Convenience and expert forms use the same normalization:

```js
const session = await TensorSession.open();
const easy = await session.allocate([64, 64]);
const explicit = await session.allocate({
  dtype: 'f32',
  capacityShape: [64, 64],
  strides: [64, 1],
  access: 'read-write',
});
```

Defaults are `f32`, `read-write`, row-major contiguous storage, 128 MiB per physical tensor allocation, 256 MiB of session-owned allocation capacity, and 1,024 live tensor capabilities. Overrides are validated, copied, frozen, inspectable, and compatibility-identity-affecting.

`TensorSession.open(runtime)` borrows an injected runtime. `TensorSession.open({ runtime, runtimeOwnership: 'owned' })` explicitly transfers runtime-close authority. No-argument and device-selector forms create and own a compiler-enabled runtime so the complete generated SIMT baseline can later resolve without replacing the session epoch. A borrowed runtime is never closed by the session; whether its compiler is enabled remains inspectable and later plan resolution must reject an unavailable required backend.

V1 required alignment is the dtype width because that is the exact guarantee expressible through the current public CUDA-JS view contract. Larger alignment requests reject instead of assuming a native address. Zero-stride broadcast views are read-only. Empty tensors retain a minimal explicit allocation so their zero-element CUDA-JS view remains a valid child capability.

A changed child view may be derived from a contiguous parent when its exact byte envelope remains inside the parent. A strided parent admits only an identical child specification in this first profile; proving an arbitrary affine child reachable-element set is a subset of another strided set is deliberately not guessed from envelope overlap.

The package-internal `inspectTensorForSession` port is the only future planner/adapter bridge. It rejects cross-session and terminal resources before returning the public CUDA-JS typed view capability inside the package. It is not an installed-package export.

`npm run smoke:native:tensor-value` performs an optional bounded allocation/view/terminal-cleanup smoke on the current host. A pass proves only that lifecycle path on that invocation; it is not native qualification or numerical evidence.
