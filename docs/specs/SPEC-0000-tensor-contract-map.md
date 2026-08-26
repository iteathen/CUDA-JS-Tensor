# SPEC-0000: Tensor contract map

**Status:** Accepted
**Date:** 2026-08-26

## Normative dependency order

```text
TensorSession
  -> TensorSpec / Tensor capability
  -> TensorProgram
  -> TensorPlan
  -> ResolvedTensorPlan
  -> selected backend adapter(s)
  -> public CUDA-JS ports
```

## Global requirements

1. Every concrete program and session is finite and has explicit resource bounds.
2. Public objects expose no native CUDA identities or generated implementation.
3. Cross-session/device resources reject before device work.
4. Program semantics do not depend on backend selection.
5. Aliasing, mutability, accumulation, reduction ordering, precision and tolerance are explicit.
6. Convenience APIs normalize to canonical records and preserve exact equivalence.
7. All acquired resources close in dependency order; unproved cleanup remains explicit.
8. Backends may fuse or specialize only when the declared equivalence contract permits it.
9. Unsupported shapes/dtypes/layouts reject or select the complete fallback; they never silently compute different mathematics.
10. A support or performance statement names the exact CUDA-JS/package/backend/environment evidence.

## Upstream dependency gates

- public typed CUDA-JS device views and dtype identity;
- typed device-callable function/library composition with resource and participation contracts;
- prepared finite operation DAGs;
- context-bound CUDA library adapters and a first cuBLASLt profile.

Until these are accepted and implemented, this repository may implement portable value/program semantics but cannot invent substitute native ownership.
