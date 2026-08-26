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

- public typed CUDA-JS device views and dtype identity — implemented by CUDA-JS SPEC-0021;
- typed device-callable function/library composition with resource and participation contracts — implemented by CUDA-JS SPEC-0028 for typed leaf libraries and explicit imports;
- prepared finite operation DAGs — implemented by CUDA-JS SPEC-0020 as semantic single-stream replay, with mixed fixed-plan cuBLASLt nodes added by SPEC-0031;
- context-bound CUDA library adapters and a first cuBLASLt profile — implemented by CUDA-JS SPEC-0023/SPEC-0029.

All prerequisites, including the later dense numeric Device-JS profile, native prepared-DAG identity correction and mixed SPEC-0031 node family, are present in exact protected-main CUDA-JS revision `af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d` (`cuda-js@0.1.0-alpha.15`). This resolves mechanism availability only. Each CUDA-JS-Tensor component still requires its own semantic, lifecycle, native and performance evidence, and no substitute native ownership is permitted.

SPEC-0004 assessment exposed one executor-specific gate: complete SIMT needs consumer-neutral Device-JS f64/f16/bf16 pointer/local arithmetic and exact casts/math. CUDA-JS SPEC-0030 and the alpha.14 prepared-DAG correction now satisfy that gate. SPEC-0005 therefore owns `TENSOR-SIMT-012` without adding tensor vocabulary upstream.

SPEC-0006 consumes SPEC-0031 to replace only eligible rank-2 contiguous f32 matmul execution nodes while preserving the complete lowering, mathematical semantics and one prepared lifecycle. Tensor selection/fallback remains here; provider/native execution remains in CUDA-JS.
