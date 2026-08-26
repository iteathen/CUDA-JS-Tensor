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

All prerequisites, including dense numeric Device-JS, typed leaf-library composition, selected-runtime target propagation, the native prepared-DAG identity correction and mixed SPEC-0031 node family, are present in exact protected-main CUDA-JS revision `4971302cfb48431c0843126a59d5884d84a81641` (`cuda-js@0.1.0-alpha.16`). This resolves mechanism availability only. Each CUDA-JS-Tensor component still requires its own semantic, lifecycle, native and performance evidence, and no substitute native ownership is permitted.

SPEC-0004 assessment exposed one executor-specific gate: complete SIMT needs consumer-neutral Device-JS f64/f16/bf16 pointer/local arithmetic and exact casts/math. CUDA-JS SPEC-0030 and the alpha.14 prepared-DAG correction now satisfy that gate. SPEC-0005 therefore owns `TENSOR-SIMT-012` without adding tensor vocabulary upstream.

SPEC-0006 consumes SPEC-0031 to replace only eligible rank-2 contiguous f32 matmul execution nodes while preserving the complete lowering, mathematical semantics and one prepared lifecycle. Tensor selection/fallback remains here; provider/native execution remains in CUDA-JS.

SPEC-0007 is Tensor-owned exact elementwise realization: it selects only bounded unobservable single-consumer cast/unary/binary regions, substitutes one generated Device-JS kernel at the region's final position, and removes only internal materials. It consumes existing public CUDA-JS Device-JS and prepared-DAG contracts; it adds no upstream primitive, scheduler, provider policy or native boundary. The complete unfused SPEC-0005 path remains the default and deletion fallback.

SPEC-0009 is the first Tensor-owned device-callable realization. It proves one static item axis, compiles exact dense mathematics as one public CUDA-JS SPEC-0028 leaf library and exposes finite item/input/output/workspace ABI facts. One caller-owned participant invokes one item; Tensor owns no scheduler, queue, launch, scatter, readiness or consumer policy. The complete host-planned path remains independently deletable.
