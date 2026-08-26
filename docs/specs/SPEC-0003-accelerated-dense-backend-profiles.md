# SPEC-0003: Accelerated dense backend profiles

**Status:** Accepted architecture; tensor backends not implemented
**Date:** 2026-08-26

## Complete baseline

Every first-slice operation has a maintained JavaScript/Device-JS SIMT realization or an explicit unsupported disposition. Generated CUDA implementation is private and deterministic. Production `.cu` and `.ptx` files are forbidden.

## Host-planned accelerated profile

The first accelerated host profile maps eligible matrix multiplication nodes to the public CUDA-JS SPEC-0029 cuBLASLt adapter. Algorithm selection, heuristic result, workspace, alignment, precision, epilogue, stream/operation dependency and cleanup are resolved through bounded public records. Heuristics are reusable plan material, not repeated hidden hot-path decisions.

## Device-callable profile

Eligible compact dense subgraphs may use public CUDA-JS SPEC-0028 typed leaf-library composition. cuBLASDx is the primary candidate because it provides selected BLAS operations inside CUDA kernels. It is not presumed usable until an exact CUDA-JS provider profile plus toolkit/license/architecture/dtype/shape/shared-memory/participation and linking gates pass.

CUTLASS is a fallback source-composed candidate when cuBLASDx cannot satisfy the exact contract. It requires an accepted versioned header/library profile and generated specialization through CUDA-JS; CUDA-JS-Tensor does not vendor it casually or add native compilation.

## Selection policy

Acceleration is optional but strongly recommended for an eligible qualified resolved plan. Selection uses static pre-execution facts: operation/shape/layout/dtype/precision, device target, workspace/resource bounds, library compatibility and exact qualification evidence. It is inspectable, overridable and compatibility-identity-affecting.

Small, irregular, latency-sensitive, unsupported or resource-hostile work selects the complete SIMT path. No runtime tensor-core-utilization goal overrides total useful throughput or semantics.

## Qualification

Promotion requires independent mathematical equivalence and representative end-to-end performance against the best credible SIMT baseline. Include planning/heuristic cost, cache reuse, packing, padding, casts, transfers, workspace, synchronization, warmup, small/tail batches, latency distribution, memory high-water, failures and terminal cleanup.

## Primary references

- NVIDIA cuBLAS/cuBLASLt documentation: https://docs.nvidia.com/cuda/cublas/
- NVIDIA cuBLASDx documentation: https://docs.nvidia.com/cuda/cublasdx/
- NVIDIA CUTLASS documentation: https://docs.nvidia.com/cutlass/latest/
