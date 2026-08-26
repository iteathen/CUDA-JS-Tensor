# Dense backend options

**Date:** 2026-08-26  
**Disposition:** Use one complete generated SIMT baseline; plan cuBLASLt for host-submitted dense math; investigate cuBLASDx first for device-callable compact dense subgraphs; retain CUTLASS only as a versioned source-composed fallback.

## Evidence

- NVIDIA documents cuBLASLt as a flexible matmul interface with heuristic algorithm selection. Heuristic selection has host cost, so reusable resolved-plan material is preferable to repeated per-execution queries: https://docs.nvidia.com/cuda/cublas/
- NVIDIA documents cuBLASDx as selected BLAS operations callable inside CUDA kernels, including GEMM and device-function/tensor/shared-memory constraints: https://docs.nvidia.com/cuda/cublasdx/
- NVIDIA documents CUTLASS as modular CUDA C++/DSL building blocks for GEMM and related operations across architecture/data-type/layout policies: https://docs.nvidia.com/cutlass/latest/

## Limits

These libraries do not establish universal support for all shapes, layouts, dtypes, architectures or resource budgets. cuBLASLt heuristic choice is not a deterministic performance guarantee. cuBLASDx device functions are descriptor/profile specific and introduce participation, shared-memory, alignment and link requirements. CUTLASS is a large evolving source/template surface whose version and license must be governed.

## Result

The tensor contract stays mathematical and backend-neutral. A resolved plan selects an adapter only after exact eligibility. Unsupported or unprofitable cases use SIMT. Device-callable acceleration is a core planned feature, but no particular library is mandatory if exact gates or evidence fail.

