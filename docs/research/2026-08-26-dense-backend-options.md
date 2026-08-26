# Dense backend options

**Date:** 2026-08-26
**Disposition:** Use one complete generated SIMT baseline; use cuBLASLt only for qualified host-submitted dense math; use SPEC-0009 item-parallel Device-JS as the first device-callable correctness path; investigate cuBLASDx/CUTLASS only for later cooperative intra-item acceleration.

## Evidence

- NVIDIA documents cuBLASLt as a flexible matmul interface with heuristic algorithm selection. Heuristic selection has host cost, so reusable resolved-plan material is preferable to repeated per-execution queries: https://docs.nvidia.com/cuda/cublas/
- NVIDIA documents cuBLASDx as selected BLAS operations callable inside CUDA kernels, including GEMM and device-function/tensor/shared-memory constraints: https://docs.nvidia.com/cuda/cublasdx/
- NVIDIA documents CUTLASS as modular CUDA C++/DSL building blocks for GEMM and related operations across architecture/data-type/layout policies: https://docs.nvidia.com/cutlass/latest/

## Limits

These libraries do not establish universal support for all shapes, layouts, dtypes, architectures or resource budgets. cuBLASLt heuristic choice is not a deterministic performance guarantee. cuBLASDx device functions are descriptor/profile specific and introduce participation, shared-memory, alignment and link requirements. CUTLASS is a large evolving source/template surface whose version and license must be governed.

## Updated result

The tensor contract stays mathematical and backend-neutral. A host-resolved plan selects an adapter only after exact eligibility, while SPEC-0009 compiles an item-independent plan as a CUDA-JS leaf library with one caller participant per item. That supplies real parallel batch participation without requiring cuBLASDx/CUTLASS or guessing a consumer scheduler. Cooperative device-callable acceleration remains a core planned feature, but no particular library is mandatory if exact gates or evidence fail.
