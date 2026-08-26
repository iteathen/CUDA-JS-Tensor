# SPEC-0002: Tensor program, plan and dense operations

**Status:** Accepted
**Date:** 2026-08-26

## Program and plan

`TensorProgram` is an immutable backend-neutral DAG of typed tensor inputs, constants, operations and outputs. It owns mathematical dependency only; it does not own a CUDA stream, library handle, allocation, or native artifact.

`TensorPlan` validates and normalizes shape/dtype/layout inference, broadcast rules, aliasing, liveness, output allocation, workspace upper bounds, legal fusion regions and observable operation boundaries.

`ResolvedTensorPlan` binds one plan to one session/device/CUDA-JS compatibility identity, exact backend choices, generated program identities, resource/workspace limits, precision/tolerance contract, and cleanup graph. Resolution occurs before execution and is immutable. Re-resolution creates a new identity.

## First operations

The first dense slice includes:

- allocation, fill, host/device copy and dtype cast;
- reshape when element order is preserved, transpose/permute views, slice views, broadcast views and contiguous materialization;
- unary and binary elementwise operations with explicit finite broadcast semantics;
- reductions over explicit axes with declared identity, accumulator dtype, deterministic/order policy and output shape;
- rank-2 matrix multiplication;
- rank-3 batched matrix multiplication with explicit batch broadcasting only where accepted.

No convolution, sparse tensor, autodiff, optimizer, random distribution, string tensor, ragged tensor, distributed tensor, or unbounded dynamic shape is implied.

## Semantics and equivalence

Integer overflow, floating rounding, NaN/Infinity, cast, accumulation and reduction order are dtype/operation contract facts. A backend may reassociate or use reduced-precision instructions only under an explicitly selected tolerance/precision profile. Exact control and integer facts never silently adopt approximate semantics.

Fusion cannot erase an observable intermediate requested by the caller, widen aliasing, change failure attribution beyond the declared aggregate, exceed workspace, or weaken cleanup. A complete unfused/SIMT realization remains available.

## Execution

Host mode submits the resolved finite DAG through public CUDA-JS operations/prepared execution. Device-callable mode compiles a bounded eligible subgraph into another device program and requires an accepted public CUDA-JS device-library composition contract. Ineligible nodes fail resolution or form explicit host/device boundaries; no backend callback loop advances device-resident work.
