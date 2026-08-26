# CUDA-JS-Tensor

CUDA-JS-Tensor is a consumer-neutral tensor library for CUDA-JS. Its public alpha layers an easy canonical facade over complete expert contracts for finite tensor values, immutable operation DAGs, session-bound resolved plans, complete generated SIMT execution and optional bounded dense acceleration.

The design deliberately separates layers:

```text
convenience overloads and documented defaults
  -> canonical tensor/session/program contracts
    -> finite validated TensorPlan
      -> public CUDA-JS memory/execution/library ports
        -> complete generated SIMT or qualified accelerated adapter
```

The project is not a neural-network framework, training system, CUDA binding, or CUDA-MCGS implementation. Neural inference, MCGS, DSP, simulation, image processing, and other domains are consumers. No first consumer may permanently shape the core.

## Current status

Public pre-release. `cuda-js-tensor@0.1.0-alpha.4` implements SPEC-0006 at protected `main@8910309a0aff9b8da4fc281949068d8d1fcaa6ea` over exact public `cuda-js@0.1.0-alpha.15` from protected `main@af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d`. It retains the complete SPEC-0005 generated Device-JS SIMT baseline and adds explicit `prefer-cublaslt` and strict `cublaslt` policies for eligible rank-2 contiguous f32 matmul. Selected library nodes and generated kernels share one public CUDA-JS prepared DAG; provider, plan, fallback and workspace facts enter immutable resolved identity. `simt` remains the convenience default because no performance promotion has been established. The package remains publication-guarded; exact native evidence is correctness/lifecycle evidence only and does not imply tensor-core use, Linux, broader provider/device support, multi-GPU behavior or a speedup.

The first dense slice targets:

- `TensorSession.open(...)` convenience/expert normalization;
- rank 0–16 static capacity shapes with an optional bounded active axis 0;
- row-major default, explicit positive/zero strides, and no negative strides in v1;
- allocation, copy, cast, views, elementwise operations, reductions, matrix multiplication, and batched matrix multiplication;
- immutable `TensorProgram` and finite static `TensorPlan`, followed by backend-owned `ResolvedTensorPlan`;
- a complete generated Device-JS SIMT baseline;
- host-planned cuBLASLt and device-callable dense-subgraph adapters only through the exact accepted public CUDA-JS capabilities and separate tensor qualification.

See [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md), [`docs/architecture/TARGET_ARCHITECTURE.md`](docs/architecture/TARGET_ARCHITECTURE.md), and [`docs/plans/2026-08-26-foundation-plan.md`](docs/plans/2026-08-26-foundation-plan.md).

## Development

Node 26.1.0 or later is the initial testing substrate. Read [`AGENTS.md`](AGENTS.md), then run:

```bash
npm run verify
```

CUDA-JS-Tensor is licensed under AGPL-3.0-or-later. Separate commercial terms may be available; see [`LICENSING.md`](LICENSING.md).
