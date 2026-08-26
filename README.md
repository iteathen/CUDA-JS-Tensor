# CUDA-JS-Tensor

CUDA-JS-Tensor is a consumer-neutral tensor library for CUDA-JS. Its first portable value slice provides an easy canonical facade and complete expert contracts for finite tensor shapes, views, session ownership and resource bounds. Operation DAGs, resolved execution plans, generated SIMT kernels, and qualified accelerated dense-math adapters remain the next bounded slices.

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

Public pre-release. `cuda-js-tensor@0.1.0-alpha.1` implements the portable/package `TensorSession`, `TensorSpec`, and opaque `Tensor` value foundation over the exact public `cuda-js@0.1.0-alpha.12` compatibility revision. The package remains publication-guarded; no npm publication, native qualification, dense execution, or performance support claim exists.

The first dense slice targets:

- `TensorSession.open(...)` convenience/expert normalization;
- rank 0–16 static capacity shapes with an optional bounded active axis 0;
- row-major default, explicit positive/zero strides, and no negative strides in v1;
- allocation, copy, cast, views, elementwise operations, reductions, matrix multiplication, and batched matrix multiplication;
- immutable `TensorProgram`, `TensorPlan`, and `ResolvedTensorPlan`;
- a complete generated Device-JS SIMT baseline;
- host-planned cuBLASLt and device-callable dense-subgraph adapters only through the exact accepted public CUDA-JS capabilities and separate tensor qualification.

See [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md), [`docs/architecture/TARGET_ARCHITECTURE.md`](docs/architecture/TARGET_ARCHITECTURE.md), and [`docs/plans/2026-08-26-foundation-plan.md`](docs/plans/2026-08-26-foundation-plan.md).

## Development

Node 26.1.0 or later is the initial testing substrate. Read [`AGENTS.md`](AGENTS.md), then run:

```bash
npm run verify
```

CUDA-JS-Tensor is licensed under AGPL-3.0-or-later. Separate commercial terms may be available; see [`LICENSING.md`](LICENSING.md).
