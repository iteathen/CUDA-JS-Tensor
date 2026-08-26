# CUDA-JS-Tensor

CUDA-JS-Tensor is a consumer-neutral tensor library for CUDA-JS. Its public alpha layers an easy canonical facade over complete expert contracts for finite tensor values, immutable operation DAGs, session-bound resolved plans and generated SIMT execution. Qualified accelerated dense-math adapters remain separate bounded successors.

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

Public pre-release. `cuda-js-tensor@0.1.0-alpha.3` is the validated SPEC-0005 candidate over exact public `cuda-js@0.1.0-alpha.14` from protected `main@fb27296cffd7191180b0e3cd609224ed2ded182e`. It retains `TensorSession`/`TensorSpec`/`Tensor`, immutable first-dense `TensorProgram` and finite static `TensorPlan`, and adds public copied-byte seeding/observation plus session-bound `ResolvedTensorPlan`, complete generated Device-JS SIMT lowering, one prepared DAG and per-run `TensorExecutionResult` ownership. Full portable/package validation and an installed public-package Windows CUDA 13.3/compute_75/GTX 1660 Ti mathematical/replay/cleanup fixture pass. The package remains publication-guarded; this evidence does not imply npm publication, another native profile, accelerator, tensor-core, fusion or performance support.

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
