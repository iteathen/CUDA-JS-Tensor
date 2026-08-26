# CUDA-JS-Tensor

CUDA-JS-Tensor is a consumer-neutral tensor library for CUDA-JS. Its public alpha layers an easy canonical facade over complete expert contracts for finite tensor values, immutable operation DAGs, session-bound resolved plans, complete generated SIMT execution, item-parallel device-callable programs and optional bounded dense acceleration.

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

Public pre-release. `cuda-js-tensor@0.1.0-alpha.6` retains the complete SPEC-0005 generated Device-JS SIMT baseline, SPEC-0006 bounded cuBLASLt policies and SPEC-0007 exact elementwise fusion over exact public `cuda-js@0.1.0-alpha.16` from protected `main@4971302cfb48431c0843126a59d5884d84a81641`. SPEC-0009 adds `compileTensorDeviceProgram(...)`: one caller-owned Device-JS participant executes one statically independent item while many items can run in parallel under the consumer's scheduler. Tensor owns item-axis proof, exact dense mathematics, a finite typed ABI and item-isolated workspace; CUDA-JS owns selected-device target resolution plus library compilation/linking; the consumer owns batching, progress, publication and domain meaning. Independent sessions/programs compose naturally one per selected GPU, while cross-device scheduling remains consumer-owned. The package remains publication-guarded, and no performance, Tensor Core, Linux, broader physical-device qualification or multi-GPU speedup claim is implied.

The first dense slice targets:

- `TensorSession.open(...)` convenience/expert normalization;
- rank 0–16 static capacity shapes with an optional bounded active axis 0;
- row-major default, explicit positive/zero strides, and no negative strides in v1;
- allocation, copy, cast, views, elementwise operations, reductions, matrix multiplication, and batched matrix multiplication;
- immutable `TensorProgram` and finite static `TensorPlan`, followed by backend-owned `ResolvedTensorPlan`;
- a complete generated Device-JS SIMT baseline;
- host-planned cuBLASLt plus an item-parallel device-callable correctness profile through exact accepted public CUDA-JS capabilities;
- optional later cooperative/provider acceleration only after exact participation and end-to-end evidence justify it.

See [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md), [`docs/architecture/TARGET_ARCHITECTURE.md`](docs/architecture/TARGET_ARCHITECTURE.md), and [`docs/plans/2026-08-26-foundation-plan.md`](docs/plans/2026-08-26-foundation-plan.md).

## Development

Node 26.1.0 or later is the initial testing substrate. Read [`AGENTS.md`](AGENTS.md), then run:

```bash
npm run verify
```

CUDA-JS-Tensor is licensed under AGPL-3.0-or-later. Separate commercial terms may be available; see [`LICENSING.md`](LICENSING.md).
