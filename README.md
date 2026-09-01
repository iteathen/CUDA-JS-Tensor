# CUDA-JS-Tensor

[![Verify](https://github.com/iteathen/CUDA-JS-Tensor/actions/workflows/verify.yml/badge.svg)](https://github.com/iteathen/CUDA-JS-Tensor/actions/workflows/verify.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

CUDA-JS-Tensor is a consumer-neutral tensor planning and execution library built on public CUDA-JS contracts.

## Current reality

| Area | Status |
| --- | --- |
| Package identity | `cuda-js-tensor@0.1.0-alpha.6` source/package candidate |
| npm release | **Not published**; `package.json` remains private |
| Production support | **No** — public pre-release development |
| Executable code | Tensor value, program, execution, public facade, and test suites exist in-repo |
| CUDA dependency | Exact pinned CUDA-JS source revision through the package dependency |
| Native/performance support | Limited to exact recorded evidence; no general Linux, Tensor Core, multi-GPU, or speedup claim |

Implemented in the repository today are finite tensor values and views, immutable tensor programs/plans, generated Device-JS SIMT execution, bounded dense-library policies, exact elementwise fusion, and an item-parallel device-callable Tensor program boundary. Exact capability and support status belongs to the owning specifications and [`STATUS.md`](STATUS.md), not to broad README claims.

CUDA-JS-Tensor is **not** a neural-network framework, training system, CUDA binding, CUDA-MCGS implementation, or application scheduler.

## Verify what exists

Requirements: Node.js 26.1.0 or later.

```bash
npm install
npm run verify
```

`npm run verify` runs repository validation plus the Node test suite. Native smoke commands exist for exact supported environments, but a portable/test pass is not a broad native-performance or platform qualification claim.

Current project state and dependency-ready work are tracked in [`STATUS.md`](STATUS.md) and [`next_step.yaml`](next_step.yaml).

## Public boundary

CUDA-JS-Tensor owns:

- tensor shapes, dtypes, strides, active extents, views, aliasing, and finite workspace meaning;
- tensor operation/program/plan semantics;
- generated SIMT realization and selected consumer-neutral dense-operation adapters;
- tensor lifecycle, validation, identity, failure, and cleanup.

CUDA-JS owns CUDA device/runtime/compiler/memory/execution mechanisms. Consumers own model, search, batching, scheduling, publication, and domain semantics.

If Tensor needs a generic CUDA mechanism that public CUDA-JS cannot express naturally, the missing capability is classified upstream rather than implemented through direct FFI, native source, private CUDA-JS imports, or another local escape path.

## Execution shape

```text
consumer
   |
   v
CUDA-JS-Tensor public contracts
   |
   +--> finite TensorPlan / generated Device-JS SIMT
   +--> selected bounded dense adapter
   |
   v
public CUDA-JS contracts
   |
   v
GPU
```

Detailed architecture lives in [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md) and [`docs/architecture/TARGET_ARCHITECTURE.md`](docs/architecture/TARGET_ARCHITECTURE.md).

## Development rule

New operation breadth, acceleration machinery, concurrency, or optimization must be justified by the next executable consumer boundary or measured evidence. Once a tensor boundary is sufficiently specified, prefer a thin executable composition through public CUDA-JS/Tensor contracts over additional speculative layering.

Architecture disposition, implementation status, qualification/support status, and priority remain separate. Missing GPU/host/CI evidence stays an evidence task unless implementation is independently falsified.

## Contributing and security

Read [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing behavior. Use the current [`STATUS.md`](STATUS.md) and [`next_step.yaml`](next_step.yaml) rather than stale README prose to select work.

Report vulnerabilities privately according to [`SECURITY.md`](SECURITY.md), never through a public issue.

CUDA-JS-Tensor is licensed under [AGPL-3.0-or-later](LICENSE). Separate commercial terms may be available; see [`LICENSING.md`](LICENSING.md).
