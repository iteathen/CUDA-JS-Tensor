# System Registry

| ID | Owner | Current authority |
|---|---|---|
| `tensor.session` | One CUDA-JS runtime/device, tensor resources, program/plan compatibility, terminal close | SPEC-0001 |
| `tensor.spec` | Dtype, rank, capacity shape, active axis-0 bound, strides, layout and byte-range identity | SPEC-0001 |
| `tensor.value` | Opaque session-owned allocation/view capability, exact contiguous copied-byte transfer and lifecycle | SPEC-0001; alpha.6 compatibility over CUDA-JS alpha.16 at `4971302c` |
| `tensor.program` | Immutable backend-neutral tensor operation DAG and mathematical semantics | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.plan` | Finite static validation, alias/lifetime/distinct-allocation schedule; resolved backend choices remain execution-owned | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.ops.elementwise` | Typed elementwise, view and cast semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.reduction` | Axes, identity, accumulation, order/determinism and output shape | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.matmul` | Rank-2/rank-3 dense matrix multiplication semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.resolved-plan` | Session/backend/fusion/program/workspace/provider/compiled/prepared compatibility identity | SPEC-0005/SPEC-0006/SPEC-0007; alpha.5 integrated at `821b1cbe` |
| `tensor.execution-result` | Per-run realized-material/output-view/workspace ownership and terminal cleanup | SPEC-0005/SPEC-0006/SPEC-0007; alpha.5 integrated at `821b1cbe` |
| `tensor.backend.simt` | Complete generated Device-JS baseline and default/fallback | SPEC-0003/SPEC-0005/SPEC-0006/SPEC-0007; alpha.5 integrated at `821b1cbe` over CUDA-JS SPEC-0030/SPEC-0020 |
| `tensor.backend.cublaslt` | Optional rank-2 contiguous f32 matmul selection, fallback, identity and workspace policy | SPEC-0003/SPEC-0006 over CUDA-JS SPEC-0029/SPEC-0031; compatibility pair advanced to alpha.16 at `4971302c` |
| `tensor.backend.fusion` | Optional exact elementwise chain selection, generated-kernel substitution and unobservable-intermediate deletion | SPEC-0007/TENSOR-FUSION-017 integrated at `821b1cbe`; complete unfused path remains the default |
| `tensor.memory.arena` | Optional per-run alias-closed material suballocation, range identity and child-before-parent cleanup | Planned SPEC-0008/TENSOR-ARENA-018 ([issue #17](https://github.com/iteathen/CUDA-JS-Tensor/issues/17)) after fusion; current CUDA-JS SPEC-0021 mechanisms sufficient; no production implementation |
| `tensor.backend.cublaslt-batched` | Optional rank-3 f32 batched matmul selection and Tensor broadcast mapping | TENSOR-BATCHED-GEMM-015; blocked on public CUDA-JS strided-batched plan/prepared-node child |
| `tensor.backend.cublaslt-precision` | Optional f16/bf16/f64 matmul selection and exact Tensor precision mapping | TENSOR-PRECISION-GEMM-016; blocked on public CUDA-JS typed plan/prepared-node children |
| `tensor.backend.device-callable` | Item-axis independence, typed device-function ABI, item-major outputs, dtype-partitioned per-item workspace and copied leaf-library identity | SPEC-0009/TENSOR-DEVICE-ITEM-014 integrated through PR #20 at `9ecc1d78`; issue #19 closed; portable/package and exact Windows native import/call/output/cleanup evidence pass |
| `tensor.conformance` | Backend equivalence, failure, lifecycle, package and native evidence | SPEC-0000 through SPEC-0009 |

CUDA-JS owns device discovery/selection, contexts, memory, typed device views, compilation/linking, prepared execution, device-callable libraries, CUDA library loading, streams/operations, and native cleanup. This repository consumes only public contracts.

The exact current compatibility pair is `cuda-js-tensor@0.1.0-alpha.6` from protected `main@9ecc1d78bca989ec456c897dec215e82ce4cd311` with `cuda-js@0.1.0-alpha.16` from protected `main@4971302cfb48431c0843126a59d5884d84a81641`. CUDA-JS resolves the compile target per selected runtime, so separate Tensor sessions/programs can target separate GPUs without Tensor owning a multi-device coordinator. Exact Windows device-callable evidence qualifies only the recorded correctness/lifecycle cell and does not promote performance, multi-GPU speedup or broader platform support.
