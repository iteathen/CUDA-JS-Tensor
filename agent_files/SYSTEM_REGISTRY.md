# System Registry

| ID | Owner | Current authority |
|---|---|---|
| `tensor.session` | One CUDA-JS runtime/device, tensor resources, program/plan compatibility, terminal close | SPEC-0001 |
| `tensor.spec` | Dtype, rank, capacity shape, active axis-0 bound, strides, layout and byte-range identity | SPEC-0001 |
| `tensor.value` | Opaque session-owned allocation/view capability, exact contiguous copied-byte transfer and lifecycle | SPEC-0001; alpha.4 compatibility over CUDA-JS alpha.15 at `af29b95e` |
| `tensor.program` | Immutable backend-neutral tensor operation DAG and mathematical semantics | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.plan` | Finite static validation, alias/lifetime/distinct-allocation schedule; resolved backend choices remain execution-owned | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.ops.elementwise` | Typed elementwise, view and cast semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.reduction` | Axes, identity, accumulation, order/determinism and output shape | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.matmul` | Rank-2/rank-3 dense matrix multiplication semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.resolved-plan` | Session/backend/program/workspace/provider/compiled/prepared compatibility identity | SPEC-0005/SPEC-0006; alpha.4 integrated at `8910309a` |
| `tensor.execution-result` | Per-run material/output-view/workspace ownership and terminal cleanup | SPEC-0005/SPEC-0006; alpha.4 integrated at `8910309a` |
| `tensor.backend.simt` | Complete generated Device-JS baseline and default/fallback | SPEC-0003/SPEC-0005/SPEC-0006; alpha.4 integrated at `8910309a` over CUDA-JS SPEC-0030/SPEC-0020 |
| `tensor.backend.cublaslt` | Optional rank-2 contiguous f32 matmul selection, fallback, identity and workspace policy | SPEC-0003/SPEC-0006 over CUDA-JS SPEC-0029/SPEC-0031 at `af29b95e`; alpha.4 integrated at `8910309a` |
| `tensor.backend.fusion` | Optional exact elementwise chain selection, generated-kernel substitution and unobservable-intermediate deletion | Planned SPEC-0007/TENSOR-FUSION-017; current CUDA-JS mechanisms sufficient; no production implementation |
| `tensor.memory.arena` | Optional per-run alias-closed material suballocation, range identity and child-before-parent cleanup | Planned SPEC-0008/TENSOR-ARENA-018 after fusion; current CUDA-JS SPEC-0021 mechanisms sufficient; no production implementation |
| `tensor.backend.cublaslt-batched` | Optional rank-3 f32 batched matmul selection and Tensor broadcast mapping | TENSOR-BATCHED-GEMM-015; blocked on public CUDA-JS strided-batched plan/prepared-node child |
| `tensor.backend.cublaslt-precision` | Optional f16/bf16/f64 matmul selection and exact Tensor precision mapping | TENSOR-PRECISION-GEMM-016; blocked on public CUDA-JS typed plan/prepared-node children |
| `tensor.backend.device-callable` | Device-callable dense subgraph ABI, participation, workspace and Tensor publication semantics | SPEC-0003/TENSOR-DEVICE-DENSE-014; CUDA-JS SPEC-0028 composition exists, but efficient collective participation helpers and the Tensor profile remain pending |
| `tensor.conformance` | Backend equivalence, failure, lifecycle, package and native evidence | SPEC-0000 through SPEC-0006 |

CUDA-JS owns device discovery/selection, contexts, memory, typed device views, compilation/linking, prepared execution, device-callable libraries, CUDA library loading, streams/operations, and native cleanup. This repository consumes only public contracts.

The exact current compatibility pair is `cuda-js-tensor@0.1.0-alpha.4` with `cuda-js@0.1.0-alpha.15` from protected-main revision `af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d`. Portable/package evidence does not promote CUDA-JS-Tensor native or performance support.
