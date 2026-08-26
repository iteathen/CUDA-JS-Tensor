# System Registry

| ID | Owner | Current authority |
|---|---|---|
| `tensor.session` | One CUDA-JS runtime/device, tensor resources, program/plan compatibility, terminal close | SPEC-0001 |
| `tensor.spec` | Dtype, rank, capacity shape, active axis-0 bound, strides, layout and byte-range identity | SPEC-0001 |
| `tensor.value` | Opaque session-owned allocation/view capability and lifecycle | SPEC-0001; public CUDA-JS typed-view dependency pending |
| `tensor.program` | Immutable backend-neutral tensor operation DAG and mathematical semantics | SPEC-0002 |
| `tensor.plan` | Finite validation, alias/lifetime/workspace schedule and resolved backend choices | SPEC-0002 |
| `tensor.ops.elementwise` | Typed elementwise and cast semantics | SPEC-0002 |
| `tensor.ops.reduction` | Axes, identity, accumulation, order/determinism and output shape | SPEC-0002 |
| `tensor.ops.matmul` | Rank-2/rank-3 dense matrix multiplication semantics | SPEC-0002 |
| `tensor.backend.simt` | Complete generated Device-JS baseline | SPEC-0003 |
| `tensor.backend.cublaslt` | Host-planned accelerated dense math adapter | SPEC-0003; CUDA-JS library-adapter dependency pending |
| `tensor.backend.device-callable` | Device-callable dense subgraph adapter | SPEC-0003; CUDA-JS device-library dependency pending |
| `tensor.conformance` | Backend equivalence, failure, lifecycle, package and native evidence | SPEC-0000 through SPEC-0003 |

CUDA-JS owns device discovery/selection, contexts, memory, typed device views, compilation/linking, prepared execution, device-callable libraries, CUDA library loading, streams/operations, and native cleanup. This repository consumes only public contracts.

