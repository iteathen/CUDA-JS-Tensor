# System Registry

| ID | Owner | Current authority |
|---|---|---|
| `tensor.session` | One CUDA-JS runtime/device, tensor resources, program/plan compatibility, terminal close | SPEC-0001 |
| `tensor.spec` | Dtype, rank, capacity shape, active axis-0 bound, strides, layout and byte-range identity | SPEC-0001 |
| `tensor.value` | Opaque session-owned allocation/view capability and lifecycle | SPEC-0001; portable/package implementation over CUDA-JS SPEC-0021 at `2da65ff2` |
| `tensor.program` | Immutable backend-neutral tensor operation DAG and mathematical semantics | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.plan` | Finite static validation, alias/lifetime/distinct-allocation schedule; resolved backend choices remain TENSOR-SIMT-012-owned | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.ops.elementwise` | Typed elementwise, view and cast semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.reduction` | Axes, identity, accumulation, order/determinism and output shape | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.matmul` | Rank-2/rank-3 dense matrix multiplication semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.backend.simt` | Complete generated Device-JS baseline | SPEC-0003; CUDA-JS dense numeric Device-JS profile gap recorded by `CJS-DEVICE-NUMERIC-005` |
| `tensor.backend.cublaslt` | Host-planned accelerated dense math adapter | SPEC-0003; CUDA-JS SPEC-0023/SPEC-0029 prerequisite available at `2da65ff2`; tensor adapter pending |
| `tensor.backend.device-callable` | Device-callable dense subgraph adapter | SPEC-0003; CUDA-JS SPEC-0028 prerequisite available at `2da65ff2`; tensor profile pending |
| `tensor.conformance` | Backend equivalence, failure, lifecycle, package and native evidence | SPEC-0000 through SPEC-0003 |

CUDA-JS owns device discovery/selection, contexts, memory, typed device views, compilation/linking, prepared execution, device-callable libraries, CUDA library loading, streams/operations, and native cleanup. This repository consumes only public contracts.

The exact current compatibility pair is `cuda-js-tensor@0.1.0-alpha.2` with `cuda-js@0.1.0-alpha.12` from protected-main revision `2da65ff2e4287450171c477031dd380a21fa095f`. Portable/package evidence does not promote CUDA-JS-Tensor native or performance support.
