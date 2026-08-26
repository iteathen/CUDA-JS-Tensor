# System Registry

| ID | Owner | Current authority |
|---|---|---|
| `tensor.session` | One CUDA-JS runtime/device, tensor resources, program/plan compatibility, terminal close | SPEC-0001 |
| `tensor.spec` | Dtype, rank, capacity shape, active axis-0 bound, strides, layout and byte-range identity | SPEC-0001 |
| `tensor.value` | Opaque session-owned allocation/view capability, exact contiguous copied-byte transfer and lifecycle | SPEC-0001; alpha.3 implementation over CUDA-JS SPEC-0021 at `fb27296c` |
| `tensor.program` | Immutable backend-neutral tensor operation DAG and mathematical semantics | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.plan` | Finite static validation, alias/lifetime/distinct-allocation schedule; resolved backend choices remain TENSOR-SIMT-012-owned | SPEC-0002/SPEC-0004; portable/package implementation |
| `tensor.ops.elementwise` | Typed elementwise, view and cast semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.reduction` | Axes, identity, accumulation, order/determinism and output shape | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.ops.matmul` | Rank-2/rank-3 dense matrix multiplication semantics | SPEC-0002/SPEC-0004; portable/package inference implementation |
| `tensor.resolved-plan` | Session/backend/program/workspace/compiled/prepared compatibility identity | SPEC-0005; alpha.3 implementation candidate |
| `tensor.execution-result` | Per-run material/output-view/workspace ownership and terminal cleanup | SPEC-0005; alpha.3 implementation candidate |
| `tensor.backend.simt` | Complete generated Device-JS baseline | SPEC-0003/SPEC-0005; alpha.3 implementation candidate over CUDA-JS SPEC-0030/SPEC-0020 |
| `tensor.backend.cublaslt` | Host-planned accelerated dense math adapter | SPEC-0003; CUDA-JS SPEC-0023/SPEC-0029 prerequisite available at `fb27296c`; tensor adapter pending |
| `tensor.backend.device-callable` | Device-callable dense subgraph adapter | SPEC-0003; CUDA-JS SPEC-0028 prerequisite available at `fb27296c`; tensor profile pending |
| `tensor.conformance` | Backend equivalence, failure, lifecycle, package and native evidence | SPEC-0000 through SPEC-0005 |

CUDA-JS owns device discovery/selection, contexts, memory, typed device views, compilation/linking, prepared execution, device-callable libraries, CUDA library loading, streams/operations, and native cleanup. This repository consumes only public contracts.

The exact current compatibility pair is `cuda-js-tensor@0.1.0-alpha.3` with `cuda-js@0.1.0-alpha.14` from protected-main revision `fb27296cffd7191180b0e3cd609224ed2ded182e`. Portable/package evidence does not promote CUDA-JS-Tensor native or performance support.
