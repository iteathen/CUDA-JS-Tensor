# Remaining tensor implementation frames

**Status:** Implemented analysis scaffolding; not implementation authority

**Owner direction:** Create real, well-commented incomplete code quickly and preserve deterministic analysis without changing production claims

## Outcome

The remaining known tensor work is split into five independently finishable implementation chunks under [`experiments/tensor-implementation-frames/`](../../experiments/tensor-implementation-frames/). The experiment is outside production components and package exports. Each result is an executable analysis record and cannot affect `ResolvedTensorPlan`.

This frame does not change the implemented alpha.4 support surface. Complete SIMT execution and optional contiguous rank-2 f32 cuBLASLt matmul remain the only production execution profiles.

## Chunk map

| ID | Owned outcome | Real code preserved now | Required before production | Explicit non-goals |
|---|---|---|---|---|
| `TENSOR-DEVICE-DENSE-014` | One finite typed device-callable dense region through public CUDA-JS SPEC-0028 | Maximal connected matmul-anchored region and exact external-boundary discovery | Second-instance value; accepted Tensor callable ABI/resource/participation profile; exact provider/toolkit/license/architecture gates | Elementwise-only fusion, host scheduler, private CUDA-JS imports, mandatory cuBLASDx/CUTLASS |
| `TENSOR-BATCHED-GEMM-015` | Rank-3 contiguous f32 batched matmul in one prepared DAG | Separate SIMT semantic coverage, current backend support, proposed structural eligibility, and capacity batch/m/n/k facts | Public CUDA-JS batched/strided plan and prepared-node contract; workspace/access/failure/cleanup profile | Host loop over rank-2 calls, hidden packing, active batch extent in first profile |
| `TENSOR-PRECISION-GEMM-016` | f16, bf16, and f64 matmul profiles | Dtype/rank/accumulator structural classification without contract-readiness claims | Typed public CUDA-JS plans/nodes; compute/scale/output semantics; numeric equivalence and lifecycle evidence | Implicit reduced precision, Tensor Core claim, default promotion |
| `TENSOR-FUSION-017` | Specialized fused elementwise chains | Conservative single-consumer cast/unary/binary chain discovery | Accepted rounding/observability/failure contract; generated source/access/launch identity; measured value | General graph optimizer, erased public intermediates, fused reductions in first slice |
| `TENSOR-ARENA-018` | Bounded result-owned material reuse | Alias-closed deterministic first-fit candidate arena with explicit alignment and signed byte effect | Accepted suballocation/view ownership; range/rollback/child-view cleanup proof; high-water evidence | Hidden allocation, backend workspace, cross-run arena, cross-session memory, automatic spill |

## Integration order

`TENSOR-DEVICE-DENSE-014` remains the current assessment target because CUDA-JS SPEC-0028 exists, but its concrete second-instance value and Tensor-owned ABI are not yet accepted. The assessment may still defer it.

The other chunks are independent candidates rather than a forced sequence. Start one only when its public CUDA-JS dependency and child specification are ready. Each chunk must replace—not wrap—the corresponding experimental frame, retain the complete SIMT path, and delete its experiment file when its unique continuation value has moved into accepted source and tests.

Performance recommendation is a later evidence gate across completed profiles, not a sixth runtime policy manager. Multi-device tensor composition remains a separate future owner above one-device sessions and is not framed here. Convolution, sparse/ragged tensors, collectives, neural/training layers, package publication, and `the_restaurant` integration remain outside this frame.

## Acceptance and cleanup

No chunk is accepted, exported, natively qualified, performance-qualified, or eligible for a support claim in this task. Each frame now returns executable analysis records and remains useful only while it prevents reconstruction and keeps boundaries visible. A chunk is complete only after accepted authority, production implementation, risk-appropriate focused/full/native evidence, exact-head review, protected integration, and removal or archival of its superseded experiment scaffold.

The analysis frames now have focused portable tests for deterministic identity and immutability, valid and blocked matmul profiles, connected matmul-region boundaries, observable/fan-out/duplicate-edge fusion behavior, alias-closed allocation lifetimes, reuse, and alignment overhead. This evidence qualifies only the disposable analysis behavior; it does not satisfy the native or performance gates and does not promote any frame into production support.
